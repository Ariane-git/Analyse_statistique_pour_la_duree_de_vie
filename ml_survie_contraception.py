#!/usr/bin/env python3
# ============================================================================
# MACHINE LEARNING POUR LA SURVIE
# EDS Cameroun 2018 — Temps avant première contraception moderne
# Méthodes : Cox PH (scikit-survival), Random Survival Forest, SHAP
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, brier_score, integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator

import shap

# ============================================================================
# 1. CHARGEMENT ET PRÉPARATION
# ============================================================================
print("=" * 70)
print("1. CHARGEMENT ET PRÉPARATION")
print("=" * 70)

df = pd.read_csv("donnees_contraception_survie1.csv",
                 sep=";", encoding="latin1")

print(f"Données chargées : {df.shape[0]} femmes × {df.shape[1]} variables")
print(f"Événements       : {df['evenement'].sum()} ({df['evenement'].mean()*100:.1f}%)")
print(f"Censurées        : {(df['evenement']==0).sum()}")

# Variables retenues (toutes numériques, disponibles, NA < 30%)
FEATURES = [
    'groupe_age',           # Groupe d'âge (1-7)
    'statut_matrimonial',   # Statut matrimonial
    'niveau_instruction',   # Niveau d'instruction (0-3)
    'quintile_richesse',    # Quintile de richesse (1-5)
    'milieu_residence',     # Milieu (1=urbain, 2=rural)
    'region',               # Région (1-12)
    'religion_code',        # Code religion
    'ecoute_radio',         # Écoute radio (0/1)
    'vision_television',    # Télévision (0/1)
    'lecture_journaux',     # Journaux (0/1)
    'nombre_enfants_vivants', # Nb enfants vivants
    'nombre_naissances',    # Nb naissances
    'occupation',           # Statut emploi
    'type_union',           # Type d'union
]

LABELS = {
    'groupe_age'              : "Groupe d'âge",
    'statut_matrimonial'      : "Statut matrimonial",
    'niveau_instruction'      : "Niveau instruction",
    'quintile_richesse'       : "Quintile richesse",
    'milieu_residence'        : "Milieu résidence",
    'region'                  : "Région",
    'religion_code'           : "Religion",
    'ecoute_radio'            : "Écoute radio",
    'vision_television'       : "Télévision",
    'lecture_journaux'        : "Journaux",
    'nombre_enfants_vivants'  : "Nb enfants vivants",
    'nombre_naissances'       : "Nb naissances",
    'occupation'              : "Occupation",
    'type_union'              : "Type d'union",
}

df_ml = df[FEATURES + ['evenement', 'temps_survie']].dropna()
print(f"\nAprès suppression NA : {df_ml.shape[0]} femmes")

X = df_ml[FEATURES].astype(float)
y = Surv.from_dataframe('evenement', 'temps_survie', df_ml)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")

# ============================================================================
# 2. MODÈLE 1 — COX PH (scikit-survival)
# ============================================================================
print("\n" + "=" * 70)
print("2. MODÈLE COX PH (SCIKIT-SURVIVAL)")
print("=" * 70)

cox = CoxPHSurvivalAnalysis(alpha=0.1, ties='efron')
cox.fit(X_train, y_train)

c_train_cox = cox.score(X_train, y_train)
c_test_cox  = cox.score(X_test,  y_test)
print(f"C-index train : {c_train_cox:.4f}")
print(f"C-index test  : {c_test_cox:.4f}")

# Coefficients / Hazard Ratios
print("\n--- Hazard Ratios (Cox PH) ---")
coefs = pd.Series(cox.coef_, index=FEATURES).sort_values()
hr    = np.exp(coefs)
print(pd.DataFrame({
    'Coefficient (β)': coefs.values,
    'Hazard Ratio'   : hr.values
}, index=[LABELS[f] for f in coefs.index]).to_string())

# ============================================================================
# 3. MODÈLE 2 — RANDOM SURVIVAL FOREST
# ============================================================================
print("\n" + "=" * 70)
print("3. RANDOM SURVIVAL FOREST (RSF)")
print("=" * 70)

rsf = RandomSurvivalForest(
    n_estimators=30,
    min_samples_split=15,
    min_samples_leaf=8,
    max_depth=6,
    n_jobs=-1,
    random_state=42
)
rsf.fit(X_train, y_train)

c_train_rsf = rsf.score(X_train, y_train)
c_test_rsf  = rsf.score(X_test,  y_test)
print(f"C-index train : {c_train_rsf:.4f}")
print(f"C-index test  : {c_test_rsf:.4f}")

# Importance des variables — permutation importance (feature_importances_ non dispo)
from sklearn.inspection import permutation_importance

print("Calcul de la permutation importance (quelques secondes)...")
perm = permutation_importance(
    rsf, X_test, y_test,
    n_repeats=5,
    random_state=42,
    n_jobs=1      # séquentiel — évite les erreurs DLL Windows
)
importance = pd.Series(perm.importances_mean, index=FEATURES) \
               .sort_values(ascending=False)
print("\n--- Importance des variables (RSF — permutation) ---")
for feat, imp in importance.items():
    print(f"  {LABELS[feat]:<25s} : {imp:.4f}")

# ============================================================================
# 4. COMPARAISON DES MODÈLES
# ============================================================================
print("\n" + "=" * 70)
print("4. COMPARAISON DES MODÈLES")
print("=" * 70)

resultats = pd.DataFrame({
    'Modèle'        : ['Cox PH', 'RSF (30 arbres)'],
    'C-index Train' : [c_train_cox, c_train_rsf],
    'C-index Test'  : [c_test_cox,  c_test_rsf],
    'Surapprentissage' : [c_train_cox - c_test_cox, c_train_rsf - c_test_rsf]
})
print(resultats.to_string(index=False))

# ============================================================================
# 5. VALIDATION CROISÉE (Cox)
# ============================================================================
print("\n" + "=" * 70)
print("5. VALIDATION CROISÉE 5-FOLD (Cox PH)")
print("=" * 70)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
c_scores = []
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx],      y[val_idx]
    m = CoxPHSurvivalAnalysis(alpha=0.1)
    m.fit(X_tr, y_tr)
    c_scores.append(m.score(X_val, y_val))

print(f"C-index par fold : {[round(c, 4) for c in c_scores]}")
print(f"Moyenne ± ÉT     : {np.mean(c_scores):.4f} ± {np.std(c_scores):.4f}")

# ============================================================================
# 6. ANALYSE SHAP — INTERPRÉTABILITÉ DU RSF
# ============================================================================
# ============================================================================
# 6. ANALYSE SHAP (Interprétabilité RSF)
# ============================================================================
print("\n" + "=" * 70)
print("6. ANALYSE SHAP (Interprétabilité RSF)")
print("=" * 70)

import shap

# Fonction de prédiction : retourne le risque cumulatif moyen (scalaire par individu)
def predict_risk(X_array):
    X_df = pd.DataFrame(X_array, columns=FEATURES)
    chf_funcs = rsf.predict_cumulative_hazard_function(X_df)
    # On retourne la valeur finale de la fonction de risque cumulatif
    return np.array([fn.y[-1] for fn in chf_funcs])

# Petit échantillon pour KernelExplainer (lent mais universel)
np.random.seed(42)
idx_bg   = np.random.choice(len(X_train), size=50,  replace=False)
idx_shap = np.random.choice(len(X_test),  size=100, replace=False)

X_background = X_train.iloc[idx_bg].values
X_shap_sample = X_test.iloc[idx_shap]

print(f"Background : {len(X_background)} individus | Échantillon SHAP : {len(X_shap_sample)}")
print("Calcul des valeurs SHAP (peut prendre 2-3 minutes)...")

explainer   = shap.KernelExplainer(predict_risk, X_background)
shap_values = explainer.shap_values(X_shap_sample.values, nsamples=100, silent=True)

print(f"✓ Valeurs SHAP calculées")

# Importance SHAP globale
shap_importance = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=[LABELS[f] for f in FEATURES]
).sort_values(ascending=False)
print("\n--- Importance SHAP (mean |SHAP|) ---")
for feat, val in shap_importance.items():
    print(f"  {feat:<25s} : {val:.4f}")

# Stocker pour les graphiques et la sauvegarde
X_shap = X_shap_sample.copy()

# ============================================================================
# 7. VISUALISATIONS
# ============================================================================
print("\n" + "=" * 70)
print("7. GÉNÉRATION DES GRAPHIQUES")
print("=" * 70)

plt.rcParams.update({'font.size': 10, 'figure.dpi': 120})

# ─── Figure 1 : Comparaison C-index ─────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(7, 4))
modeles = ['Cox PH', 'RSF (30 arb.)']
c_train = [c_train_cox, c_train_rsf]
c_test  = [c_test_cox,  c_test_rsf]
x = np.arange(len(modeles))
w = 0.3
b1 = ax.bar(x - w/2, c_train, w, label='Train', color='#2A9D8F', alpha=0.85)
b2 = ax.bar(x + w/2, c_test,  w, label='Test',  color='#E76F51', alpha=0.85)
ax.set_ylabel('C-index (concordance)')
ax.set_title('Comparaison des modèles de survie ML')
ax.set_xticks(x)
ax.set_xticklabels(modeles)
ax.set_ylim(0.7, 1.0)
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Aléatoire (0.5)')
ax.legend()
ax.bar_label(b1, fmt='%.3f', padding=3)
ax.bar_label(b2, fmt='%.3f', padding=3)
plt.tight_layout()
plt.savefig('ml_comparaison_cindex.png', dpi=150)
plt.close()

# ─── Figure 2 : Importance RSF (permutation) ────────────────────────────────
fig2, ax = plt.subplots(figsize=(8, 5))
# "importance" a été calculé à l'étape 3 via permutation_importance
imp_sorted = pd.Series(importance.values,
                        index=[LABELS[f] for f in importance.index]).sort_values()
colors = ['#E63946' if v == imp_sorted.max() else '#457B9D' for v in imp_sorted]
imp_sorted.plot(kind='barh', ax=ax, color=colors)
ax.set_xlabel('Importance (permutation — baisse du C-index)')
ax.set_title('Importance des variables — Random Survival Forest')
ax.axvline(imp_sorted.mean(), color='orange', linestyle='--', linewidth=1,
           label=f'Moyenne ({imp_sorted.mean():.4f})')
ax.legend()
plt.tight_layout()
plt.savefig('ml_importance_rsf.png', dpi=150)
plt.close()

# ─── Figure 3 : SHAP summary plot ────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(9, 6))
X_shap_labeled = X_shap.copy()
X_shap_labeled.columns = [LABELS[f] for f in FEATURES]
shap.summary_plot(shap_values, X_shap_labeled, plot_type='dot',
                  show=False, max_display=14)
plt.title("SHAP — Importance des variables (RSF)")
plt.tight_layout()
plt.savefig('ml_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── Figure 4 : Courbes de survie prédites par RSF ──────────────────────────
# Profils contrastés
profil_A = pd.DataFrame([[2, 1, 2, 4, 1, 5, 1, 1, 1, 1, 1, 2, 1, 1]],
                         columns=FEATURES)  # Urbain/instruit/riche
profil_B = pd.DataFrame([[5, 1, 0, 1, 2, 3, 3, 0, 0, 0, 4, 5, 0, 1]],
                         columns=FEATURES)  # Rural/non instruit/pauvre

surv_A = rsf.predict_survival_function(profil_A)[0]
surv_B = rsf.predict_survival_function(profil_B)[0]

fig4, ax = plt.subplots(figsize=(9, 5))
ax.step(surv_A.x, surv_A.y, where='post', color='#2A9D8F', linewidth=2,
        label='Profil A : Urbain / Instruit / Riche / Jeune')
ax.step(surv_B.x, surv_B.y, where='post', color='#E63946', linewidth=2,
        label='Profil B : Rural / Non instruit / Pauvre / Âgée')
ax.set_xlabel("Temps depuis premier rapport (années)")
ax.set_ylabel("Probabilité de non-adoption S(t)")
ax.set_title("Courbes de survie prédites — RSF (profils contrastés)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ml_courbes_predites_rsf.png', dpi=150)
plt.close()

print("✓ Graphiques sauvegardés :")
for f in ['ml_comparaison_cindex.png', 'ml_importance_rsf.png',
          'ml_shap_summary.png', 'ml_courbes_predites_rsf.png']:
    print(f"  - {f}")

# ============================================================================
# 8. SAUVEGARDE DES MODÈLES
# ============================================================================
print("\n" + "=" * 70)
print("8. SAUVEGARDE DES MODÈLES")
print("=" * 70)

with open('cox_model.pkl', 'wb') as f:
    pickle.dump({'model': cox, 'features': FEATURES, 'labels': LABELS,
                 'c_index_train': c_train_cox, 'c_index_test': c_test_cox}, f)

with open('rsf_model.pkl', 'wb') as f:
    pickle.dump({'model': rsf, 'features': FEATURES, 'labels': LABELS,
                 'c_index_train': c_train_rsf, 'c_index_test': c_test_rsf,
                 'shap_values': shap_values, 'X_shap': X_shap}, f)

print("✓ cox_model.pkl sauvegardé")
print("✓ rsf_model.pkl sauvegardé")

# ============================================================================
# 9. RÉSUMÉ FINAL
# ============================================================================
print("\n" + "=" * 70)
print("RÉSUMÉ FINAL")
print("=" * 70)
print(f"""
Dataset      : {df_ml.shape[0]} femmes | {int(df_ml['evenement'].sum())} événements
               ({df_ml['evenement'].mean()*100:.1f}% adoptions méthode moderne)

RÉSULTATS ML :
  Cox PH  — C-index test : {c_test_cox:.4f}
  RSF     — C-index test : {c_test_rsf:.4f}

VARIABLES LES PLUS IMPORTANTES (RSF) :
  1. {importance.index[0]:<25s} ({importance.iloc[0]:.4f})
  2. {importance.index[1]:<25s} ({importance.iloc[1]:.4f})
  3. {importance.index[2]:<25s} ({importance.iloc[2]:.4f})

C-index > 0.85 → modèles à bonne capacité discriminante
C-index > 0.70 → acceptable en recherche en santé publique
""")

# ============================================================================
# FIN
# ============================================================================
