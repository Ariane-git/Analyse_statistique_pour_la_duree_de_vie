"""
============================================================================
APPLICATION STREAMLIT — ANALYSE DE SURVIE : CONTRACEPTION MODERNE
EDS Cameroun 2018
Superviseur : Pr. Nguefack Tsague Georges

CHARGEMENT DES MODÈLES :
  Les fichiers cox_model.pkl et rsf_model.pkl doivent être produits par le
  script ml_survie_contraception.py et placés dans le même dossier que ce
  fichier app.py.  Aucun téléchargement GitHub n'est nécessaire.

CHARGEMENT DES DONNÉES :
  Le fichier donnees_contraception_survie1.csv doit se trouver dans le même
  dossier.  Si l'utilisateur ne charge pas de fichier via la sidebar,
  l'application lit automatiquement ce CSV local.
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io, pickle, os
import warnings
warnings.filterwarnings("ignore")

from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================
st.set_page_config(
    page_title="Survie — Contraception Moderne · EDS Cameroun 2018",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Style CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title {
    font-size: 1.8rem; font-weight: 700; color: #1B4F72;
    border-bottom: 3px solid #2A9D8F; padding-bottom: 8px; margin-bottom: 0;
  }
  .subtitle { font-size: 1rem; color: #555; margin-top: 4px; }
  .metric-card {
    background: #f0f4f8; border-radius: 8px; padding: 14px 18px;
    border-left: 4px solid #2A9D8F; margin-bottom: 8px;
  }
  .metric-val { font-size: 1.6rem; font-weight: 700; color: #1B4F72; }
  .metric-lbl { font-size: 0.85rem; color: #666; }
  .section-header {
    background: linear-gradient(90deg,#1B4F72,#2A9D8F);
    color: white; padding: 7px 14px; border-radius: 5px;
    font-weight: 600; margin: 18px 0 10px;
  }
  .info-box {
    background: #e8f4f8; border: 1px solid #a8d8ea;
    border-radius: 6px; padding: 10px 14px; font-size: 0.9rem;
  }
  .success-box {
    background: #e8f8f0; border: 1px solid #a8e6c3;
    border-radius: 6px; padding: 10px 14px; font-size: 0.9rem;
  }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTES
# ============================================================================
FEATURES = [
    'groupe_age', 'statut_matrimonial', 'niveau_instruction',
    'quintile_richesse', 'milieu_residence', 'region', 'religion_code',
    'ecoute_radio', 'vision_television', 'lecture_journaux',
    'nombre_enfants_vivants', 'nombre_naissances', 'occupation', 'type_union'
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

# Chemin du dossier de l'application (même dossier que app.py)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_COX  = os.path.join(APP_DIR, "cox_model.pkl")
PATH_RSF  = os.path.join(APP_DIR, "rsf_model.pkl")
PATH_DATA = os.path.join(APP_DIR, "donnees_contraception_survie1.csv")

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

@st.cache_resource(show_spinner=False)
def charger_modele_local(chemin: str, nom: str) -> dict | None:
    """Charge un modèle .pkl depuis le disque local (même dossier que app.py)."""
    if not os.path.exists(chemin):
        st.error(
            f"❌ Fichier introuvable : **{chemin}**\n\n"
            f"Assurez-vous d'avoir exécuté `ml_survie_contraception.py` pour "
            f"générer `{os.path.basename(chemin)}` dans le même dossier que `app.py`."
        )
        return None
    try:
        with open(chemin, "rb") as f:
            obj = pickle.load(f)
        return obj
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de {nom} : {e}")
        return None


@st.cache_data(show_spinner=False)
def charger_donnees_local() -> pd.DataFrame | None:
    """Charge le CSV local (même dossier que app.py)."""
    if not os.path.exists(PATH_DATA):
        return None
    return pd.read_csv(PATH_DATA, sep=";", encoding="latin1")


@st.cache_data(show_spinner=False)
def charger_donnees_upload(fichier_upload) -> pd.DataFrame:
    """Charge le CSV uploadé via la sidebar."""
    return pd.read_csv(fichier_upload, sep=";", encoding="latin1")


def preparer_donnees(df: pd.DataFrame) -> pd.DataFrame:
    """Recodage des variables catégorielles."""
    df = df.copy()
    df['milieu_f']      = df['milieu_residence'].map({1: "Urbain", 2: "Rural"})
    df['instruction_f'] = df['niveau_instruction'].map(
        {0: "Aucun", 1: "Primaire", 2: "Secondaire", 3: "Supérieur"})
    df['quintile_f']    = df['quintile_richesse'].map(
        {1: "Très pauvre", 2: "Pauvre", 3: "Moyen",
         4: "Riche",       5: "Très riche"})
    df['groupe_age_f']  = df['groupe_age'].map(
        {1: "15-19", 2: "20-24", 3: "25-29", 4: "30-34",
         5: "35-39", 6: "40-44", 7: "45-49"})
    df['radio_f']       = df['ecoute_radio'].map({0: "Non", 1: "Oui"})
    df['tv_f']          = df['vision_television'].map({0: "Non", 1: "Oui"})
    return df


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    return buf.read()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 📊 Navigation")

    onglet_choisi = st.radio(
        "Section",
        ["🏠 Accueil",
         "📈 Kaplan-Meier",
         "⚕️ Modèle de Cox",
         "🤖 Machine Learning",
         "🔮 Prédiction individuelle"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Données (optionnel)**")
    st.markdown(
        "<small>Si votre CSV est déjà dans le dossier de l'application, "
        "aucun upload n'est nécessaire.</small>",
        unsafe_allow_html=True
    )
    fichier = st.file_uploader(
        "Charger un autre CSV (séparateur ;)",
        type=["csv"],
        help="Laissez vide pour utiliser le fichier local donnees_contraception_survie1.csv"
    )

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================
if fichier is not None:
    # Priorité à l'upload utilisateur
    df_raw = charger_donnees_upload(fichier)
    source_donnees = f"CSV uploadé : **{fichier.name}**"
else:
    # Fallback : CSV local dans le dossier de l'app
    df_raw = charger_donnees_local()
    if df_raw is None:
        st.markdown(
            '<p class="main-title">Analyse de survie — Temps avant la première '
            'adoption de contraception moderne</p>'
            '<p class="subtitle">EDS Cameroun 2018 · M1 Statistique · '
            'Institut Saint Jean, Yaoundé</p>',
            unsafe_allow_html=True
        )
        st.error(
            "❌ Aucune donnée trouvée.\n\n"
            f"Placez **donnees_contraception_survie1.csv** dans le même dossier que `app.py` "
            f"(`{APP_DIR}`), ou uploadez votre fichier via la sidebar."
        )
        st.stop()
    source_donnees = f"Fichier local : **donnees_contraception_survie1.csv** ({APP_DIR})"

df      = preparer_donnees(df_raw)
n_total = len(df)
n_evt   = int(df['evenement'].sum())
n_cens  = n_total - n_evt

# ============================================================================
# ONGLET : ACCUEIL
# ============================================================================
if onglet_choisi == "🏠 Accueil":
    st.markdown(
        '<p class="main-title">Analyse de survie — Contraception moderne</p>'
        '<p class="subtitle">EDS Cameroun 2018</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="success-box">✅ Données chargées — {source_donnees}</div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{n_total:,}</div>'
                    f'<div class="metric-lbl">Femmes incluses</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{n_evt:,}</div>'
                    f'<div class="metric-lbl">Événements (adoption moderne)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{n_cens:,}</div>'
                    f'<div class="metric-lbl">Censurées</div></div>', unsafe_allow_html=True)
    with c4:
        taux = round(n_evt / n_total * 100, 1)
        st.markdown(f'<div class="metric-card"><div class="metric-val">{taux}%</div>'
                    f'<div class="metric-lbl">Taux d\'adoption</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Aperçu des données</div>', unsafe_allow_html=True)
    col_affich = ['age_femme', 'age_premier_rapport', 'temps_survie', 'evenement',
                  'milieu_f', 'instruction_f', 'quintile_f', 'religion_label']
    st.dataframe(df[[c for c in col_affich if c in df.columns]].head(10), use_container_width=True)

    st.markdown('<div class="section-header">Distribution du temps de survie</div>', unsafe_allow_html=True)
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df['temps_survie'], bins=40, edgecolor='white',
                color='#2A9D8F', alpha=0.85)
        ax.set_xlabel("Temps de survie (années)")
        ax.set_ylabel("Effectif")
        ax.set_title("Distribution du temps de survie")
        st.pyplot(fig)
        plt.close()
    with col_h2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = ['#2A9D8F', '#E76F51']
        ax.bar(['Adoption\n(événement=1)', 'Censurée\n(événement=0)'],
               [n_evt, n_cens], color=colors, edgecolor='white')
        for i, v in enumerate([n_evt, n_cens]):
            ax.text(i, v + 30, f'{v:,}\n({v/n_total*100:.1f}%)',
                    ha='center', fontsize=9)
        ax.set_title("Répartition événement / censure")
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="info-box">📌 <strong>Variable cible :</strong> '
                '<code>temps_survie</code> = durée (en années) entre le premier rapport '
                'sexuel et l\'adoption d\'une contraception moderne. '
                'L\'événement est l\'adoption (<code>evenement=1</code>). '
                'Les femmes sans contraception moderne au moment de l\'enquête '
                'sont censurées à droite (<code>evenement=0</code>).</div>',
                unsafe_allow_html=True)

# ============================================================================
# ONGLET : KAPLAN-MEIER
# ============================================================================
elif onglet_choisi == "📈 Kaplan-Meier":
    st.markdown('<p class="main-title">Estimateur de Kaplan-Meier</p>', unsafe_allow_html=True)

    col_km, _ = st.columns([1, 2])
    with col_km:
        variable_km = st.selectbox(
            "Comparer par :",
            ["Global", "Milieu de résidence", "Niveau d'instruction",
             "Quintile de richesse", "Groupe d'âge",
             "Écoute radio", "Télévision"]
        )

    mapping_km = {
        "Global"               : (None, None),
        "Milieu de résidence"  : ("milieu_f", ["Urbain", "Rural"]),
        "Niveau d'instruction" : ("instruction_f", ["Aucun", "Primaire", "Secondaire", "Supérieur"]),
        "Quintile de richesse" : ("quintile_f", ["Très pauvre","Pauvre","Moyen","Riche","Très riche"]),
        "Groupe d'âge"         : ("groupe_age_f", ["15-19","20-24","25-29","30-34","35-39","40-44","45-49"]),
        "Écoute radio"         : ("radio_f", ["Non","Oui"]),
        "Télévision"           : ("tv_f", ["Non","Oui"]),
    }
    col_var, groupes = mapping_km[variable_km]

    palette = ['#2A9D8F','#E63946','#457B9D','#F4A261','#264653','#E9C46A','#A8DADC']

    fig, ax = plt.subplots(figsize=(9, 5))

    if col_var is None:
        kmf = KaplanMeierFitter()
        kmf.fit(df['temps_survie'], df['evenement'], label="Global")
        kmf.plot_survival_function(ax=ax, ci_show=True, color='#2A9D8F', linewidth=2)
        t50 = kmf.median_survival_time_
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
        ax.axvline(t50, color='gray', linestyle=':', linewidth=0.8)
        st.success(f"**Temps médian de survie :** {t50:.2f} années")
    else:
        groupes_presents = [g for g in groupes if g in df[col_var].dropna().unique()]
        for i, grp in enumerate(groupes_presents):
            mask = df[col_var] == grp
            kmf = KaplanMeierFitter()
            kmf.fit(df.loc[mask, 'temps_survie'], df.loc[mask, 'evenement'], label=grp)
            kmf.plot_survival_function(ax=ax, ci_show=False,
                                       color=palette[i % len(palette)], linewidth=2)

        if len(groupes_presents) == 2:
            g0 = df[col_var] == groupes_presents[0]
            g1 = df[col_var] == groupes_presents[1]
            lr = logrank_test(
                df.loc[g0, 'temps_survie'], df.loc[g1, 'temps_survie'],
                df.loc[g0, 'evenement'],    df.loc[g1, 'evenement']
            )
            p = lr.p_value
        else:
            lr = multivariate_logrank_test(df['temps_survie'], df[col_var], df['evenement'])
            p = lr.p_value

        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        ax.text(0.98, 0.98, f"Log-rank p = {p:.4f} {sig}",
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Test log-rank p-value", f"{p:.4f}")
        col_r2.metric("Signification", sig)
        col_r3.metric("Groupes comparés", len(groupes_presents))

    ax.set_xlabel("Temps depuis premier rapport (années)", fontsize=11)
    ax.set_ylabel("Probabilité de non-adoption S(t)", fontsize=11)
    ax.set_title(f"Courbe de Kaplan-Meier — {variable_km}", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()

    st.pyplot(fig)
    st.download_button("📥 Télécharger la courbe",
                        data=fig_to_bytes(fig),
                        file_name=f"km_{variable_km.lower().replace(' ', '_')}.png",
                        mime="image/png")
    plt.close()

# ============================================================================
# ONGLET : MODÈLE DE COX
# ============================================================================
elif onglet_choisi == "⚕️ Modèle de Cox":
    st.markdown('<p class="main-title">Modèle de Cox (régression à risques proportionnels)</p>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-header">Ajustement du modèle</div>', unsafe_allow_html=True)

    df_cox = df[[
        'temps_survie', 'evenement', 'milieu_residence', 'niveau_instruction',
        'quintile_richesse', 'groupe_age', 'ecoute_radio', 'vision_television',
        'nombre_enfants_vivants', 'statut_matrimonial', 'religion_code'
    ]].dropna()

    with st.spinner("Ajustement du modèle de Cox..."):
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df_cox, duration_col='temps_survie', event_col='evenement')

    st.success(f"✅ Modèle ajusté sur {len(df_cox):,} femmes | C-index : {cph.concordance_index_:.4f}")

    st.markdown('<div class="section-header">Résumé des coefficients</div>', unsafe_allow_html=True)
    summary = cph.summary.copy()
    summary.columns = [c.replace('coef lower 95%','IC 2.5%').replace('coef upper 95%','IC 97.5%')
                         .replace('exp(coef) lower 95%','HR IC 2.5%').replace('exp(coef) upper 95%','HR IC 97.5%')
                       for c in summary.columns]
    st.dataframe(summary[['coef', 'exp(coef)', 'HR IC 2.5%', 'HR IC 97.5%', 'p']].round(4),
                 use_container_width=True)

    st.markdown('<div class="section-header">Forest Plot — Hazard Ratios</div>', unsafe_allow_html=True)
    fig_fp, ax_fp = plt.subplots(figsize=(8, 6))
    cph.plot(ax=ax_fp)
    ax_fp.set_title("Forest Plot — Modèle de Cox multivarié", fontweight='bold')
    ax_fp.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    st.pyplot(fig_fp)
    st.download_button("📥 Forest Plot",
                        data=fig_to_bytes(fig_fp),
                        file_name="forest_plot_cox.png", mime="image/png")
    plt.close()

    st.markdown('<div class="section-header">Courbes de survie prédites (profils contrastés)</div>',
                unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**Profil A — Urbain / Instruit / Riche**")
        milieu_A = st.selectbox("Milieu A", [1, 2], format_func=lambda x: "Urbain" if x == 1 else "Rural", key="mA")
        instr_A  = st.selectbox("Instruction A", [0,1,2,3], format_func=lambda x: ["Aucun","Primaire","Secondaire","Supérieur"][x], index=2, key="iA")
        quint_A_lbl = st.selectbox("Niveau de richesse A",
                                    ["Très pauvre","Pauvre","Moyen","Riche","Très riche"], index=4, key="qA")
        quint_A = {"Très pauvre":1,"Pauvre":2,"Moyen":3,
                   "Riche":4,"Très riche":5}[quint_A_lbl]
    with col_p2:
        st.markdown("**Profil B — Rural / Non instruit / Pauvre**")
        milieu_B = st.selectbox("Milieu B", [1,2], format_func=lambda x: "Urbain" if x==1 else "Rural", index=1, key="mB")
        instr_B  = st.selectbox("Instruction B", [0,1,2,3], format_func=lambda x: ["Aucun","Primaire","Secondaire","Supérieur"][x], index=0, key="iB")
        quint_B_lbl = st.selectbox("Niveau de richesse B",
                                    ["Très pauvre","Pauvre","Moyen","Riche","Très riche"], index=0, key="qB")
        quint_B = {"Très pauvre":1,"Pauvre":2,"Moyen":3,
                   "Riche":4,"Très riche":5}[quint_B_lbl]

    profils_df = pd.DataFrame({
        'milieu_residence'      : [milieu_A,   milieu_B],
        'niveau_instruction'    : [instr_A,    instr_B],
        'quintile_richesse'     : [quint_A,    quint_B],
        'groupe_age'            : [2,          5],
        'ecoute_radio'          : [1,          0],
        'vision_television'     : [1,          0],
        'nombre_enfants_vivants': [1,          4],
        'statut_matrimonial'    : [1,          1],
        'religion_code'         : [1,          3],
    })

    # predict_survival_function retourne un DataFrame dont les colonnes sont
    # les index (0, 1). On les renomme explicitement avant de tracer.
    NOM_INSTRUCT = ["Aucun", "Primaire", "Secondaire", "Supérieur"]
    NOM_MILIEU   = {1: "Urbain", 2: "Rural"}
    label_A = f"Profil A — {NOM_MILIEU[milieu_A]} / {NOM_INSTRUCT[instr_A]} / {quint_A_lbl}"
    label_B = f"Profil B — {NOM_MILIEU[milieu_B]} / {NOM_INSTRUCT[instr_B]} / {quint_B_lbl}"

    surv_pred = cph.predict_survival_function(profils_df)
    surv_pred.columns = [label_A, label_B]

    fig_pred, ax_pred = plt.subplots(figsize=(9, 5))
    ax_pred.plot(surv_pred.index, surv_pred[label_A],
                 color='#2A9D8F', linewidth=2.5, label=label_A)
    ax_pred.plot(surv_pred.index, surv_pred[label_B],
                 color='#E63946',  linewidth=2.5, label=label_B)

    # Annotation du temps médian pour chaque courbe
    for col, couleur in [(label_A, '#2A9D8F'), (label_B, '#E63946')]:
        s = surv_pred[col]
        if s.min() <= 0.5:
            t50 = float(s.index[s <= 0.5][0])
            ax_pred.axvline(t50, color=couleur, linestyle=':', linewidth=1.2, alpha=0.7)
            ax_pred.text(t50 + 0.4, 0.52, f't₅₀ = {t50:.1f} ans',
                         color=couleur, fontsize=8)

    ax_pred.set_xlabel("Temps depuis le premier rapport sexuel (années)", fontsize=11)
    ax_pred.set_ylabel("Probabilité de non-adoption S(t)", fontsize=11)
    ax_pred.set_title("Courbes de survie prédites — Modèle de Cox", fontsize=12, fontweight='bold')
    ax_pred.legend(fontsize=9, loc='upper right')
    ax_pred.grid(True, alpha=0.25)
    ax_pred.set_ylim(0, 1.05)
    plt.tight_layout()
    st.pyplot(fig_pred)
    st.download_button("📥 Courbes prédites",
                        data=fig_to_bytes(fig_pred),
                        file_name="cox_courbes_predites.png", mime="image/png")
    plt.close()

# ============================================================================
# ONGLET : MACHINE LEARNING
# ============================================================================
elif onglet_choisi == "🤖 Machine Learning":
    st.markdown('<p class="main-title">Machine Learning — Modèles de survie</p>', unsafe_allow_html=True)

    st.markdown(
        '<div class="info-box">ℹ️ Les modèles sont chargés depuis les fichiers <code>cox_model.pkl</code> '
        'et <code>rsf_model.pkl</code> générés par <code>ml_survie_contraception.py</code>. '
        f'Répertoire attendu : <code>{APP_DIR}</code></div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    col_btn1, col_btn2, _ = st.columns([1, 1, 3])

    with col_btn1:
        charger_cox = st.button("📂 Charger modèle Cox", use_container_width=True)
    with col_btn2:
        charger_rsf = st.button("📂 Charger modèle RSF", use_container_width=True)

    # ── Cox ──────────────────────────────────────────────────────────────────
    if charger_cox or st.session_state.get('cox_loaded'):
        cox_pkg = charger_modele_local(PATH_COX, "Cox PH")
        if cox_pkg:
            st.session_state['cox_loaded'] = True
            st.session_state['cox_pkg']    = cox_pkg
            cox_mod = cox_pkg['model']
            c_train = cox_pkg.get('c_index_train', 'N/A')
            c_test  = cox_pkg.get('c_index_test',  'N/A')

            st.markdown('<div class="section-header">Cox PH — Résultats</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("C-index (train)", f"{c_train:.4f}")
            c2.metric("C-index (test)",  f"{c_test:.4f}")

            coefs = pd.Series(cox_mod.coef_, index=FEATURES)
            hr_df = pd.DataFrame({
                'Variable'        : [LABELS[f] for f in FEATURES],
                'β (coefficient)' : coefs.values.round(4),
                'HR'              : np.exp(coefs.values).round(4),
                'Effet'           : ['↑ Risque' if hr > 1 else '↓ Risque'
                                     for hr in np.exp(coefs.values)]
            }).sort_values('HR', ascending=False)
            st.dataframe(hr_df, use_container_width=True)

            # Forest plot
            fig_cox, ax_cox = plt.subplots(figsize=(7, 5))
            hr_sorted = hr_df.sort_values('HR')
            colors = ['#E63946' if hr > 1 else '#2A9D8F' for hr in hr_sorted['HR']]
            ax_cox.barh(hr_sorted['Variable'], np.log(hr_sorted['HR']), color=colors, alpha=0.8)
            ax_cox.axvline(0, color='black', linewidth=1.2)
            ax_cox.set_xlabel("log(HR)")
            ax_cox.set_title("Hazard Ratios — Cox PH (ML)")
            st.pyplot(fig_cox)
            plt.close()

    # ── RSF ──────────────────────────────────────────────────────────────────
    if charger_rsf or st.session_state.get('rsf_loaded'):
        rsf_pkg = charger_modele_local(PATH_RSF, "RSF")
        if rsf_pkg:
            st.session_state['rsf_loaded'] = True
            st.session_state['rsf_pkg']    = rsf_pkg
            rsf_mod = rsf_pkg['model']
            c_train = rsf_pkg.get('c_index_train', 'N/A')
            c_test  = rsf_pkg.get('c_index_test',  'N/A')

            st.markdown('<div class="section-header">RSF — Résultats</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("C-index (train)", f"{c_train:.4f}")
            c2.metric("C-index (test)",  f"{c_test:.4f}")

            # Importance des variables (depuis permutation importance stockée dans le pkl)
            # Le pkl stocke "permutation_importance" si disponible, sinon feature_importances_
            if 'permutation_importance' in rsf_pkg:
                imp = pd.Series(rsf_pkg['permutation_importance'], index=FEATURES).sort_values(ascending=False)
                imp_label = "Permutation importance"
            else:
                # Fallback : feature_importances_ n'existe pas dans sksurv RSF —
                # on recalcule la permutation importance depuis X_shap si disponible
                imp = None

            if imp is not None:
                fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
                imp_labeled = pd.Series(imp.values, index=[LABELS[f] for f in imp.index])
                colors = ['#E63946' if i == 0 else '#457B9D' for i in range(len(imp))]
                imp_labeled.sort_values().plot(kind='barh', ax=ax_imp, color=colors[::-1])
                ax_imp.axvline(imp.mean(), color='orange', linestyle='--',
                               linewidth=1.2, label=f'Moyenne ({imp.mean():.4f})')
                ax_imp.set_xlabel("Importance (baisse du C-index)")
                ax_imp.set_title(f"Importance des variables — Random Survival Forest ({imp_label})")
                ax_imp.legend()
                st.pyplot(fig_imp)
                st.download_button("📥 Graphique importance RSF",
                                    data=fig_to_bytes(fig_imp),
                                    file_name="rsf_importance.png", mime="image/png")
                plt.close()

            # SHAP (si présents dans le pkl)
            if 'shap_values' in rsf_pkg and 'X_shap' in rsf_pkg:
                st.markdown('<div class="section-header">Interprétabilité SHAP</div>', unsafe_allow_html=True)
                import shap
                shap_values = rsf_pkg['shap_values']
                X_shap      = rsf_pkg['X_shap']
                X_shap_lbl  = X_shap.copy()
                X_shap_lbl.columns = [LABELS[f] for f in FEATURES]
                fig_shap, _ = plt.subplots(figsize=(9, 5))
                shap.summary_plot(shap_values, X_shap_lbl, show=False, max_display=14)
                plt.title("SHAP — Impact des variables (RSF)")
                st.pyplot(fig_shap)
                st.download_button("📥 Graphique SHAP",
                                    data=fig_to_bytes(fig_shap),
                                    file_name="shap_summary.png", mime="image/png")
                plt.close()

    if not st.session_state.get('cox_loaded') and not st.session_state.get('rsf_loaded'):
        st.info("Cliquez sur un bouton ci-dessus pour charger et afficher les résultats d'un modèle.")

# ============================================================================
# ONGLET : PRÉDICTION INDIVIDUELLE
# ============================================================================
elif onglet_choisi == "🔮 Prédiction individuelle":
    st.markdown('<p class="main-title">Prédiction individuelle</p>', unsafe_allow_html=True)
    st.markdown("Renseignez le profil d'une femme pour estimer sa probabilité de ne pas encore "
                "avoir adopté une contraception moderne à différents horizons.")

    # Chargement automatique du RSF depuis le fichier local
    rsf_pkg = st.session_state.get('rsf_pkg', None)
    if rsf_pkg is None:
        with st.spinner("Chargement du modèle RSF (fichier local)..."):
            rsf_pkg = charger_modele_local(PATH_RSF, "RSF")
        if rsf_pkg:
            st.session_state['rsf_pkg'] = rsf_pkg

    if rsf_pkg is None:
        st.error(
            f"❌ Modèle RSF introuvable.\n\n"
            f"Exécutez d'abord `ml_survie_contraception.py` pour générer "
            f"`rsf_model.pkl` dans `{APP_DIR}`."
        )
        st.stop()

    rsf_mod = rsf_pkg['model']

    st.markdown('<div class="section-header">Profil individuel</div>', unsafe_allow_html=True)

    # ── Dictionnaires de correspondance label → code ──────────────────────────
    MAP_AGE       = {"15-19 ans":1, "20-24 ans":2, "25-29 ans":3, "30-34 ans":4,
                     "35-39 ans":5, "40-44 ans":6, "45-49 ans":7}
    MAP_INSTRUCT  = {"Aucun":0, "Primaire":1, "Secondaire":2, "Supérieur":3}
    MAP_QUINTILE  = {"Très pauvre":1, "Pauvre":2, "Moyen":3,
                     "Riche":4, "Très riche":5}
    MAP_MILIEU    = {"Urbain":1, "Rural":2}
    MAP_STATUT    = {"Jamais marié(e)":0, "Marié(e) / En couple":1,
                     "En union libre":2, "Veuf / Veuve":3,
                     "Divorcé(e) / Séparé(e)":4, "Non en union":5}
    MAP_RELIGION  = {"Catholique":1, "Protestante":2, "Musulmane":3,
                     "Autre chrétienne":4, "Animiste":5, "Autre / Sans religion":6}
    MAP_REGION    = {"Adamaoua":1, "Centre":2, "Est":3, "Extrême-Nord":4,
                     "Littoral":5, "Nord":6, "Nord-Ouest":7, "Ouest":8,
                     "Sud":9, "Sud-Ouest":10, "Ville de Yaoundé":11,
                     "Ville de Douala":12}
    MAP_UNION     = {"Pas en union":0, "Monogame":1, "Polygame":2, "Non précisé":3}
    MAP_OCC       = {"Sans emploi / Ménagère":0, "Agricultrice":1,
                     "Travail manuel non agricole":2, "Commerce / Vente":3,
                     "Service / Administration":4, "Profession libérale / Cadre":5,
                     "Autre":6}

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**👤 Caractéristiques personnelles**")
        age_lbl   = st.selectbox("Tranche d'âge", list(MAP_AGE.keys()))
        instr_lbl = st.selectbox("Niveau d'instruction", list(MAP_INSTRUCT.keys()))
        relig_lbl = st.selectbox("Religion", list(MAP_RELIGION.keys()))
        region_lbl = st.selectbox("Région de résidence", list(MAP_REGION.keys()))

    with col2:
        st.markdown("**💍 Situation familiale**")
        sm_lbl      = st.selectbox("Statut matrimonial", list(MAP_STATUT.keys()))
        union_lbl   = st.selectbox("Type d'union", list(MAP_UNION.keys()))
        nb_enf      = st.slider("Nombre d'enfants vivants", 0, 10, 2)
        nb_naiss    = st.slider("Nombre de naissances au total", 0, 12, 2)

    with col3:
        st.markdown("**🏘️ Contexte socio-économique**")
        milieu_lbl  = st.selectbox("Milieu de résidence", list(MAP_MILIEU.keys()))
        quint_lbl   = st.selectbox("Niveau de richesse du ménage", list(MAP_QUINTILE.keys()))
        occ_lbl     = st.selectbox("Occupation / Emploi", list(MAP_OCC.keys()))
        radio_lbl   = st.selectbox("Écoute la radio ?", ["Non", "Oui"])
        tv_lbl      = st.selectbox("Regarde la télévision ?", ["Non", "Oui"])
        jrnx_lbl    = st.selectbox("Lit des journaux / magazines ?", ["Non", "Oui"])

    # Conversion des labels vers les codes numériques attendus par le modèle
    ga       = MAP_AGE[age_lbl]
    instr    = MAP_INSTRUCT[instr_lbl]
    quint    = MAP_QUINTILE[quint_lbl]
    milieu   = MAP_MILIEU[milieu_lbl]
    sm       = MAP_STATUT[sm_lbl]
    relig    = MAP_RELIGION[relig_lbl]
    region   = MAP_REGION[region_lbl]
    typ_union = MAP_UNION[union_lbl]
    occ      = MAP_OCC[occ_lbl]
    radio    = 1 if radio_lbl == "Oui" else 0
    tv       = 1 if tv_lbl   == "Oui" else 0
    jrnx     = 1 if jrnx_lbl == "Oui" else 0

    profil_saisie = pd.DataFrame([[
        ga, sm, instr, quint, milieu, region, relig,
        radio, tv, jrnx, nb_enf, nb_naiss, occ, typ_union
    ]], columns=FEATURES)

    if st.button("🔮 Calculer la prédiction", type="primary", use_container_width=True):
        surv_fn = rsf_mod.predict_survival_function(profil_saisie)[0]

        # ── Limite temporelle selon la tranche d'âge ─────────────────────────
        age_min_par_groupe = {1:15, 2:20, 3:25, 4:30, 5:35, 6:40, 7:45}
        age_min = age_min_par_groupe[ga]
        t_max   = 49 - age_min

        # Tronquer la courbe au temps max plausible
        mask   = surv_fn.x <= t_max
        times_ = surv_fn.x[mask]
        survs_ = surv_fn.y[mask]

        # ── Temps médian prédit ───────────────────────────────────────────────
        # C'est le t pour lequel S(t) <= 0.5 (50% des femmes similaires
        # ont adopté avant ce délai)
        idx50 = np.searchsorted(-survs_, -0.5)  # premier t où S(t) <= 0.5
        if idx50 < len(times_) and survs_[idx50] <= 0.5:
            t_median = float(times_[idx50])
            age_median = age_min + t_median
            median_txt  = f"{t_median:.1f} ans après le premier rapport"
            age_txt     = f"soit vers {age_median:.0f} ans"
            median_atteint = True
        else:
            t_median = None
            median_txt  = "Non atteint dans la période d'observation"
            age_txt     = f"(S(t) reste > 50% jusqu'à {t_max} ans)"
            median_atteint = False

        # ── Quantiles 25% et 75% ─────────────────────────────────────────────
        def get_quantile(q):
            idx_q = np.searchsorted(-survs_, -(1-q))
            if idx_q < len(times_) and survs_[idx_q] <= (1-q):
                return float(times_[idx_q])
            return None

        t_q25 = get_quantile(0.25)  # 25% ont adopté avant t_q25
        t_q75 = get_quantile(0.75)  # 75% ont adopté avant t_q75

        # ── Probabilité S(t) à quelques horizons clés ─────────────────────────
        horizons_tous = [5, 10, 15, 20, 25, 30]
        horizons = [h for h in horizons_tous if h <= t_max]
        probs_non = {}
        for h in horizons:
            i_h = min(np.searchsorted(times_, h), len(survs_) - 1)
            probs_non[h] = float(survs_[i_h])

        # ── Affichage ─────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Délai prédit avant adoption</div>',
                    unsafe_allow_html=True)

        st.info(
            "**Comment lire ces résultats :** Le modèle estime le **délai attendu "
            "(en années depuis le premier rapport sexuel)** avant qu'une femme ayant "
            "ce profil adopte une contraception moderne. "
            f"L'horizon maximal est **{t_max} ans** (tranche {age_lbl}, "
            f"âge min = {age_min} ans, fin période reproductive = 49 ans)."
        )

        # Métriques principales : délais
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "⏱️ Délai médian d'adoption",
            median_txt,
            age_txt
        )
        c2.metric(
            "⏱️ Délai Q25 (adoption précoce)",
            f"{t_q25:.1f} ans" if t_q25 else "Non atteint",
            "25% adoptent avant ce délai"
        )
        c3.metric(
            "⏱️ Délai Q75 (adoption tardive)",
            f"{t_q75:.1f} ans" if t_q75 else "Non atteint",
            "75% adoptent avant ce délai"
        )

        # Tableau des probabilités de non-adoption aux horizons clés
        if horizons:
            st.markdown("**Probabilité de ne pas encore avoir adopté à chaque horizon :**")
            cols_h = st.columns(len(horizons))
            for i, h in enumerate(horizons):
                s_h = probs_non[h]
                cols_h[i].metric(
                    f"À {h} ans",
                    f"S(t) = {s_h*100:.0f}%",
                    f"{(1-s_h)*100:.0f}% ont adopté"
                )

        # ── Courbe de survie S(t) ─────────────────────────────────────────────
        fig_ind, ax_ind = plt.subplots(figsize=(9, 5))

        ax_ind.step(times_, survs_, where='post',
                    color='#2A9D8F', linewidth=2.5, label='S(t) — non-adoption')
        ax_ind.fill_between(times_, survs_, step='post',
                            alpha=0.12, color='#2A9D8F')

        # Ligne médiane
        if median_atteint:
            ax_ind.axhline(0.5, color='gray', linestyle='--', linewidth=1,
                           alpha=0.7, label='Seuil 50%')
            ax_ind.axvline(t_median, color='#E63946', linestyle='--',
                           linewidth=1.5, label=f'Délai médian = {t_median:.1f} ans')
            ax_ind.annotate(
                f"Délai médian\n{t_median:.1f} ans\n(~{age_median:.0f} ans)",
                xy=(t_median, 0.5),
                xytext=(t_median + 1, 0.62),
                fontsize=9, color='#E63946',
                arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.5)
            )

        # Points aux horizons
        for h in horizons:
            i_h = min(np.searchsorted(times_, h), len(survs_) - 1)
            ax_ind.plot(h, survs_[i_h], 'o', color='#457B9D',
                        markersize=6, zorder=5)

        ax_ind.set_xlabel("Années depuis le premier rapport sexuel", fontsize=11)
        ax_ind.set_ylabel("Probabilité de non-adoption S(t)", fontsize=11)
        ax_ind.set_title(
            f"Délai prédit avant adoption — Profil : {age_lbl} / {instr_lbl} / "
            f"{milieu_lbl} / {quint_lbl}",
            fontsize=11, fontweight='bold'
        )
        ax_ind.set_xlim(0, t_max + 0.5)
        ax_ind.set_ylim(0, 1.05)
        ax_ind.legend(fontsize=9, loc='upper right')
        ax_ind.grid(True, alpha=0.25)
        plt.tight_layout()

        st.pyplot(fig_ind)
        st.download_button("📥 Télécharger la courbe",
                            data=fig_to_bytes(fig_ind),
                            file_name="delai_adoption_predit.png",
                            mime="image/png")
        plt.close()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    "<center><small>EDS Cameroun 2018 · M1 Statistique & Analyse des données · "
    "Institut Saint Jean, Yaoundé · Superviseur : Pr. Nguefack Tsague Georges</small></center>",
    unsafe_allow_html=True
)
