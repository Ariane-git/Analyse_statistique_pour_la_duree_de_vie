import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle, os
from scipy import stats

# ── Configuration ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Contraception au Cameroun — EDS 2018",
    page_icon="🇨🇲",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Palette ───────────────────────────────────────────────────────
C3 = ["#E74C3C", "#F39C12", "#27AE60"]
COULEURS = {"Aucune":"#E74C3C","Traditionnelle":"#F39C12","Moderne":"#27AE60"}
EMOJIS   = {"Aucune":"⛔","Traditionnelle":"🌿","Moderne":"💊"}

# ════════════════════════════════════════════════════════════════
# CHARGEMENT
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def charger_modele():
    with open(os.path.join(BASE_DIR, "model_final.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "feature_cols.pkl"), "rb") as f:
        features = pickle.load(f)
    with open(os.path.join(BASE_DIR, "classes.pkl"), "rb") as f:
        classes = pickle.load(f)
    return model, features, classes

@st.cache_data
def charger_donnees():
    base = pd.read_csv(os.path.join(BASE_DIR, "base_eds2018.csv"))
    cats = {
        "type_methode": (["Aucune","Traditionnelle","Moderne"], False),
        "age_grp":      (["15-19","20-24","25-29","30-34",
                          "35-39","40-44","45-49"], True),
        "instruction":  (["Aucun","Primaire","Secondaire","Superieur"], True),
        "milieu":       (["Rural","Urbain"], False),
        "richesse":     (["Tres pauvre","Pauvre","Moyen",
                          "Riche","Tres riche"], True),
        "religion":     (["Catholique","Protestante","Autres chretiens",
                          "Musulmane","Animiste","Sans religion"], False),
        "statut_mat":   (["En union","Jamais en union",
                          "Anciennement en union"], False),
        "parite":       (["Nullipare","Non nullipare"], False),
        "expo_media":   (["Non exposee","Exposee"], False),
    }
    for col, (cats_list, ordered) in cats.items():
        if col in base.columns:
            base[col] = pd.Categorical(
                base[col], categories=cats_list, ordered=ordered)
    base["utilise"] = (base["type_methode"] != "Aucune").astype(int)
    return base

@st.cache_data
def charger_resultats():
    or_mod  = pd.read_csv(os.path.join(BASE_DIR, "or_moderne.csv"),
                          index_col=0)
    or_trad = pd.read_csv(os.path.join(BASE_DIR, "or_traditionnelle.csv"),
                          index_col=0)
    return or_mod, or_trad

model, feature_cols, classes = charger_modele()
base = charger_donnees()
or_moderne, or_trad = charger_resultats()

# ════════════════════════════════════════════════════════════════
# SIDEBAR — NAVIGATION (2 pages)
# ════════════════════════════════════════════════════════════════
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/4/4f/Flag_of_Cameroon.svg",
    width=72
)
st.sidebar.title("EDS Cameroun 2018")
st.sidebar.caption("Analyse de la contraception")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊  Statistiques & Résultats",
     "🎯  Prédiction"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Base :** {len(base):,} femmes  \n"
    f"**Source :** EDS Cameroun 2018  \n"
    f"**Cours :** MAD4017/MAD4037"
)

# ════════════════════════════════════════════════════════════════
# PAGE 1 — STATISTIQUES & RÉSULTATS
# ════════════════════════════════════════════════════════════════
if page == "📊  Statistiques & Résultats":

    st.title("📊 Statistiques & Résultats")
    st.markdown(
        "Analyse multivariée des déterminants de l'utilisation "
        "contraceptive au Cameroun — EDS 2018."
    )

    # Métriques globales
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Effectif analysé", f"{len(base):,}")
    c2.metric("Aucune méthode",
              f"{(base['type_methode']=='Aucune').sum():,}",
              f"{(base['type_methode']=='Aucune').mean()*100:.1f}%")
    c3.metric("Méthode moderne",
              f"{(base['type_methode']=='Moderne').sum():,}",
              f"{(base['type_methode']=='Moderne').mean()*100:.1f}%")
    c4.metric("Méthode traditionnelle",
              f"{(base['type_methode']=='Traditionnelle').sum():,}",
              f"{(base['type_methode']=='Traditionnelle').mean()*100:.1f}%")

    st.markdown("---")

    # ── 4 onglets ─────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs([
        "🔢 Analyse descriptive",
        "🔗 Analyse bivariée",
        "📈 Modélisation statistique",
        "🤖 Machine Learning"
    ])

    # ── ONGLET 1 : DESCRIPTIF ─────────────────────────────────
    with t1:
        st.subheader("Distribution des variables")

        # Camembert V313
        col1, col2 = st.columns([1, 1])
        with col1:
            vc  = base["type_methode"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(5, 5))
            wedges, texts, autotexts = ax.pie(
                vc.values,
                labels=vc.index,
                autopct="%1.1f%%",
                colors=C3,
                startangle=90,
                wedgeprops={"edgecolor":"white","linewidth":2}
            )
            for at in autotexts:
                at.set_fontsize(11)
                at.set_fontweight("bold")
                at.set_color("white")
            ax.set_title("Type de méthode contraceptive (V313)",
                         fontweight="bold", pad=15)
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("**Effectifs par modalité**")
            pct = (vc / len(base) * 100).round(1)
            df_vc = pd.DataFrame({"Effectif": vc, "%": pct})
            st.dataframe(df_vc, use_container_width=True)
            st.markdown(" ")
            st.markdown("**Distribution par milieu de résidence**")
            ct_m = (pd.crosstab(base["milieu"],
                                base["type_methode"],
                                normalize="index") * 100).round(1)
            st.dataframe(ct_m, use_container_width=True)

        st.markdown("---")
        st.markdown("**Distribution par variable explicative**")

        var_sel = st.selectbox(
            "Choisir une variable",
            ["Niveau d'instruction","Groupe d'âge","Indice de richesse",
             "Religion","Statut matrimonial","Exposition aux médias",
             "Milieu de résidence","Parité"],
            key="desc_var"
        )
        var_map_desc = {
            "Niveau d'instruction":  "instruction",
            "Groupe d'âge":          "age_grp",
            "Indice de richesse":    "richesse",
            "Religion":              "religion",
            "Statut matrimonial":    "statut_mat",
            "Exposition aux médias": "expo_media",
            "Milieu de résidence":   "milieu",
            "Parité":                "parite",
        }
        col_d = var_map_desc[var_sel]
        ct_d  = (pd.crosstab(base[col_d], base["type_methode"],
                              normalize="index") * 100)

        fig, ax = plt.subplots(figsize=(9, 4))
        ct_d.plot(kind="bar", ax=ax, color=C3,
                  edgecolor="black", width=0.7)
        ax.set_title(f"Utilisation contraceptive par {var_sel}",
                     fontweight="bold")
        ax.set_ylabel("Pourcentage (%)")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=20, ha="right")
        ax.legend(title="Méthode", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_facecolor("#F8F9FA")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── ONGLET 2 : BIVARIÉE ────────────────────────────────────
    with t2:
        st.subheader("Analyse bivariée — Tests du Chi²")

        explicatives = {
            "Niveau d'instruction":  "instruction",
            "Groupe d'âge":          "age_grp",
            "Milieu de résidence":   "milieu",
            "Indice de richesse":    "richesse",
            "Religion":              "religion",
            "Statut matrimonial":    "statut_mat",
            "Parité":                "parite",
            "Exposition aux médias": "expo_media",
        }

        recap_chi2 = []
        for nom, col in explicatives.items():
            tab = pd.crosstab(base[col], base["type_methode"])
            chi2, p, ddl, _ = stats.chi2_contingency(tab)
            sig = ("***" if p<0.001 else "**" if p<0.01
                   else "*" if p<0.05 else "ns")
            recap_chi2.append({
                "Variable": nom,
                "Chi²":     round(chi2, 2),
                "Ddl":      ddl,
                "p-valeur": f"{p:.2e}",
                "Sig.":     sig
            })

        df_chi2 = pd.DataFrame(recap_chi2).sort_values(
            "Chi²", ascending=False)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Récapitulatif des tests du Chi²**")
            st.dataframe(df_chi2, use_container_width=True,
                         hide_index=True)
            st.caption("*** p<0,001  ** p<0,01  * p<0,05")

        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(df_chi2["Variable"][::-1],
                    df_chi2["Chi²"][::-1],
                    color="#065A82", edgecolor="black", alpha=0.85)
            ax.set_xlabel("Valeur du Chi²")
            ax.set_title("Chi² par variable (p<0,001 pour toutes)",
                         fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            ax.set_facecolor("#F8F9FA")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        var_biv = st.selectbox(
            "Tableau croisé détaillé",
            list(explicatives.keys()), key="biv_var"
        )
        col_biv = explicatives[var_biv]
        tab_pct = (pd.crosstab(base[col_biv],
                               base["type_methode"],
                               normalize="index") * 100).round(1)
        tab_eff = pd.crosstab(base[col_biv], base["type_methode"])
        chi2_d, p_d, _, _ = stats.chi2_contingency(tab_eff)
        sig_d = ("***" if p_d<0.001 else "**" if p_d<0.01
                 else "*" if p_d<0.05 else "ns")

        st.markdown(f"**Pourcentages en ligne (%) — {var_biv}**")
        st.dataframe(tab_pct, use_container_width=True)
        st.markdown(
            f"Chi² = **{chi2_d:.2f}** | "
            f"p = **{p_d:.2e}** | {sig_d}"
        )

    # ── ONGLET 3 : MODÉLISATION ────────────────────────────────
    with t3:
        st.subheader("Modèle logistique multinomial — M4 (modèle final)")

        col1, col2, col3 = st.columns(3)
        col1.metric("AIC", "38 297,5")
        col2.metric("McFadden R²", "0,095")
        col3.metric("Observations", f"{len(base):,}")

        st.markdown("---")

        categorie = st.radio(
            "Équation à afficher",
            ["Moderne vs Aucune", "Traditionnelle vs Aucune"],
            horizontal=True
        )
        or_tab = (or_moderne if categorie == "Moderne vs Aucune"
                  else or_trad)
        or_tab = or_tab.drop("const", errors="ignore")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"**Tableau des Odds Ratios — {categorie}**")
            st.dataframe(or_tab, use_container_width=True)
            st.caption(
                "OR > 1 = facteur favorisant · "
                "OR < 1 = facteur frein  "
                "· *** p<0,001 ** p<0,01 * p<0,05"
            )

        with col2:
            st.markdown("**Forest plot — variables significatives**")
            or_sig = or_tab[or_tab["Sig."] != "ns"].copy()
            or_sig = or_sig.sort_values("OR", ascending=True)
            noms = {
                "age_grp_20-24":"Âge 20-24","age_grp_25-29":"Âge 25-29",
                "age_grp_30-34":"Âge 30-34","age_grp_35-39":"Âge 35-39",
                "age_grp_40-44":"Âge 40-44","age_grp_45-49":"Âge 45-49",
                "statut_mat_Jamais en union":"Jamais en union",
                "statut_mat_Anciennement en union":"Anc. en union",
                "parite_Non nullipare":"Non nullipare",
                "instruction_Primaire":"Instr. Primaire",
                "instruction_Secondaire":"Instr. Secondaire",
                "instruction_Superieur":"Instr. Supérieur",
                "richesse_Pauvre":"Richesse Pauvre",
                "richesse_Moyen":"Richesse Moyen",
                "richesse_Riche":"Richesse Riche",
                "richesse_Tres riche":"Richesse T. Riche",
                "milieu_Urbain":"Urbain",
                "religion_Protestante":"Protestante",
                "religion_Autres chretiens":"Autres chrét.",
                "religion_Musulmane":"Musulmane",
                "religion_Animiste":"Animiste",
                "religion_Sans religion":"Sans religion",
                "expo_media_Exposee":"Exposée médias",
            }
            or_sig.index = [noms.get(i, i) for i in or_sig.index]

            if len(or_sig) > 0:
                fig, ax = plt.subplots(
                    figsize=(6, max(3, len(or_sig)*0.38+1)))
                y_pos  = range(len(or_sig))
                colors = ["#E74C3C" if v<1 else "#27AE60"
                          for v in or_sig["OR"]]
                ax.hlines(list(y_pos), or_sig["IC 2.5%"],
                          or_sig["IC 97.5%"],
                          color=colors, linewidth=2, alpha=0.7)
                ax.scatter(or_sig["OR"], list(y_pos),
                           color=colors, s=50, zorder=5)
                ax.axvline(1, color="black", linestyle="--",
                           linewidth=0.8)
                for i in range(len(or_sig)):
                    ax.text(or_sig["IC 97.5%"].iloc[i]+0.05, i,
                            f"{or_sig['OR'].iloc[i]:.2f}",
                            va="center", fontsize=8)
                ax.set_yticks(list(y_pos))
                ax.set_yticklabels(or_sig.index, fontsize=8)
                ax.set_xlabel("Odds Ratio (IC 95%)")
                ax.grid(axis="x", alpha=0.3)
                ax.set_facecolor("#F8F9FA")
                p1 = mpatches.Patch(color="#27AE60",
                                    label="OR > 1 (favorisant)")
                p2 = mpatches.Patch(color="#E74C3C",
                                    label="OR < 1 (frein)")
                ax.legend(handles=[p1,p2], fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    # ── ONGLET 4 : MACHINE LEARNING ───────────────────────────
    with t4:
        st.subheader("Machine Learning — Performances et importance")

        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("GB Balanced Acc. (multi)", "voir notebook")
        col2.metric("RF Balanced Acc. (multi)", "voir notebook")
        col3.metric("GB AUC-ROC (binaire)",     "voir notebook")
        col4.metric("RF AUC-ROC (binaire)",     "voir notebook")

        st.info(
            "Les métriques exactes sont calculées dans le notebook "
            "lors de l'exécution (CV-5 fold). Le modèle retenu est "
            "**Gradient Boosting multinomial corrigé** (meilleure "
            "Balanced Accuracy sur données EDS 2018)."
        )

        st.markdown("---")

        # Importance des variables depuis le modèle chargé
        X_ml = pd.get_dummies(
            base[["age_grp","statut_mat","parite","instruction",
                  "richesse","milieu","religion","expo_media"]],
            drop_first=False
        ).astype(float)
        X_aligned = X_ml.reindex(columns=feature_cols, fill_value=0.0)

        fi = pd.DataFrame({
            "Variable":   feature_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        col1, col2 = st.columns([1, 1])

        with col1:
            n_top = st.slider("Nombre de variables à afficher",
                              5, 20, 12)
            fi_plot = fi.head(n_top)
            fig, ax = plt.subplots(figsize=(6, n_top*0.4+1))
            ax.barh(fi_plot["Variable"][::-1],
                    fi_plot["Importance"][::-1],
                    color="#065A82", edgecolor="black", alpha=0.85)
            ax.set_xlabel("Importance (Gini)")
            ax.set_title(f"Top {n_top} variables — Modèle retenu",
                         fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            ax.set_facecolor("#F8F9FA")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("**Tableau complet de l'importance**")
            st.dataframe(fi.head(n_top), use_container_width=True,
                         hide_index=True)

        st.markdown("---")
        st.markdown("### Rappel méthodologique")
        st.markdown("""
        | Choix | Justification |
        |-------|--------------|
        | Pas de standardisation | Features = dummies (0/1), même échelle · RF/GB invariants à l'échelle |
        | Pas de SMOTE | Effectif Traditionnelle = 1 333 (suffisant) · `sample_weight` suffit |
        | Balanced Accuracy | Métrique adaptée au déséquilibre (78,5% / 3,9% / 17,6%) |
        | Validation CV-5 | Évite le surapprentissage, stratification préserve les proportions |
        | GB retenu | Meilleure Balanced Accuracy sur données EDS 2018 |
        """)

# ════════════════════════════════════════════════════════════════
# PAGE 2 — PRÉDICTION
# ════════════════════════════════════════════════════════════════
elif page == "🎯  Prédiction":

    st.title("🎯 Prédiction du type de méthode contraceptive")
    st.markdown(
        "Renseignez les caractéristiques d'une femme. "
        "Le modèle **Gradient Boosting** prédit le type de méthode "
        "contraceptive parmi : **⛔ Aucune · 🌿 Traditionnelle · 💊 Moderne**"
    )
    st.markdown("---")

    col_form, col_res = st.columns([1, 1], gap="large")

    # ── Formulaire ────────────────────────────────────────────
    with col_form:
        st.subheader("Profil de la femme")

        age = st.selectbox(
            "Groupe d'âge",
            ["15-19","20-24","25-29","30-34",
             "35-39","40-44","45-49"]
        )
        instruction = st.selectbox(
            "Niveau d'instruction",
            ["Aucun","Primaire","Secondaire","Superieur"]
        )
        richesse = st.selectbox(
            "Indice de richesse",
            ["Tres pauvre","Pauvre","Moyen","Riche","Tres riche"]
        )
        milieu = st.selectbox(
            "Milieu de résidence",
            ["Rural","Urbain"]
        )
        religion = st.selectbox(
            "Religion",
            ["Catholique","Protestante","Autres chretiens",
             "Musulmane","Animiste","Sans religion"]
        )
        statut = st.selectbox(
            "Statut matrimonial",
            ["En union","Jamais en union","Anciennement en union"]
        )
        parite = st.selectbox(
            "Parité",
            ["Nullipare","Non nullipare"],
            help="Nullipare = aucun enfant vivant"
        )
        media = st.selectbox(
            "Exposition aux médias",
            ["Non exposee","Exposee"],
            help="Exposée = radio / TV / journal ≥ 1×/semaine"
        )

        btn = st.button(
            "🔍 Prédire", type="primary",
            use_container_width=True
        )

    # ── Résultat ──────────────────────────────────────────────
    with col_res:
        st.subheader("Résultat")

        if btn:
            params = {
                "age_grp":     age,
                "statut_mat":  statut,
                "parite":      parite,
                "instruction": instruction,
                "richesse":    richesse,
                "milieu":      milieu,
                "religion":    religion,
                "expo_media":  media,
            }

            # Préparation du vecteur de features
            df_input = pd.DataFrame([params])
            dummies  = pd.get_dummies(df_input,
                                      drop_first=False).astype(float)
            X_input  = dummies.reindex(
                columns=feature_cols, fill_value=0.0)

            probas     = model.predict_proba(X_input)[0]
            prediction = classes[int(np.argmax(probas))]
            color      = COULEURS[prediction]
            confiance  = max(probas) * 100

            # Boîte résultat
            st.markdown(
                f"""
                <div style="background:{color}15;
                            border:2px solid {color};
                            border-radius:12px;
                            padding:22px;
                            text-align:center;
                            margin-bottom:16px;">
                  <div style="font-size:52px;">
                    {EMOJIS[prediction]}
                  </div>
                  <div style="font-size:22px;font-weight:700;
                              color:{color};margin-top:8px;">
                    {prediction}
                  </div>
                  <div style="font-size:28px;font-weight:800;
                              color:{color};margin-top:6px;">
                    {confiance:.1f}%
                  </div>
                  <div style="font-size:12px;color:#666;
                              margin-top:4px;">
                    confiance du modèle
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Graphique probabilités
            st.markdown("**Probabilités par classe :**")
            fig, ax = plt.subplots(figsize=(6, 2.5))
            colors_b = [COULEURS[c] for c in classes]
            bars = ax.barh(classes, probas, color=colors_b,
                           edgecolor="white", height=0.5)
            for bar, prob, cl in zip(bars, probas, classes):
                ax.text(
                    min(prob + 0.01, 0.88),
                    bar.get_y() + bar.get_height()/2,
                    f"{prob*100:.1f}%",
                    va="center", fontsize=11,
                    fontweight="bold",
                    color=COULEURS[cl]
                )
            ax.set_xlim(0, 1)
            ax.axvline(1/3, color="gray", linestyle="--",
                       linewidth=0.8, alpha=0.5,
                       label="Seuil aléatoire (33%)")
            ax.set_facecolor("#F8F9FA")
            fig.patch.set_facecolor("#F8F9FA")
            ax.spines[["top","right"]].set_visible(False)
            ax.set_xlabel("Probabilité")
            ax.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Facteurs actifs du profil
            st.markdown("---")
            st.markdown("**Poids des facteurs dans ce profil :**")

            fi_all   = pd.Series(model.feature_importances_,
                                 index=feature_cols)
            actifs   = X_input.columns[X_input.iloc[0]==1].tolist()
            fi_actif = (fi_all[actifs]
                        .sort_values(ascending=False)
                        .head(6))

            labels_lisibles = {
                "age_grp_15-19":"Âge 15-19","age_grp_20-24":"Âge 20-24",
                "age_grp_25-29":"Âge 25-29","age_grp_30-34":"Âge 30-34",
                "age_grp_35-39":"Âge 35-39","age_grp_40-44":"Âge 40-44",
                "age_grp_45-49":"Âge 45-49",
                "statut_mat_En union":"En union",
                "statut_mat_Jamais en union":"Jamais en union",
                "statut_mat_Anciennement en union":"Anc. en union",
                "parite_Nullipare":"Nullipare",
                "parite_Non nullipare":"Non nullipare",
                "instruction_Aucun":"Sans instruction",
                "instruction_Primaire":"Instruction primaire",
                "instruction_Secondaire":"Instruction secondaire",
                "instruction_Superieur":"Instruction supérieure",
                "richesse_Tres pauvre":"Très pauvre",
                "richesse_Pauvre":"Pauvre",
                "richesse_Moyen":"Richesse moyenne",
                "richesse_Riche":"Riche",
                "richesse_Tres riche":"Très riche",
                "milieu_Rural":"Milieu rural",
                "milieu_Urbain":"Milieu urbain",
                "religion_Catholique":"Catholique",
                "religion_Protestante":"Protestante",
                "religion_Autres chretiens":"Autres chrétiens",
                "religion_Musulmane":"Musulmane",
                "religion_Animiste":"Animiste",
                "religion_Sans religion":"Sans religion",
                "expo_media_Non exposee":"Non exposée aux médias",
                "expo_media_Exposee":"Exposée aux médias",
            }

            max_imp = fi_actif.max() if not fi_actif.empty else 1
            for feat, imp in fi_actif.items():
                label   = labels_lisibles.get(feat, feat)
                largeur = int(imp / max_imp * 100)
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;
                                margin-bottom:8px;gap:12px;">
                      <span style="min-width:200px;font-size:13px;
                                   color:#333;">{label}</span>
                      <div style="flex:1;background:#E2E8F0;
                                  border-radius:6px;height:14px;">
                        <div style="width:{largeur}%;
                                    background:{color};
                                    height:14px;
                                    border-radius:6px;"></div>
                      </div>
                      <span style="font-size:12px;color:#888;
                                   min-width:45px;text-align:right;">
                        {imp*100:.2f}%
                      </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Interprétation
            st.markdown("---")
            messages = {
                "Aucune": (
                    "D'après ce profil, la femme est **peu susceptible** "
                    "d'utiliser une méthode contraceptive selon les "
                    "données EDS Cameroun 2018."
                ),
                "Traditionnelle": (
                    "D'après ce profil, la femme est susceptible "
                    "d'utiliser une **méthode traditionnelle** "
                    "(abstinence périodique, retrait…)"
                ),
                "Moderne": (
                    "D'après ce profil, la femme est susceptible "
                    "d'utiliser une **méthode moderne** "
                    "(pilule, DIU, injectable, implant…)"
                ),
            }
            st.markdown(messages[prediction])

        else:
            st.info(
                "👈 Renseignez le profil à gauche "
                "puis cliquez sur **Prédire**."
            )
            st.markdown("""
            **Les 3 classes prédites :**

            | Classe | Description |
            |--------|-------------|
            | ⛔ Aucune | Aucune méthode contraceptive |
            | 🌿 Traditionnelle | Abstinence périodique, retrait… |
            | 💊 Moderne | Pilule, DIU, injectable, implant… |

            **Modèle utilisé :** Gradient Boosting multinomial corrigé  
            **Entraîné sur :** 33 862 femmes — EDS Cameroun 2018  
            **Critère de sélection :** Balanced Accuracy maximale (CV-5)
            """)
