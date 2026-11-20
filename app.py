
import streamlit as st
import pandas as pd
import joblib
import json

from pathlib import Path

# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="TuniShop Intelligence",
    page_icon="🛍️",
    layout="wide"
)

RACINE = Path(__file__).parent

CHEMIN_DONNEES = RACINE / "data" / "processed" / "ventes_tunishop.csv"
CHEMIN_ANOMALIES = RACINE / "data" / "processed" / "anomalies_tunishop.csv"
CHEMIN_MODELE = RACINE / "models" / "random_forest_ca.pkl"
CHEMIN_METRIQUES = RACINE / "outputs" / "performances_modele.json"


# =====================================================
# 2. DESIGN PERSONNALISE
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #F5F7FB;
}

h1 {
    color: #17365D;
    font-weight: 800;
}

h2, h3 {
    color: #24476B;
}

[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #E1E7EF;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

[data-testid="stMetricValue"] {
    color: #167D8D;
    font-weight: bold;
}

[data-testid="stSidebar"] {
    background-color: #EAF0F8;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. CHARGEMENT DES DONNEES
# =====================================================

@st.cache_data
def charger_donnees():
    donnees = pd.read_csv(CHEMIN_DONNEES)
    donnees["Date"] = pd.to_datetime(
        donnees["Date"], errors="coerce"
    )
    return donnees


try:
    df = charger_donnees()

except FileNotFoundError:
    st.error("Fichier ventes_tunishop.csv introuvable.")
    st.stop()


# =====================================================
# 4. EN-TETE
# =====================================================

st.markdown("""
<div style="
    background-color:#17365D;
    padding:30px;
    border-radius:15px;
    margin-bottom:20px;
">
    <h1 style="color:white !important; margin:0;">
        🛍️ TuniShop Intelligence
    </h1>
    <p style="color:#D9E7F5; margin-top:10px;">
        Business Intelligence | Analyse des ventes |
        Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

st.subheader("Tableau de bord décisionnel des ventes")
st.caption("Analyse des performances commerciales et de la rentabilité")


# =====================================================
# 5. FILTRES
# =====================================================

st.sidebar.title("🔎 Filtres")

regions = ["Toutes"] + sorted(
    df["Region"].dropna().unique().tolist()
)

categories = ["Toutes"] + sorted(
    df["Categorie"].dropna().unique().tolist()
)

region_selectionnee = st.sidebar.selectbox(
    "Sélectionner une région",
    regions
)

categorie_selectionnee = st.sidebar.selectbox(
    "Sélectionner une catégorie",
    categories
)

donnees = df.copy()

if region_selectionnee != "Toutes":
    donnees = donnees[
        donnees["Region"] == region_selectionnee
    ]

if categorie_selectionnee != "Toutes":
    donnees = donnees[
        donnees["Categorie"] == categorie_selectionnee
    ]


# =====================================================
# 6. DASHBOARD KPI
# =====================================================

st.divider()
st.subheader("📊 Indicateurs clés de performance")

if donnees.empty:

    st.warning("Aucune donnée disponible pour ces filtres.")

else:

    ca = donnees["Chiffre_Affaires"].sum()
    marge = donnees["Marge_Brute"].sum()

    taux_marge = (
        marge / ca * 100 if ca != 0 else 0
    )

    commandes = donnees["ID_Commande"].nunique()

    panier_moyen = (
        ca / commandes if commandes != 0 else 0
    )

    quantite = donnees["Quantite"].sum()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Chiffre d'affaires",
        f"{ca:,.2f} DT"
    )

    col2.metric(
        "Marge brute",
        f"{marge:,.2f} DT"
    )

    col3.metric(
        "Taux de marge",
        f"{taux_marge:.2f}%"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric("Nombre de commandes", f"{commandes:,}")
    col5.metric("Panier moyen", f"{panier_moyen:,.2f} DT")
    col6.metric("Quantité vendue", f"{quantite:,}")


    # =================================================
    # 7. EVOLUTION MENSUELLE
    # =================================================

    st.divider()
    st.subheader("📈 Évolution mensuelle du chiffre d'affaires")

    mensuel = donnees.dropna(subset=["Date"]).copy()

    mensuel["Mois"] = mensuel["Date"].dt.to_period("M").astype(str)

    ca_mensuel = (
        mensuel.groupby("Mois")["Chiffre_Affaires"]
        .sum()
        .sort_index()
    )

    st.line_chart(
        ca_mensuel,
        x_label="Mois",
        y_label="Chiffre d'affaires (DT)"
    )


    # =================================================
    # 8. PERFORMANCE REGIONALE
    # =================================================

    st.divider()
    st.subheader("🌍 Performance par région")

    performance_region = (
        donnees.groupby("Region")["Chiffre_Affaires"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(
        performance_region,
        x_label="Région",
        y_label="Chiffre d'affaires (DT)"
    )


    # =================================================
    # 9. PERFORMANCE PAR CATEGORIE
    # =================================================

    st.divider()
    st.subheader("🏷️ Rentabilité par catégorie")

    performance_categorie = (
        donnees.groupby("Categorie")
        .agg(
            Chiffre_Affaires=("Chiffre_Affaires", "sum"),
            Marge_Brute=("Marge_Brute", "sum"),
            Quantite=("Quantite", "sum")
        )
    )

    performance_categorie["Taux_Marge"] = (
        performance_categorie["Marge_Brute"]
        / performance_categorie["Chiffre_Affaires"] * 100
    ).fillna(0)

    st.bar_chart(
        performance_categorie["Taux_Marge"].sort_values(
            ascending=False
        ),
        x_label="Catégorie",
        y_label="Taux de marge (%)"
    )

    st.dataframe(
        performance_categorie.round(2),
        use_container_width=True
    )


    # =================================================
    # 10. RECOMMANDATIONS COMMERCIALES
    # =================================================

    st.divider()
    st.subheader("💡 Recommandations commerciales")

    seuil_ca = performance_categorie["Chiffre_Affaires"].median()
    seuil_marge = performance_categorie["Taux_Marge"].median()

    def generer_recommandation(ligne):

        ca_categorie = ligne["Chiffre_Affaires"]
        taux = ligne["Taux_Marge"]

        if ca_categorie >= seuil_ca and taux >= seuil_marge:
            return pd.Series([
                "Performante",
                "Maintenir la stratégie et sécuriser le stock"
            ])

        elif ca_categorie >= seuil_ca:
            return pd.Series([
                "Volume élevé",
                "Optimiser les coûts et améliorer les marges"
            ])

        elif taux >= seuil_marge:
            return pd.Series([
                "Niche rentable",
                "Renforcer la promotion"
            ])

        else:
            return pd.Series([
                "À améliorer",
                "Réexaminer les prix et la stratégie commerciale"
            ])

    performance_categorie[
        ["Segment", "Recommandation"]
    ] = performance_categorie.apply(
        generer_recommandation,
        axis=1
    )

    st.dataframe(
        performance_categorie.sort_values(
            "Chiffre_Affaires",
            ascending=False
        ),
        use_container_width=True
    )


    # =================================================
    # 11. DETECTION DES ANOMALIES
    # =================================================

    st.divider()
    st.subheader("🔍 Détection des transactions inhabituelles")

    if CHEMIN_ANOMALIES.exists():

        anomalies = pd.read_csv(CHEMIN_ANOMALIES)

        if region_selectionnee != "Toutes":
            anomalies = anomalies[
                anomalies["Region"] == region_selectionnee
            ]

        if categorie_selectionnee != "Toutes":
            anomalies = anomalies[
                anomalies["Categorie"] == categorie_selectionnee
            ]

        transactions_atypiques = anomalies[
            anomalies["Statut_Anomalie"] == "Anomalie"
        ]

        nombre_analyse = len(anomalies)
        nombre_anomalies = len(transactions_atypiques)

        taux_anomalies = (
            nombre_anomalies / nombre_analyse * 100
            if nombre_analyse > 0 else 0
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Transactions analysées", nombre_analyse)
        col2.metric("Anomalies détectées", nombre_anomalies)
        col3.metric("Taux d'anomalies", f"{taux_anomalies:.2f}%")

        if nombre_anomalies > 0:

            st.warning(
                "Certaines transactions présentent "
                "des caractéristiques inhabituelles."
            )

            st.dataframe(
                transactions_atypiques,
                use_container_width=True
            )

            st.download_button(
                "📥 Télécharger les anomalies",
                transactions_atypiques.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="anomalies_tunishop.csv",
                mime="text/csv"
            )

        else:
            st.success("Aucune anomalie détectée pour cette sélection.")

        st.caption(
            "Une anomalie n'est pas nécessairement une fraude. "
            "Une vérification humaine reste nécessaire."
        )

    else:
        st.warning("Fichier anomalies_tunishop.csv introuvable.")


    # =================================================
    # 12. PREDICTION DU CHIFFRE D'AFFAIRES
    # =================================================

    st.divider()
    st.subheader("🤖 Estimation du chiffre d'affaires")

    if CHEMIN_MODELE.exists():

        @st.cache_resource
        def charger_modele():
            return joblib.load(CHEMIN_MODELE)

        try:
            modele = charger_modele()

            with st.form("formulaire_prediction"):

                col1, col2 = st.columns(2)

                with col1:

                    quantite_saisie = st.number_input(
                        "Quantité commandée",
                        min_value=1,
                        max_value=1000,
                        value=4
                    )

                    region_saisie = st.selectbox(
                        "Région de la commande",
                        sorted(df["Region"].dropna().unique())
                    )

                with col2:

                    categorie_saisie = st.selectbox(
                        "Catégorie",
                        sorted(df["Categorie"].dropna().unique())
                    )

                    mois_saisie = st.selectbox(
                        "Mois",
                        list(range(1, 13)),
                        index=10
                    )

                lancer_prediction = st.form_submit_button(
                    "Calculer l'estimation"
                )

            if lancer_prediction:

                nouvelle_commande = pd.DataFrame({
                    "Quantite": [quantite_saisie],
                    "Region": [region_saisie],
                    "Categorie": [categorie_saisie],
                    "Mois": [mois_saisie]
                })

                prediction = modele.predict(
                    nouvelle_commande
                )[0]

                st.success(
                    f"Chiffre d'affaires estimé : "
                    f"{prediction:,.2f} DT"
                )

                st.caption(
                    "Estimation issue du modèle Random Forest. "
                    "Il s'agit d'une estimation de commande, "
                    "pas d'une prévision mensuelle."
                )

        except Exception as erreur:
            st.error(f"Erreur lors du chargement ou de la prédiction : {erreur}")

    else:
        st.warning("Modèle Random Forest introuvable.")


    # =================================================
    # 13. EVALUATION DU MODELE
    # =================================================

    st.divider()
    st.subheader("📊 Évaluation du modèle prédictif")

    if CHEMIN_METRIQUES.exists():

        with open(CHEMIN_METRIQUES, "r", encoding="utf-8") as fichier:
            metriques = json.load(fichier)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "MAE",
            f"{metriques['MAE']:,.2f} DT"
        )

        col2.metric(
            "RMSE",
            f"{metriques['RMSE']:,.2f} DT"
        )

        col3.metric(
            "R²",
            f"{metriques['R2']:.4f}"
        )

        st.markdown("**Interprétation :**")

        st.write(
            f"Le MAE indique une erreur absolue moyenne de "
            f"{metriques['MAE']:,.2f} DT."
        )

        st.write(
            f"Le RMSE est de {metriques['RMSE']:,.2f} DT."
        )

        r2 = metriques["R2"]

        if r2 >= 0.75:
            st.success("Bon pouvoir explicatif sur les données de test.")

        elif r2 >= 0.50:
            st.info("Pouvoir explicatif modéré.")

        else:
            st.warning("Pouvoir explicatif faible : amélioration nécessaire.")

    else:
        st.warning("Fichier performances_modele.json introuvable.")


# =====================================================
# 14. PIED DE PAGE
# =====================================================

st.divider()

st.markdown("""
<div style="text-align:center; color:#718096; padding:15px;">
    <b>TuniShop Intelligence</b><br>
    Projet de Business Intelligence et Machine Learning
</div>
""", unsafe_allow_html=True)
