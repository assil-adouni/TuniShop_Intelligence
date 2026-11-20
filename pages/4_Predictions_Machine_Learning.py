
import streamlit as st
import pandas as pd
import joblib
import json

from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Prédictions et Machine Learning",
    page_icon="🤖",
    layout="wide"
)

RACINE = Path(__file__).resolve().parents[1]

CHEMIN_DONNEES = (
    RACINE / "data" / "processed" / "ventes_tunishop.csv"
)

CHEMIN_MODELE = (
    RACINE / "models" / "random_forest_ca.pkl"
)

CHEMIN_METRIQUES = (
    RACINE / "outputs" / "performances_modele.json"
)

# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

@st.cache_data
def charger_donnees():
    return pd.read_csv(CHEMIN_DONNEES)

if not CHEMIN_DONNEES.exists():
    st.error("Le fichier des ventes est introuvable.")
    st.stop()

df = charger_donnees()

# ==========================================
# 3. TITRE
# ==========================================

st.title("🤖 Prédictions et Machine Learning")

st.write(
    "Estimation du chiffre d'affaires d'une commande "
    "à l'aide du modèle Random Forest Regressor."
)

st.divider()

# ==========================================
# 4. CHARGEMENT DU MODELE
# ==========================================

if not CHEMIN_MODELE.exists():
    st.error(
        "Modèle introuvable : random_forest_ca.pkl. "
        "Vérifiez le dossier models."
    )
    st.stop()

@st.cache_resource
def charger_modele():
    return joblib.load(CHEMIN_MODELE)

try:
    modele = charger_modele()

except Exception as erreur:
    st.error(f"Impossible de charger le modèle : {erreur}")
    st.stop()

# ==========================================
# 5. FORMULAIRE DE PREDICTION
# ==========================================

st.subheader("🧮 Simuler une nouvelle commande")

st.write(
    "Renseignez les caractéristiques de la commande "
    "pour obtenir une estimation."
)

with st.form("formulaire_prediction"):

    col1, col2 = st.columns(2)

    with col1:

        quantite = st.number_input(
            "Quantité commandée",
            min_value=1,
            max_value=1000,
            value=4
        )

        region = st.selectbox(
            "Région",
            sorted(df["Region"].dropna().unique().tolist())
        )

    with col2:

        categorie = st.selectbox(
            "Catégorie",
            sorted(df["Categorie"].dropna().unique().tolist())
        )

        mois = st.selectbox(
            "Mois",
            options=list(range(1, 13)),
            index=10
        )

    lancer = st.form_submit_button(
        "🚀 Estimer le chiffre d'affaires"
    )

# ==========================================
# 6. RESULTAT DE LA PREDICTION
# ==========================================

if lancer:

    nouvelle_commande = pd.DataFrame({
        "Quantite": [quantite],
        "Region": [region],
        "Categorie": [categorie],
        "Mois": [mois]
    })

    try:

        prediction = modele.predict(nouvelle_commande)[0]

        st.success("Estimation réalisée avec succès !")

        st.metric(
            "Chiffre d'affaires estimé",
            f"{prediction:,.2f} DT"
        )

        st.write("Caractéristiques de la commande :")

        st.dataframe(
            nouvelle_commande,
            use_container_width=True
        )

    except Exception as erreur:

        st.error(
            f"Erreur pendant la prédiction : {erreur}"
        )

st.divider()

# ==========================================
# 7. EVALUATION DU MODELE
# ==========================================

st.subheader("📊 Évaluation des performances")

if CHEMIN_METRIQUES.exists():

    with open(
        CHEMIN_METRIQUES,
        "r",
        encoding="utf-8"
    ) as fichier:

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

    st.markdown("### Interprétation")

    st.write(
        f"Le MAE représente une erreur absolue moyenne "
        f"de {metriques['MAE']:,.2f} DT."
    )

    st.write(
        f"Le RMSE est de {metriques['RMSE']:,.2f} DT."
    )

    r2 = metriques["R2"]

    if r2 >= 0.75:
        st.success(
            "Le modèle présente un bon pouvoir explicatif "
            "sur les données de test."
        )

    elif r2 >= 0.50:
        st.info(
            "Le modèle présente un pouvoir explicatif modéré."
        )

    else:
        st.warning(
            "Le pouvoir explicatif du modèle est faible. "
            "Une amélioration est nécessaire."
        )

else:

    st.warning(
        "Le fichier performances_modele.json est introuvable."
    )

# ==========================================
# 8. EXPLICATION DU MODELE
# ==========================================

st.divider()

st.subheader("🧠 À propos du modèle")

st.write("""
**Random Forest Regressor** est un algorithme de Machine Learning
qui combine plusieurs arbres de décision afin d'estimer
une valeur numérique.

Dans ce projet, il utilise quatre variables :

- Quantité commandée
- Région
- Catégorie
- Mois

La variable cible est le chiffre d'affaires de la commande.
""")

st.info(
    "Cette fonctionnalité estime le chiffre d'affaires "
    "d'une commande individuelle. Elle ne constitue pas "
    "une prévision des ventes futures par mois."
)

st.caption(
    "TuniShop Intelligence | Module Machine Learning"
)
