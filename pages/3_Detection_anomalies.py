
import streamlit as st
import pandas as pd
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Détection des anomalies",
    page_icon="🔍",
    layout="wide"
)

RACINE = Path(__file__).resolve().parents[1]

CHEMIN_ANOMALIES = (
    RACINE / "data" / "processed" / "anomalies_tunishop.csv"
)

# ==========================================
# 2. CHARGEMENT DES RESULTATS
# ==========================================

@st.cache_data
def charger_anomalies():
    return pd.read_csv(CHEMIN_ANOMALIES)

st.title("🔍 Détection des anomalies")
st.write(
    "Identification des transactions présentant "
    "des caractéristiques inhabituelles grâce à Isolation Forest."
)

st.divider()

if not CHEMIN_ANOMALIES.exists():
    st.error("Le fichier anomalies_tunishop.csv est introuvable.")
    st.stop()

df = charger_anomalies()

# Vérification de la colonne de statut
if "Statut_Anomalie" not in df.columns:
    st.error("La colonne Statut_Anomalie est absente du fichier.")
    st.stop()

# ==========================================
# 3. FILTRES
# ==========================================

st.sidebar.header("🔎 Filtres")

donnees = df.copy()

if "Region" in donnees.columns:
    regions = ["Toutes"] + sorted(
        donnees["Region"].dropna().unique().tolist()
    )

    region = st.sidebar.selectbox("Région", regions)

    if region != "Toutes":
        donnees = donnees[donnees["Region"] == region]

if "Categorie" in donnees.columns:
    categories = ["Toutes"] + sorted(
        donnees["Categorie"].dropna().unique().tolist()
    )

    categorie = st.sidebar.selectbox("Catégorie", categories)

    if categorie != "Toutes":
        donnees = donnees[
            donnees["Categorie"] == categorie
        ]

# ==========================================
# 4. IDENTIFICATION DES ANOMALIES
# ==========================================

anomalies = donnees[
    donnees["Statut_Anomalie"].astype(str).str.strip().str.lower()
    == "anomalie"
]

normales = donnees[
    donnees["Statut_Anomalie"].astype(str).str.strip().str.lower()
    == "normal"
]

total = len(donnees)
nombre_anomalies = len(anomalies)
nombre_normales = len(normales)

taux = (
    nombre_anomalies / total * 100
    if total > 0 else 0
)

# ==========================================
# 5. INDICATEURS
# ==========================================

st.subheader("📊 Résumé de la détection")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Transactions analysées", f"{total:,}")
col2.metric("Anomalies détectées", f"{nombre_anomalies:,}")
col3.metric("Transactions normales", f"{nombre_normales:,}")
col4.metric("Taux d'anomalies", f"{taux:.2f}%")

st.divider()

# ==========================================
# 6. VISUALISATION
# ==========================================

st.subheader("📈 Répartition des transactions")

repartition = pd.DataFrame({
    "Type": ["Normales", "Anomalies"],
    "Nombre": [nombre_normales, nombre_anomalies]
})

st.bar_chart(
    repartition,
    x="Type",
    y="Nombre"
)

st.divider()

# ==========================================
# 7. ANALYSE DES TRANSACTIONS ATYPIQUES
# ==========================================

st.subheader("🚨 Transactions inhabituelles")

if nombre_anomalies > 0:

    st.warning(
        f"{nombre_anomalies} transaction(s) inhabituelle(s) "
        "ont été identifiées."
    )

    st.dataframe(
        anomalies,
        use_container_width=True
    )

    st.download_button(
        "📥 Télécharger les anomalies",
        data=anomalies.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="anomalies_tunishop.csv",
        mime="text/csv"
    )

else:
    st.success("Aucune anomalie détectée pour cette sélection.")

# ==========================================
# 8. INTERPRETATION
# ==========================================

st.divider()

st.subheader("💡 Interprétation")

st.write(
    "Isolation Forest identifie les observations "
    "qui présentent des caractéristiques différentes "
    "du comportement général des transactions."
)

st.info(
    "Une transaction atypique n'est pas nécessairement "
    "frauduleuse. Elle doit être examinée avant toute "
    "décision commerciale."
)

st.caption(
    "TuniShop Intelligence | Module de détection des anomalies"
)
