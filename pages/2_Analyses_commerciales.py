
import streamlit as st
import pandas as pd
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Analyses commerciales",
    page_icon="📊",
    layout="wide"
)

RACINE = Path(__file__).resolve().parents[1]

CHEMIN_DONNEES = (
    RACINE / "data" / "processed" / "ventes_tunishop.csv"
)

# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

@st.cache_data
def charger_donnees():
    donnees = pd.read_csv(CHEMIN_DONNEES)
    donnees["Date"] = pd.to_datetime(
        donnees["Date"], errors="coerce"
    )
    return donnees

df = charger_donnees()

# ==========================================
# 3. TITRE
# ==========================================

st.title("📊 Analyses commerciales")
st.write(
    "Exploration détaillée des performances "
    "commerciales de TuniShop."
)

st.divider()

# ==========================================
# 4. FILTRES
# ==========================================

st.sidebar.header("Filtres d'analyse")

regions = ["Toutes"] + sorted(
    df["Region"].dropna().unique().tolist()
)

categories = ["Toutes"] + sorted(
    df["Categorie"].dropna().unique().tolist()
)

region = st.sidebar.selectbox(
    "Région",
    regions
)

categorie = st.sidebar.selectbox(
    "Catégorie",
    categories
)

donnees = df.copy()

if region != "Toutes":
    donnees = donnees[donnees["Region"] == region]

if categorie != "Toutes":
    donnees = donnees[donnees["Categorie"] == categorie]

if donnees.empty:
    st.warning("Aucune donnée pour cette sélection.")
    st.stop()

# ==========================================
# 5. INDICATEURS
# ==========================================

ca = donnees["Chiffre_Affaires"].sum()
marge = donnees["Marge_Brute"].sum()
quantite = donnees["Quantite"].sum()

taux_marge = marge / ca * 100 if ca else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Chiffre d'affaires", f"{ca:,.2f} DT")
col2.metric("Marge brute", f"{marge:,.2f} DT")
col3.metric("Taux de marge", f"{taux_marge:.2f}%")
col4.metric("Quantité vendue", f"{quantite:,}")

st.divider()

# ==========================================
# 6. ANALYSE TEMPORELLE
# ==========================================

st.subheader("📈 Évolution mensuelle des ventes")

temporel = donnees.dropna(subset=["Date"]).copy()

temporel["Mois"] = (
    temporel["Date"].dt.to_period("M").astype(str)
)

ca_mensuel = (
    temporel.groupby("Mois")["Chiffre_Affaires"]
    .sum()
    .sort_index()
)

st.line_chart(
    ca_mensuel,
    x_label="Mois",
    y_label="Chiffre d'affaires (DT)"
)

st.divider()

# ==========================================
# 7. ANALYSE REGIONALE
# ==========================================

st.subheader("🌍 Chiffre d'affaires par région")

ca_region = (
    donnees.groupby("Region")["Chiffre_Affaires"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(
    ca_region,
    x_label="Région",
    y_label="Chiffre d'affaires (DT)"
)

st.divider()

# ==========================================
# 8. ANALYSE PAR CATEGORIE
# ==========================================

st.subheader("🏷️ Performance par catégorie")

performance = (
    donnees.groupby("Categorie")
    .agg(
        Chiffre_Affaires=("Chiffre_Affaires", "sum"),
        Marge_Brute=("Marge_Brute", "sum"),
        Quantite=("Quantite", "sum")
    )
)

performance["Taux_Marge"] = (
    performance["Marge_Brute"]
    / performance["Chiffre_Affaires"] * 100
).fillna(0)

st.dataframe(
    performance.sort_values(
        "Chiffre_Affaires",
        ascending=False
    ).round(2),
    use_container_width=True
)

st.bar_chart(
    performance["Taux_Marge"].sort_values(
        ascending=False
    ),
    x_label="Catégorie",
    y_label="Taux de marge (%)"
)

# ==========================================
# 9. EXPORTATION
# ==========================================

st.divider()

st.download_button(
    "📥 Télécharger les données filtrées",
    data=donnees.to_csv(index=False).encode("utf-8-sig"),
    file_name="analyse_commerciale_tunishop.csv",
    mime="text/csv"
)

st.caption("TuniShop Intelligence | Analyses commerciales")
