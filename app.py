import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bankruptcy Prediction App", layout="centered")

# ===========================================================
# 1. FULL MAPPING: AttrXX → Actual Feature Name (X1–X64)
# ===========================================================

feature_names = {
    "Attr1": "Net profit / total assets",
    "Attr2": "Total liabilities / total assets",
    "Attr3": "Working capital / total assets",
    "Attr4": "Current assets / short-term liabilities",
    "Attr5": "Cash + receivables + ST securities - ST liabilities / (OpEx - Depreciation) × 365",
    "Attr6": "Retained earnings / total assets",
    "Attr7": "EBIT / total assets",
    "Attr8": "Book value of equity / total liabilities",
    "Attr9": "Sales / total assets",
    "Attr10": "Equity / total assets",
    "Attr11": "(Gross profit + extraordinary items + financial expenses) / total assets",
    "Attr12": "Gross profit / short-term liabilities",
    "Attr13": "(Gross profit + depreciation) / sales",
    "Attr14": "(Gross profit + interest) / total assets",
    "Attr15": "(Total liabilities × 365) / (Gross profit + depreciation)",
    "Attr16": "(Gross profit + depreciation) / total liabilities",
    "Attr17": "Total assets / total liabilities",
    "Attr18": "Gross profit / total assets",
    "Attr19": "Gross profit / sales",
    "Attr20": "(Inventory × 365) / sales",
    "Attr21": "Sales (n) / sales (n−1)",
    "Attr22": "Profit on operating activities / total assets",
    "Attr23": "Net profit / sales",
    "Attr24": "(Gross profit in 3 years) / total assets",
    "Attr25": "(Equity − share capital) / total assets",
    "Attr26": "(Net profit + depreciation) / total liabilities",
    "Attr27": "Profit on operating activities / financial expenses",
    "Attr28": "Working capital / fixed assets",
    "Attr29": "Logarithm of total assets",
    "Attr30": "(Total liabilities − cash) / sales",
    "Attr31": "(Gross profit + interest) / sales",
    "Attr32": "(Current liabilities × 365) / cost of products sold",
    "Attr33": "Operating expenses / short-term liabilities",
    "Attr34": "Operating expenses / total liabilities",
    "Attr35": "Profit on sales / total assets",
    "Attr36": "Total sales / total assets",
    "Attr37": "(Current assets − inventories) / long-term liabilities",
    "Attr38": "Constant capital / total assets",
    "Attr39": "Profit on sales / sales",
    "Attr40": "(Current assets − inventory − receivables) / short-term liabilities",
    "Attr41": "Total liabilities / ((Profit on operating activities + depreciation) × (12/365))",
    "Attr42": "Profit on operating activities / sales",
    "Attr43": "Receivables + inventory turnover in days",
    "Attr44": "(Receivables × 365) / sales",
    "Attr45": "Net profit / inventory",
    "Attr46": "(Current assets − inventory) / short-term liabilities",
    "Attr47": "(Inventory × 365) / cost of products sold",
    "Attr48": "EBITDA / total assets",
    "Attr49": "EBITDA / sales",
    "Attr50": "Current assets / total liabilities",
    "Attr51": "Short-term liabilities / total assets",
    "Attr52": "(Short-term liabilities × 365) / cost of products sold",
    "Attr53": "Equity / fixed assets",
    "Attr54": "Constant capital / fixed assets",
    "Attr55": "Working capital",
    "Attr56": "(Sales − cost of products sold) / sales",
    "Attr57": "(CA − inventory − ST liabilities) / (Sales − gross profit − depreciation)",
    "Attr58": "Total costs / total sales",
    "Attr59": "Long-term liabilities / equity",
    "Attr60": "Sales / inventory",
    "Attr61": "Sales / receivables",
    "Attr62": "(Short-term liabilities × 365) / sales",
    "Attr63": "Sales / short-term liabilities",
    "Attr64": "Sales / fixed assets"
}

# ===========================================================
# 2. Load Models
# ===========================================================

models = {
    1: joblib.load("model_h1_top10_improved.pkl"),
    2: joblib.load("model_h2_top10_improved.pkl"),
    3: joblib.load("model_h3_top10_improved.pkl"),
    4: joblib.load("model_h4_top10_improved.pkl"),
    5: joblib.load("model_h5_top10_improved.pkl"),
}

# ===========================================================
# 3. Top features per horizon
# ===========================================================

top_features = {
    1: ['Attr58', 'Attr27', 'Attr9', 'Attr39', 'Attr5', 'Attr13', 'Attr30', 'Attr41', 'Attr56', 'Attr55', 'Attr45', 'Attr19', 'Attr47', 'Attr52', 'Attr49'],
    2: ['Attr27', 'Attr58', 'Attr34', 'Attr46', 'Attr5', 'Attr24', 'Attr29', 'Attr36', 'Attr9', 'Attr21', 'Attr42', 'Attr39', 'Attr6', 'Attr55', 'Attr44'],
    3: ['Attr5', 'Attr34', 'Attr27', 'Attr46', 'Attr58', 'Attr21', 'Attr39', 'Attr29', 'Attr41', 'Attr40', 'Attr37', 'Attr25', 'Attr56', 'Attr35', 'Attr24'],
    4: ['Attr46', 'Attr27', 'Attr34', 'Attr5', 'Attr21', 'Attr58', 'Attr36', 'Attr24', 'Attr9', 'Attr13', 'Attr29', 'Attr56', 'Attr25', 'Attr40', 'Attr39'],
    5: ['Attr27', 'Attr34', 'Attr21', 'Attr46', 'Attr35', 'Attr58', 'Attr36', 'Attr25', 'Attr24', 'Attr9', 'Attr39', 'Attr56', 'Attr29', 'Attr6', 'Attr41']
}

# ===========================================================
# 4. UI
# ===========================================================

st.title("📊 Bankruptcy Prediction App")
st.write("Enter company financial attributes to predict bankruptcy probability.")

# User selects horizon
horizon = st.selectbox(
    "Select Prediction Horizon (Years Ahead):",
    [1, 2, 3, 4, 5],
    index=0
)

model = models[horizon]
features = top_features[horizon]

st.subheader(f"Enter values for the top {len(features)} features (Horizon {horizon})")

input_data = {}

# Input fields with real names
for feat in features:
    readable = feature_names.get(feat, feat)
    input_data[feat] = st.number_input(
        f"{readable} ({feat})",
        value=0.0,
        format="%.6f"
    )

df_input = pd.DataFrame([input_data])

# ===========================================================
# 5. Prediction
# ===========================================================

if st.button("Predict Bankruptcy Risk"):
    prob = model.predict_proba(df_input)[0][1]
    pred = model.predict(df_input)[0]

    st.subheader("🔍 Prediction Result")
    st.write(f"**Bankruptcy Probability:** `{prob:.3f}`")

    if pred == 1:
        st.error("⚠ The model predicts: **BANKRUPT**")
    else:
        st.success("✅ The model predicts: **SAFE / NOT BANKRUPT**")
