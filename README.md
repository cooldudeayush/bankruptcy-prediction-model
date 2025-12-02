

# 📘 Bankruptcy Prediction System

### **LightGBM Models • 15-Feature Horizons • Optimized Thresholds • SHAP Explainability • Streamlit App**

This repository contains a complete end-to-end bankruptcy prediction system using:

* **LightGBM Gradient Boosting Models**
* **Top 15 Financial Attributes Per Prediction Horizon**
* **Optimal Threshold Search to Maximize F1**
* **Clean, interpretable SHAP visualizations**
* **Interactive Streamlit Web App for real-time predictions**
* **Saved `.pkl` models for deployment**

The project is designed for **high F1-score**, **robust handling of imbalanced datasets**, and **clear interpretability** for practical financial use.

---

# 🚀 Features

### 🔹 **1. Top 15 Features Per Horizon**

For each prediction horizon (1–5 years ahead), only the 15 most important financial attributes are used.
This improves performance, reduces noise, and increases interpretability.

### 🔹 **2. LightGBM With F1–Optimized Thresholding**

Each horizon searches thresholds from **0.05 → 0.95** and selects the one maximizing **validation F1**.

### 🔹 **3. SHAP Explainability**

Global and local interpretability through:

* SHAP Beeswarm Summary Plot
* SHAP Bar Feature Importance Plot

### 🔹 **4. Interactive Streamlit App**

A user-friendly frontend where anyone can input financial attributes and get live bankruptcy predictions.

### 🔹 **5. Production-ready `.pkl` Models**

Easily loadable for APIs, dashboards, or batch scoring.

---

# 📁 Repository Structure

```
├── Final_Master_Code.ipynb        # Full training + thresholding + SHAP notebook
├── streamlit_app.py               # Streamlit interface (see below)
├── models/
│   ├── model_h1.pkl
│   ├── model_h2.pkl
│   ├── model_h3.pkl
│   ├── model_h4.pkl
│   └── model_h5.pkl
├── data/                          # Original datasets 
├── README.md                      # This file
└── requirements.txt               # Libraries needed
```

---

# 🧠 Methodology

## **1️⃣ Data Preparation**

For each prediction horizon **h = 1 to 5**:

* Load dataset
* Select top 15 features (`top_features[h]`)
* Split into train / validation / test sets:

  * **70% Train**
  * **15% Validation**
  * **15% Test**
* Apply **median imputation** (no scaling to preserve tree performance)

---

## **2️⃣ LightGBM Modeling**

```python
LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.02,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,   # imbalance correction
    objective="binary",
    random_state=42,
    verbose=-1
)
```

Why this works:

* Trees do not need scaling
* LightGBM handles missing values
* `scale_pos_weight` handles class imbalance
* Large number of boosting rounds + small learning rate improves stability

---

## **3️⃣ Optimal Threshold Search (F1 Maximization)**

Instead of default threshold = 0.50, each horizon tests:

```
threshold ∈ {0.05, 0.06, ..., 0.95}
```

The threshold that gives the **highest validation F1** is selected and applied to the test set.

Stored in:

```
best_thresholds[horizon]
```

---

## **4️⃣ Model Saving**

Each fitted model pipeline is saved as:

```
model_h{horizon}.pkl
```

Load using:

```python
import joblib
model = joblib.load("model_h3.pkl")
```

---

## **5️⃣ SHAP Explainability**

Two plots are generated per horizon:

### ✔ SHAP Summary (Beeswarm)

Shows how each feature pushes predictions positive or negative.

### ✔ SHAP Bar Plot

Shows the mean absolute importance of each feature.

Example:

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed)

shap.summary_plot(shap_values, X_transformed, feature_names=feats)
```

---

# 🖥 Streamlit Application

An interactive app is included for real-time prediction:

```
streamlit run streamlit_app.py
```

### Features:

* Select prediction horizon (1–5 years)
* Input values for top 15 financial attributes
* View bankruptcy probability & classification
* Clean, responsive UI

### Example Code (Included in repo)

```python
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bankruptcy Prediction App", layout="centered")

# Load models
models = {
    1: joblib.load("model_h1.pkl"),
    2: joblib.load("model_h2.pkl"),
    3: joblib.load("model_h3.pkl"),
    4: joblib.load("model_h4.pkl"),
    5: joblib.load("model_h5.pkl"),
}

# Top 15 features per horizon
top_features = {
    1: [...],
    2: [...],
    3: [...],
    4: [...],
    5: [...]
}

st.title("📊 Bankruptcy Prediction App")
st.write("Enter company financial attributes to predict bankruptcy probability.")

# Horizon selection
horizon = st.selectbox("Select Prediction Horizon:", [1, 2, 3, 4, 5], index=0)
model = models[horizon]
features = top_features[horizon]

input_data = {}
for feat in features:
    input_data[feat] = st.number_input(feat, value=0.0, format="%.6f")
df_input = pd.DataFrame([input_data])

if st.button("Predict Bankruptcy Risk"):
    prob = model.predict_proba(df_input)[0][1]
    pred = model.predict(df_input)[0]

    st.subheader("🔍 Prediction Result")
    st.write(f"**Bankruptcy Probability:** `{prob:.3f}`")

    if pred == 1:
        st.error("⚠ The model predicts: BANKRUPT")
    else:
        st.success("✅ SAFE / NOT BANKRUPT")
```

---

# 📊 Model Performance Summary

Each horizon produces:

| Metric       | Description                                      |
| ------------ | ------------------------------------------------ |
| **val_acc**  | Validation accuracy                              |
| **val_f1**   | Validation F1-score (used for threshold picking) |
| **val_auc**  | Validation ROC AUC                               |
| **test_acc** | Test accuracy                                    |
| **test_f1**  | Test F1-score                                    |
| **test_auc** | Test ROC AUC                                     |
| **best_thr** | Optimal decision threshold                       |

SHAP plots help interpret which features drive bankruptcy risk across horizons.

---

# 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo>.git
cd <repo>
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App

```bash
streamlit run streamlit_app.py
```

---

=
# 📬 Contact

Feel free to reach out to Subhojit Sapui, Ayush Ranjan , Aditi Pandey regarding this project.


