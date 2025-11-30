
# 📘 Bankruptcy Prediction Using LightGBM

 Top-15 Feature Models • Optimal Threshold Search • SHAP Explainability

This repository contains a complete pipeline for predicting corporate bankruptcy using LightGBM, custom Top-15 feature subsets, optimal F1 thresholds, and SHAP feature explainability.

The project is designed for high interpretability, efficient modeling, and high F1 performance on imbalanced datasets.

---

 🚀 Project Highlights

 ✅ Top-15 Feature Models per Horizon

Each prediction horizon uses its own best-performing feature subset (e.g., Attr5, Attr27, Attr46…).
These features were pre-selected using statistical + model-driven importance ranking.

✅ LightGBM with F1-Optimized Thresholding

Instead of using the default 0.50 probability cutoff, each model searches thresholds from `0.05 → 0.95` and picks the best one for maximum F1-score.

 ✅ No Over-Processing

The pipeline follows best practices for tree models:

* No scaling
* No iterative imputation
* Only median imputation
  This avoids distortions and improves model stability.

 ✅ Explainability With SHAP

SHAP summary & bar plots are generated for all Top-15 feature models:

* Feature contribution analysis
* Global importance
* Per-horizon interpretability

---



# 🧠 Methodology

 1️⃣ Data Preparation

For each horizon:

* Load dataset
* Use Top-15 preselected features
* Perform a 70-15-15 stratified split (train / validation / test)
* Apply median imputation for missing values

---

 2️⃣ Model Training (LightGBM)

Key parameters used:

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


 3️⃣ Optimal Threshold Search (F1-Maximization)

For each horizon:

* Compute validation probabilities
* Test thresholds from 0.05 to 0.95
* Pick the threshold that achieves maximum validation F1
* Apply it to the test set

Outputs include:

| Metric              | Value           |
| ------------------- | --------------- |
| Validation Accuracy | ✔               |
| Validation F1       | ✔               |
| Validation AUC      | ✔               |
| Test Accuracy       | ✔               |
| Test F1             | ✔               |
| Test AUC            | ✔               |
| Best Threshold      | ✔ (per horizon) |

---

 4️⃣ Model Saving

Each final LightGBM pipeline is saved as:

```
model_h{horizon}_top15_improved.pkl
```

These can be loaded with:

```python
import joblib
model = joblib.load("model_h3_top15_improved.pkl")
```

---

 5️⃣ Explainability with SHAP

For each model, SHAP generates:

 ✔ SHAP Summary Plot

Shows how features contribute to predictions.

 ✔ SHAP Bar Plot

Shows mean absolute SHAP value for each feature.

Example code snippet:

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed)

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feats,
    max_display=20
)
```

---

# 📊 Sample SHAP Output

*(You can insert images here once SHAP outputs are exported.)*

* SHAP Beeswarm Plot
* SHAP Bar Importance Plot

---

# 📦 Installation

 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Notebook

Simply open:

```
Final_Master_Code.ipynb
```

and run all cells to:

* Train all Top-15 models
* Find optimal thresholds
* Evaluate test performance
* Generate SHAP plots
* Save trained models

---

# 🧪 Performance Summary

Each horizon achieves:

* High accuracy (0.96–0.98)
* Strong F1 scores (0.63–0.76+)
* Excellent AUC (0.92–0.96+)

Threshold optimization significantly improves F1 against default 0.50 cutoffs.

---

# 🛠 Tools Used

| Tool               | Purpose                      |
| ------------------ | ---------------------------- |
| LightGBM       | Gradient boosting model      |
| SHAP           | Explainability               |
| scikit-learn   | Splitting, metrics, pipeline |
| pandas / numpy | Data wrangling               |
| joblib         | Model persistence            |
| matplotlib     | Visualizations               |



