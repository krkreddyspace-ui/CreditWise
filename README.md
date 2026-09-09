# CreditWise

## An Explainable AI Framework for Intelligent Credit Risk Assessment and Loan Decision Support

> **Academic Prototype Disclaimer**
> CreditWise is a college Major Project prototype. All predictions are estimates
> based on historical data and should NOT be treated as financial advice or
> autonomous lending decisions. SHAP explanations describe model behaviour,
> not real-world causation.

---

## Project Overview

CreditWise is an end-to-end machine-learning system that:

1. Predicts the **probability of credit default** (serious delinquency within 2 years)
   for a loan applicant using historical data.
2. Converts that probability into a **risk category** (Low / Medium / High).
3. Explains *why* the model produced that prediction using **SHAP** (SHapley Additive exPlanations).
4. Compares four ML algorithms and selects the best by experimental evaluation.
5. Presents everything in an interactive **Streamlit dashboard**.

---

## Environment

| Requirement | Version |
|---|---|
| Python | 3.13.2 |
| scikit-learn | 1.7.2 |
| XGBoost | 3.1.1 |
| LightGBM | 4.7.0 |
| SHAP | 0.52.0 |
| Streamlit | 1.49.0 |

See [requirements.txt](requirements.txt) for the full list.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Obtain the dataset

Download `cs-training.csv` from the [Give Me Some Credit Kaggle competition](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place it at:

```
data/raw/cs-training.csv
```

See [data/README.md](data/README.md) for full instructions.

### 3. Run the training pipeline

```bash
# Basic training (uses class-weight imbalance handling)
python -m src.train

# With hyperparameter tuning (slower, ~20 iterations per model)
python -m src.train --tune
```

This will:
- Engineer features
- Split data (80% train / 20% test, stratified)
- Train Logistic Regression, Random Forest, XGBoost, LightGBM
- Evaluate all models on the held-out test set
- Select the best model by ROC-AUC
- Save all artefacts to `models/`
- Save plots to `reports/figures/`

### 4. Launch the dashboard

```bash
cd app
streamlit run app.py
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
CreditWise/
│
├── data/
│   ├── raw/             ← Place cs-training.csv here
│   ├── processed/       ← Auto-generated preprocessed arrays
│   └── README.md        ← Dataset instructions
│
├── notebooks/
│   ├── 01_eda.ipynb             ← Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb   ← Preprocessing walkthrough
│   ├── 03_model_training.ipynb  ← Model comparison narrative
│   └── 04_model_explainability.ipynb ← SHAP analysis
│
├── src/
│   ├── config.py            ← All settings, paths, thresholds
│   ├── data_loader.py       ← Dataset loading + schema validation
│   ├── preprocessing.py     ← sklearn Pipeline (impute + scale)
│   ├── feature_engineering.py ← 7 justified derived features
│   ├── train.py             ← Full training pipeline
│   ├── evaluate.py          ← All metrics + plots
│   ├── predict.py           ← Inference + input validation
│   ├── explainability.py    ← SHAP global + local explanations
│   ├── calibration.py       ← Platt / isotonic calibration
│   └── fairness.py          ← Group-level fairness analysis
│
├── models/                  ← Saved artefacts (after training)
│   ├── preprocessing_pipeline.joblib
│   ├── best_model.joblib
│   ├── feature_list.json
│   ├── model_metadata.json
│   └── risk_thresholds.json
│
├── reports/
│   ├── figures/             ← All saved plots
│   └── results/
│       └── model_comparison.csv
│
├── app/
│   └── app.py               ← Streamlit dashboard (6 pages)
│
├── tests/
│   └── test_pipeline.py     ← pytest unit tests
│
├── requirements.txt
└── README.md
```

---

## Models Compared

| Model | Type | Class Imbalance |
|---|---|---|
| Logistic Regression | Linear baseline | `class_weight='balanced'` |
| Random Forest | Ensemble | `class_weight='balanced'` |
| XGBoost | Gradient boosting | `scale_pos_weight` |
| LightGBM | Gradient boosting | `class_weight='balanced'` |

Selection criterion: **ROC-AUC** on the held-out test set.

---

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-score
- **ROC-AUC** (primary discrimination metric)
- **PR-AUC** (especially informative under class imbalance)
- Brier score (probability calibration quality)
- Confusion matrix (with explicit TP/TN/FP/FN identification)

---

## SHAP Explainability

- **Global**: Mean |SHAP| importance, beeswarm summary plot
- **Local**: Per-applicant waterfall plot, signed feature contributions
- Explainer: TreeExplainer for tree models, LinearExplainer for Logistic Regression

**Interpretation rule**: SHAP shows how features influenced the *model's prediction*.
It does NOT establish causation with real-world outcomes.

---

## Risk Categories

| Probability | Category | Decision Support |
|---|---|---|
| 0.00 – 0.30 | 🟢 Low Risk | Favorable |
| 0.30 – 0.60 | 🟡 Medium Risk | Review Recommended |
| 0.60 – 1.00 | 🔴 High Risk | Further Review Required |

Thresholds are configurable in `src/config.py`. These are not universal
banking thresholds — they are documented experimental defaults.

---

## Research Questions

| RQ | Question |
|---|---|
| RQ1 | Which ML algorithm achieves the best credit-risk prediction? |
| RQ2 | How does class imbalance affect default prediction performance? |
| RQ3 | How well calibrated are the predicted default probabilities? |
| RQ4 | Which features most strongly influence credit-risk predictions? |
| RQ5 | Can SHAP provide useful applicant-level explanations? |
| RQ6 | Are there measurable performance disparities across age groups? |
| RQ7 | What is the trade-off between predictive performance and interpretability? |

---

## Dataset

**Give Me Some Credit** (Kaggle)
- ~150,000 borrower records
- Target: `SeriousDlqin2yrs` (0 = no default, 1 = default)
- Class distribution: ~93% non-default, ~7% default (imbalanced)

Public dataset. No real personal financial information is used.

---

## Limitations

1. Model performance is dataset-specific — results may not generalise to other lending contexts.
2. Calibration may not improve all models equally.
3. Fairness analysis is limited to available proxy attributes (age, dependents).
4. This system is an academic prototype, NOT suitable for real-world lending decisions.
5. SHAP values explain model behaviour, not causal mechanisms.

---

## Academic Context

**Project Title**: CreditWise: An Explainable AI Framework for Intelligent Credit Risk Assessment and Loan Decision Support

**System positioning**: AI-assisted decision support, NOT autonomous loan approval.

All experimental results in this project are derived from actual model training on the public dataset. No results are fabricated.
