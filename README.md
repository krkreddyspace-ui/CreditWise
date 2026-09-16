# CreditWise

## An Explainable AI Framework for Intelligent Credit Risk Assessment and Loan Decision Support

![CI Pipeline](https://github.com/krkreddyspace-ui/CreditWise/actions/workflows/ci.yml/badge.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49.0-FF4B4B)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

> **Academic Prototype Disclaimer**
> CreditWise is a college Major Project prototype. All predictions are estimates
> based on historical data and should NOT be treated as financial advice or
> autonomous lending decisions. SHAP explanations describe model behaviour,
> not real-world causation.

---

## Project Overview

CreditWise is an enterprise-grade end-to-end machine-learning framework that:

1. Predicts the **probability of credit default** (serious delinquency within 2 years) for a loan applicant using historical data.
2. Converts that probability into a **risk category** (Low / Medium / High).
3. Explains *why* the model produced that prediction using **SHAP** (SHapley Additive exPlanations).
4. Compares four ML algorithms (XGBoost, LightGBM, Random Forest, Logistic Regression) and selects the best by experimental evaluation.
5. Calibrates probability estimates using **Isotonic Regression** (Brier score improved **0.1134 → 0.0498**).
6. Serves predictions via an interactive **Superdesign Streamlit Dashboard** and a production **FastAPI REST Microservice**.

---

## Quick Start

### Option A: Running with Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/krkreddyspace-ui/CreditWise.git
cd CreditWise

# Launch both Web Dashboard (8501) and REST API (8000)
docker compose up --build
```

- **Interactive Streamlit Web Dashboard**: `http://localhost:8501`
- **FastAPI OpenAPI Swagger Documentation**: `http://localhost:8000/docs`

---

### Option B: Local Environment Setup

#### 1. Install dependencies

```bash
pip install -r requirements.txt
```

#### 2. Run the training & post-analysis pipeline

```bash
# Basic training pipeline
python main.py

# With hyperparameter tuning
python -m src.train --tune
```

#### 3. Launch the Web Dashboard

```bash
streamlit run app/app.py
```

#### 4. Launch the REST API Server

```bash
uvicorn src.api:app --reload --port 8000
```

#### 5. Run automated test suite (30/30 tests)

```bash
pytest tests/ -v
```

---

## REST API Documentation

CreditWise exposes high-performance REST API endpoints for enterprise integration:

| Endpoint | Method | Description |
|---|---|---|
| `GET /health` | GET | System readiness check, loaded model name, feature count |
| `POST /predict` | POST | Single applicant risk prediction & decision support |
| `POST /predict/batch` | POST | Bulk high-throughput applicant batch inference |
| `POST /explain` | POST | Local SHAP directional feature contribution breakdown |
| `GET /docs` | GET | Interactive OpenAPI Swagger UI |

### Example REST API Request (`POST /predict`)

```json
{
  "RevolvingUtilizationOfUnsecuredLines": 0.25,
  "age": 42,
  "NumberOfTime30-59DaysPastDueNotWorse": 0,
  "DebtRatio": 0.30,
  "MonthlyIncome": 6000.0,
  "NumberOfOpenCreditLinesAndLoans": 8,
  "NumberOfTimes90DaysLate": 0,
  "NumberRealEstateLoansOrLines": 1,
  "NumberOfTime60-89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 1
}
```

### Example API Response

```json
{
  "probability": 0.0421,
  "probability_percentage": 4.21,
  "risk_category": "Low Risk",
  "decision_support": "Favorable Decision Support — Low estimated risk profile.",
  "disclaimer": "CreditWise is an academic decision-support prototype..."
}
```

---

## Project Structure

```
CreditWise/
├── .github/
│   └── workflows/
│       └── ci.yml               ← GitHub Actions CI pipeline
│
├── app/                         ← Streamlit Web Application (Superdesign Theme)
│   ├── app.py                   ← Main Streamlit router & page coordinator
│   ├── components/              ← Modular UI components (sidebar, cards, gauge, charts)
│   └── styles/
│       └── theme.css            ← Custom Superdesign dark fintech CSS system
│
├── data/
│   ├── raw/                     ← cs-training.csv location
│   ├── processed/               ← Preprocessed numpy arrays
│   └── README.md
│
├── models/                      ← Saved model artefacts
│   ├── best_model.joblib
│   ├── calibrated_model.joblib
│   ├── preprocessing_pipeline.joblib
│   ├── feature_list.json
│   └── model_metadata.json
│
├── notebooks/                   ← Academic Jupyter analysis suite (01 to 04)
│
├── reports/
│   ├── figures/                 ← Saved ROC, PR, SHAP, Calibration plots
│   └── results/                 ← Evaluation metric CSVs
│
├── src/                         ← Production Python backend & REST API
│   ├── api.py                   ← FastAPI REST microservice
│   ├── calibration.py           ← Isotonic probability calibration
│   ├── config.py                ← Central system configurations
│   ├── data_loader.py           ← Data loading & validation
│   ├── evaluate.py              ← Model metrics & plot generation
│   ├── explainability.py        ← SHAP global & local explainers
│   ├── fairness.py              ← Subgroup demographic fairness
│   ├── feature_engineering.py   ← 7 domain derived features
│   ├── predict.py               ← Single & batch inference
│   ├── preprocessing.py         ← scikit-learn Pipeline
│   └── train.py                 ← Model training pipeline
│
├── tests/                       ← Automated test suite (30 tests)
│   ├── test_api.py              ← REST API endpoint tests
│   └── test_pipeline.py         ← Pipeline & model unit tests
│
├── Dockerfile                   ← Multi-stage Docker build
├── docker-compose.yml           ← Multi-container orchestration
├── requirements.txt             ← Python dependency specifications
└── README.md
```

---

## Model Evaluation (Test Set — 29,879 records)

| Model | Accuracy | Recall | F1 Score | ROC-AUC | PR-AUC | Brier Score |
|---|---|---|---|---|---|---|
| **XGBoost (Calibrated)** ⭐ | **0.8473** | **0.7038** | **0.3818** | **0.8599** | **0.3938** | **0.0498** |
| **LightGBM** | 0.8292 | 0.7238 | 0.3622 | 0.8554 | 0.3833 | 0.1210 |
| **Logistic Regression** | 0.7981 | 0.7507 | 0.3325 | 0.8528 | 0.3611 | 0.1499 |
| **Random Forest** | 0.9221 | 0.3681 | 0.3878 | 0.8442 | 0.3412 | 0.0597 |

---

## Risk Categories

| Default Probability | Category | Decision Support |
|---|---|---|
| 0.00 – 0.30 | 🟢 Low Risk | Favorable Decision Support |
| 0.30 – 0.60 | 🟡 Medium Risk | Manual Review Recommended |
| 0.60 – 1.00 | 🔴 High Risk | High Risk — Further Review Required |

---

## Academic Context & Disclaimer

**Project Title**: CreditWise: An Explainable AI Framework for Intelligent Credit Risk Assessment and Loan Decision Support

**System positioning**: AI-assisted decision support, NOT autonomous loan approval.
