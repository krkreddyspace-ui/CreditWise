"""
CreditWise Configuration
========================
Central configuration for all paths, constants, model settings, and
risk thresholds. Modify values here rather than in individual modules.

All paths are constructed relative to the project root so the project
is portable across machines.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (one level above src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Primary dataset file — place cs-training.csv here
RAW_DATA_FILE = RAW_DATA_DIR / "cs-training.csv"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

# Saved artefact filenames
PREPROCESSING_PIPELINE_FILE = MODELS_DIR / "preprocessing_pipeline.joblib"
BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"
CALIBRATED_MODEL_FILE = MODELS_DIR / "calibrated_model.joblib"
FEATURE_LIST_FILE = MODELS_DIR / "feature_list.json"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
RISK_THRESHOLDS_FILE = MODELS_DIR / "risk_thresholds.json"
MODEL_COMPARISON_FILE = RESULTS_DIR / "model_comparison.csv"
SHAP_BACKGROUND_FILE = MODELS_DIR / "shap_background.joblib"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Dataset schema — Give Me Some Credit
# ---------------------------------------------------------------------------
TARGET_COLUMN = "SeriousDlqin2yrs"

# Original feature columns (as they appear in the raw CSV).
# The first unnamed index column is dropped automatically on load.
EXPECTED_FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20          # 80 / 20 split
STRATIFY_SPLIT = True     # stratify on target

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
CV_FOLDS = 5
CV_SCORING = "roc_auc"    # primary CV metric

# ---------------------------------------------------------------------------
# Risk probability thresholds
# Configurable — adjust based on experimental results.
# These are documented defaults; they are NOT universal banking thresholds.
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "low_max": 0.30,    # [0.00, 0.30)  → Low Risk
    "medium_max": 0.60, # [0.30, 0.60)  → Medium Risk
    # >= 0.60           → High Risk
}

RISK_LABELS = {
    "low": "Low Risk",
    "medium": "Medium Risk",
    "high": "High Risk",
}

DECISION_SUPPORT_LABELS = {
    "low": "Favorable",
    "medium": "Review Recommended",
    "high": "High Risk — Further Review Required",
}

# ---------------------------------------------------------------------------
# Class imbalance strategies to compare
# ---------------------------------------------------------------------------
IMBALANCE_STRATEGIES = ["class_weight", "smote", "random_oversample", "none"]

# ---------------------------------------------------------------------------
# Model hyperparameter search spaces (for RandomizedSearchCV)
# ---------------------------------------------------------------------------
LR_PARAM_GRID = {
    "C": [0.001, 0.01, 0.1, 1.0, 10.0],
    "solver": ["lbfgs", "saga"],
    "max_iter": [500, 1000],
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced", None],
}

XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "scale_pos_weight": [1, 5, 10],  # handles imbalance natively
}

LGBM_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [-1, 10, 20],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "num_leaves": [31, 63, 127],
    "subsample": [0.7, 0.8, 1.0],
    "class_weight": ["balanced", None],
}

RANDOMIZED_SEARCH_ITER = 20   # number of parameter combinations to try
RANDOMIZED_SEARCH_CV = 3      # inner CV folds during search

# ---------------------------------------------------------------------------
# SHAP settings
# ---------------------------------------------------------------------------
SHAP_BACKGROUND_SAMPLES = 100   # samples for KernelExplainer background
SHAP_MAX_DISPLAY = 15           # max features shown in summary plot

# ---------------------------------------------------------------------------
# Fairness analysis settings
# ---------------------------------------------------------------------------
# Age brackets used for group-level fairness analysis
AGE_BINS = [0, 30, 60, 120]
AGE_LABELS = ["Under 30", "30-60", "Over 60"]

# Dependents grouping
DEPENDENTS_BINS = [0, 1, 100]
DEPENDENTS_LABELS = ["No Dependents", "Has Dependents"]

# ---------------------------------------------------------------------------
# Application disclaimer (shown in UI)
# ---------------------------------------------------------------------------
APP_DISCLAIMER = (
    "CreditWise is an academic decision-support prototype. "
    "Predictions are estimates based on historical data and should NOT be "
    "treated as financial advice or an automated lending decision. "
    "SHAP values explain the model's prediction, not real-world causation."
)
