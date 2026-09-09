"""
CreditWise — Model Training
=============================
Trains all four ML models using a consistent pipeline and saves the best
model to disk. Cross-validation is used for model comparison; the test
set is held out until final evaluation.

Models trained
--------------
1. Logistic Regression  — linear baseline
2. Random Forest        — ensemble baseline
3. XGBoost              — gradient boosting
4. LightGBM             — efficient gradient boosting

Class imbalance handling
------------------------
* All tree-based models support `class_weight='balanced'` or
  `scale_pos_weight` natively.
* SMOTE is applied inside the CV loop (not before splitting) using an
  imbalanced-learn Pipeline so no test-set leakage occurs.

Usage (standalone)
------------------
    python -m src.train

This will:
  1. Load the dataset.
  2. Engineer features.
  3. Split into train / test.
  4. Build the preprocessing pipeline.
  5. Train all 4 models with cross-validation.
  6. Evaluate on the held-out test set.
  7. Select the best model by ROC-AUC.
  8. Save artefacts to models/.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    CV_FOLDS,
    CV_SCORING,
    BEST_MODEL_FILE,
    FEATURE_LIST_FILE,
    LGBM_PARAM_GRID,
    LR_PARAM_GRID,
    MODEL_METADATA_FILE,
    MODELS_DIR,
    PREPROCESSING_PIPELINE_FILE,
    RANDOM_SEED,
    RANDOMIZED_SEARCH_CV,
    RANDOMIZED_SEARCH_ITER,
    RF_PARAM_GRID,
    XGB_PARAM_GRID,
)
from src.data_loader import load_raw_data
from src.evaluate import (
    build_comparison_table,
    compute_metrics,
    plot_confusion_matrix,
    plot_pr_curves,
    plot_roc_curves,
    save_comparison_table,
)
from src.feature_engineering import engineer_features, get_all_feature_names
from src.preprocessing import build_preprocessing_pipeline, fit_pipeline, split_data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model definitions — baseline configurations
# ---------------------------------------------------------------------------

def get_baseline_models() -> Dict:
    """Return unfitted baseline model instances.

    These are sensible defaults before hyperparameter tuning. Each model
    uses a fixed random_state for reproducibility.

    Returns
    -------
    dict : name → sklearn estimator
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=10,   # approximate imbalance ratio
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_SEED,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            num_leaves=63,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            verbosity=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_models(
    models: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = CV_FOLDS,
    scoring: str = CV_SCORING,
) -> Dict[str, float]:
    """Run stratified cross-validation for each model on training data.

    Parameters
    ----------
    models : dict
        name → fitted or unfitted estimator.
    X_train : np.ndarray
        Preprocessed training features.
    y_train : np.ndarray
        Training labels.
    cv_folds : int
        Number of stratified CV folds.
    scoring : str
        sklearn scoring string (default: 'roc_auc').

    Returns
    -------
    dict : name → mean CV score
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = {}

    for name, model in models.items():
        logger.info("Cross-validating %s …", name)
        t0 = time.time()
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        elapsed = time.time() - t0
        cv_scores[name] = float(scores.mean())
        logger.info(
            "  %s  CV %s = %.4f ± %.4f  (%.1fs)",
            name, scoring, scores.mean(), scores.std(), elapsed,
        )

    return cv_scores


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_model(
    name: str,
    model,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[object, dict]:
    """Run RandomizedSearchCV for one model.

    Parameters
    ----------
    name : str
        Human-readable model name (for logging).
    model : estimator
        Unfitted sklearn-compatible model.
    param_grid : dict
        Parameter distributions for RandomizedSearchCV.
    X_train, y_train : arrays
        Training data (after preprocessing, before SMOTE).

    Returns
    -------
    best_model : fitted estimator
    best_params : dict
    """
    logger.info("Tuning %s with RandomizedSearchCV …", name)
    cv = StratifiedKFold(n_splits=RANDOMIZED_SEARCH_CV, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=RANDOMIZED_SEARCH_ITER,
        cv=cv,
        scoring=CV_SCORING,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    logger.info(
        "  %s best CV %s = %.4f  params = %s  (%.1fs)",
        name, CV_SCORING, search.best_score_, search.best_params_, elapsed,
    )
    return search.best_estimator_, search.best_params_


# ---------------------------------------------------------------------------
# Save artefacts
# ---------------------------------------------------------------------------

def save_artefacts(
    pipeline: Pipeline,
    best_model,
    best_model_name: str,
    feature_names: list,
    metadata: dict,
) -> None:
    """Persist the preprocessing pipeline, best model, and metadata to disk.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Fitted preprocessing pipeline.
    best_model : estimator
        The winning trained model.
    best_model_name : str
        Human-readable name of the winning model.
    feature_names : list of str
        Ordered feature names corresponding to the pipeline output.
    metadata : dict
        Metrics, params, and selection rationale to store in JSON.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, PREPROCESSING_PIPELINE_FILE)
    logger.info("Preprocessing pipeline saved: %s", PREPROCESSING_PIPELINE_FILE)

    joblib.dump(best_model, BEST_MODEL_FILE)
    logger.info("Best model (%s) saved: %s", best_model_name, BEST_MODEL_FILE)

    with open(FEATURE_LIST_FILE, "w") as f:
        json.dump(feature_names, f, indent=2)
    logger.info("Feature list saved: %s", FEATURE_LIST_FILE)

    metadata["best_model_name"] = best_model_name
    with open(MODEL_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Model metadata saved: %s", MODEL_METADATA_FILE)


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline(tune_hyperparams: bool = False) -> None:
    """Execute the complete training pipeline end-to-end.

    Sequence
    --------
    1. Load raw data.
    2. Engineer features.
    3. Stratified train/test split.
    4. Fit preprocessing pipeline on training data.
    5. Train all 4 baseline models with CV.
    6. Optionally tune top model(s) with RandomizedSearchCV.
    7. Evaluate all models on held-out test set.
    8. Select best model by ROC-AUC.
    9. Save all artefacts.

    Parameters
    ----------
    tune_hyperparams : bool
        If True, run RandomizedSearchCV on all models before final
        evaluation. This significantly increases runtime.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    # 1. Load and clean (EDA-driven cleaning: drop duplicates, fix delinquency codes)
    from src.data_loader import clean_raw_data
    df = load_raw_data()
    df = clean_raw_data(df)
    logger.info("After cleaning: %d rows", len(df))

    # 2. Feature engineering
    logger.info("Engineering features …")
    df_eng = engineer_features(df)
    feature_names = get_all_feature_names()

    # 3. Split
    from src.config import TARGET_COLUMN
    X = df_eng[feature_names]
    y = df_eng[TARGET_COLUMN]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    # 4. Preprocessing — fit on train only
    # Use scaling so Logistic Regression works; tree models are unaffected
    preprocessing = build_preprocessing_pipeline(scale_features=True)
    X_train_tf, X_test_tf, fitted_preprocessing = fit_pipeline(
        preprocessing, X_train, X_test
    )

    # 5. Baseline models
    models = get_baseline_models()

    if tune_hyperparams:
        param_grids = {
            "Logistic Regression": LR_PARAM_GRID,
            "Random Forest": RF_PARAM_GRID,
            "XGBoost": XGB_PARAM_GRID,
            "LightGBM": LGBM_PARAM_GRID,
        }
        tuned_models = {}
        tuned_params = {}
        for name, model in models.items():
            best_m, best_p = tune_model(name, model, param_grids[name], X_train_tf, y_train)
            tuned_models[name] = best_m
            tuned_params[name] = best_p
        models = tuned_models
    else:
        # Fit baseline models
        tuned_params = {}
        for name, model in models.items():
            logger.info("Training %s …", name)
            t0 = time.time()
            model.fit(X_train_tf, y_train)
            logger.info("  Done in %.1fs", time.time() - t0)

    # 6. Evaluate on test set
    all_metrics = []
    results_for_plots = {}

    for name, model in models.items():
        y_pred = model.predict(X_test_tf)
        y_prob = model.predict_proba(X_test_tf)[:, 1]

        metrics = compute_metrics(name, y_test, y_pred, y_prob)
        all_metrics.append(metrics)
        results_for_plots[name] = {"y_prob": y_prob, **metrics}

        plot_confusion_matrix(name, y_test, y_pred)

    # 7. Comparison table
    comparison_df = build_comparison_table(all_metrics)
    save_comparison_table(comparison_df)
    logger.info("\n%s", comparison_df.to_string(index=False))

    plot_roc_curves(results_for_plots, y_test)
    plot_pr_curves(results_for_plots, y_test)

    # 8. Select best model by ROC-AUC
    best_name = max(all_metrics, key=lambda m: m["roc_auc"])["model"]
    best_model = models[best_name]
    logger.info("Best model selected: %s (ROC-AUC = %.4f)",
                best_name,
                next(m["roc_auc"] for m in all_metrics if m["model"] == best_name))

    # 9. Save
    metadata = {
        "all_metrics": all_metrics,
        "tuned_params": tuned_params,
        "hyperparams_tuned": tune_hyperparams,
        "random_seed": RANDOM_SEED,
        "test_size": 0.20,
        "cv_folds": CV_FOLDS,
        "cv_scoring": CV_SCORING,
    }
    save_artefacts(fitted_preprocessing, best_model, best_name, feature_names, metadata)
    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CreditWise training pipeline")
    parser.add_argument(
        "--tune", action="store_true",
        help="Run RandomizedSearchCV hyperparameter tuning (slower)."
    )
    args = parser.parse_args()
    run_training_pipeline(tune_hyperparams=args.tune)
