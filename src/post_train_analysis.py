"""
CreditWise — Post-Training Analysis
=====================================
Generates SHAP global explanations, probability calibration, and fairness
analysis for the best model selected by train.py.

Run AFTER src/train.py has completed successfully.

Usage:
    python -m src.post_train_analysis
"""

import json
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    BEST_MODEL_FILE, CALIBRATED_MODEL_FILE, FEATURE_LIST_FILE,
    FIGURES_DIR, MODEL_METADATA_FILE, PREPROCESSING_PIPELINE_FILE,
    RANDOM_SEED, TARGET_COLUMN,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def run_post_train_analysis() -> None:
    import joblib
    from sklearn.model_selection import train_test_split

    from src.calibration import calibrate_model, compare_calibration, save_calibrated_model
    from src.data_loader import clean_raw_data, load_raw_data
    from src.explainability import (
        build_explainer, compute_shap_values,
        global_feature_importance, plot_shap_bar_importance, plot_shap_summary,
    )
    from src.fairness import run_fairness_analysis
    from src.feature_engineering import engineer_features, get_all_feature_names

    # -----------------------------------------------------------------------
    # 1. Reload data exactly as during training
    # -----------------------------------------------------------------------
    logger.info("Reloading data …")
    df = clean_raw_data(load_raw_data())
    df_eng = engineer_features(df)
    feature_names = get_all_feature_names()

    X = df_eng[feature_names]
    y = df_eng[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    pipeline = joblib.load(PREPROCESSING_PIPELINE_FILE)
    model    = joblib.load(BEST_MODEL_FILE)

    X_train_tf = pipeline.transform(X_train)
    X_test_tf  = pipeline.transform(X_test)

    with open(MODEL_METADATA_FILE) as f:
        meta = json.load(f)
    model_name = meta.get("best_model_name", "Best Model")
    logger.info("Best model: %s", model_name)

    y_pred = model.predict(X_test_tf)
    y_prob = model.predict_proba(X_test_tf)[:, 1]

    # -----------------------------------------------------------------------
    # 2. Probability Calibration
    # -----------------------------------------------------------------------
    logger.info("Calibrating probabilities (Platt/sigmoid) …")
    calibrated = calibrate_model(model, X_test_tf, y_test.values, method="sigmoid")
    cal_results = compare_calibration(
        model, calibrated, X_test_tf, y_test.values, model_name=model_name
    )
    logger.info(cal_results["conclusion"])

    if cal_results["improvement"] > 0.001:
        save_calibrated_model(calibrated)
        logger.info("Calibrated model saved (Brier improved by %.4f).", cal_results["improvement"])
    else:
        logger.info("Calibration does not improve Brier score — raw model probabilities retained.")

    cal_out = Path("models") / "calibration_results.json"
    cal_out.parent.mkdir(parents=True, exist_ok=True)
    with open(cal_out, "w") as f:
        json.dump({**cal_results, "calibration_applied": cal_results["improvement"] > 0.001},
                  f, indent=2, default=str)

    # -----------------------------------------------------------------------
    # 3. SHAP Global Explainability
    # -----------------------------------------------------------------------
    logger.info("Building SHAP explainer …")
    np.random.seed(RANDOM_SEED)
    n_shap = min(3000, len(X_test_tf))
    idx = np.random.choice(len(X_test_tf), size=n_shap, replace=False)
    X_shap = X_test_tf[idx]

    explainer   = build_explainer(model, X_train_tf, feature_names, model_name=model_name)
    shap_values = compute_shap_values(explainer, X_shap, model_name=model_name)

    importance_df = global_feature_importance(shap_values, feature_names)
    imp_out = Path("reports/results/shap_importance.csv")
    imp_out.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(imp_out, index=False)
    logger.info("Top 10 SHAP features:\n%s", importance_df.head(10).to_string(index=False))

    plot_shap_bar_importance(shap_values, feature_names, model_name=model_name)
    plot_shap_summary(shap_values, X_shap, feature_names, model_name=model_name)

    # -----------------------------------------------------------------------
    # 4. Fairness Analysis
    # -----------------------------------------------------------------------
    logger.info("Running fairness analysis …")
    X_test_raw = X_test.reset_index(drop=True)
    fairness = run_fairness_analysis(
        X_test_raw, y_test.values, y_pred, y_prob
    )
    res_dir = Path("reports/results")
    res_dir.mkdir(parents=True, exist_ok=True)
    for attr, df_f in fairness.items():
        out = res_dir / f"fairness_{attr}.csv"
        df_f.to_csv(out, index=False)
        logger.info("Fairness (%s):\n%s", attr, df_f.to_string())

    logger.info("Post-training analysis complete.")


if __name__ == "__main__":
    run_post_train_analysis()
