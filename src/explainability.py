"""
CreditWise — SHAP Explainability
==================================
Generates both global and local SHAP explanations for the best trained model.

Important interpretation note (must be surfaced in the UI and report)
----------------------------------------------------------------------
"SHAP values show how each feature contributed to the model's prediction
for a specific input or across the dataset. They do NOT establish that a
feature caused the real-world outcome — they explain the model's internal
behaviour, not ground truth."

Global explainability
---------------------
* Feature importance (mean |SHAP|)
* SHAP summary plot (beeswarm)

Local explainability (individual applicant)
-------------------------------------------
* SHAP waterfall / bar plot for one prediction
* Signed feature contributions (+ = increases risk, − = decreases risk)

Explainer selection
-------------------
* Tree-based models (RF, XGBoost, LightGBM): TreeExplainer (fast, exact)
* Logistic Regression: LinearExplainer (exact for linear models)
* Fallback: KernelExplainer (slow but model-agnostic)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/script use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import FIGURES_DIR, SHAP_BACKGROUND_SAMPLES, SHAP_MAX_DISPLAY

logger = logging.getLogger(__name__)

# Model type → explainer mapping
TREE_MODEL_TYPES = ("RandomForestClassifier", "XGBClassifier", "LGBMClassifier")
LINEAR_MODEL_TYPES = ("LogisticRegression",)


# ---------------------------------------------------------------------------
# Explainer factory
# ---------------------------------------------------------------------------

def build_explainer(
    model,
    X_background: np.ndarray,
    feature_names: List[str],
    model_name: str = "",
) -> shap.Explainer:
    """Select and build the appropriate SHAP explainer for *model*.

    Parameters
    ----------
    model : sklearn/xgboost/lightgbm estimator
        Fitted model. Must have a `predict_proba` method.
    X_background : np.ndarray
        Background dataset (training data sample) for KernelExplainer.
        Not used for Tree/Linear explainers but passed for API consistency.
    feature_names : list of str
        Feature names in column order.
    model_name : str
        Human-readable model name for logging.

    Returns
    -------
    shap.Explainer (TreeExplainer, LinearExplainer, or KernelExplainer)
    """
    model_type = type(model).__name__

    if model_type in TREE_MODEL_TYPES:
        logger.info("Using TreeExplainer for %s", model_name or model_type)
        # model_output="raw" is required for tree_path_dependent perturbation (the default).
        # Raw SHAP values are in the log-odds / margin space for classifiers.
        # Signs and rankings are equivalent to probability-space explanations;
        # only the absolute scale differs. We note this in the UI.
        explainer = shap.TreeExplainer(
            model,
            feature_names=feature_names,
        )

    elif model_type in LINEAR_MODEL_TYPES:
        logger.info("Using LinearExplainer for %s", model_name or model_type)
        explainer = shap.LinearExplainer(
            model,
            X_background,
            feature_names=feature_names,
        )

    else:
        logger.warning(
            "Model type '%s' not natively supported — falling back to "
            "KernelExplainer (this may be slow).", model_type
        )
        background = shap.sample(X_background, SHAP_BACKGROUND_SAMPLES)
        explainer = shap.KernelExplainer(
            model.predict_proba,
            background,
            feature_names=feature_names,
        )

    return explainer


# ---------------------------------------------------------------------------
# Global explanations
# ---------------------------------------------------------------------------

def compute_shap_values(
    explainer: shap.Explainer,
    X: np.ndarray,
    model_name: str = "",
) -> shap.Explanation:
    """Compute SHAP values for a dataset.

    For TreeExplainer this is fast even on the full test set.
    For KernelExplainer, sample X to a manageable size first.

    Parameters
    ----------
    explainer : shap.Explainer
        Built by build_explainer().
    X : np.ndarray
        Feature matrix to explain (rows = samples, cols = features).
    model_name : str
        For logging purposes.

    Returns
    -------
    shap.Explanation object with .values, .base_values, .data attributes.
    """
    logger.info("Computing SHAP values for %d samples …", len(X))
    # For TreeExplainer with model_output='probability', shap_values is
    # returned as a single array for the positive class.
    shap_values = explainer(X)

    # If multi-output (some tree explainers return per-class), take class 1
    if isinstance(shap_values.values, np.ndarray) and shap_values.values.ndim == 3:
        logger.debug("Multi-class SHAP output detected — selecting class 1 (default).")
        return shap.Explanation(
            values=shap_values.values[:, :, 1],
            base_values=shap_values.base_values[:, 1]
            if shap_values.base_values.ndim > 1
            else shap_values.base_values,
            data=shap_values.data,
            feature_names=shap_values.feature_names,
        )

    return shap_values


def global_feature_importance(
    shap_values: shap.Explanation,
    feature_names: List[str],
) -> pd.DataFrame:
    """Compute mean |SHAP| importance ranking.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values from compute_shap_values().
    feature_names : list of str

    Returns
    -------
    pd.DataFrame
        Columns: ['feature', 'mean_abs_shap'].
        Sorted by importance descending.
    """
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df


def plot_shap_summary(
    shap_values: shap.Explanation,
    X: np.ndarray,
    feature_names: List[str],
    model_name: str = "",
    save_dir: Path = FIGURES_DIR,
    max_display: int = SHAP_MAX_DISPLAY,
) -> Path:
    """Generate and save a SHAP beeswarm summary plot.

    Parameters
    ----------
    shap_values : shap.Explanation
    X : np.ndarray
        Original (unscaled) or scaled feature matrix used for colour values.
    feature_names : list of str
    model_name : str
        Appended to filename.
    save_dir : Path
    max_display : int
        Maximum number of features to display.

    Returns
    -------
    Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.45)))
    shap.summary_plot(
        shap_values.values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.title(f"SHAP Summary — {model_name}", fontsize=13, pad=10)
    plt.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = model_name.lower().replace(" ", "_") or "model"
    fname = save_dir / f"shap_summary_{suffix}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved: %s", fname)
    return fname


def plot_shap_bar_importance(
    shap_values: shap.Explanation,
    feature_names: List[str],
    model_name: str = "",
    save_dir: Path = FIGURES_DIR,
    max_display: int = SHAP_MAX_DISPLAY,
) -> Path:
    """Bar chart of mean |SHAP| feature importance."""
    fig, ax = plt.subplots(figsize=(10, max(5, max_display * 0.4)))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title(f"SHAP Feature Importance — {model_name}", fontsize=13)
    plt.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = model_name.lower().replace(" ", "_") or "model"
    fname = save_dir / f"shap_bar_importance_{suffix}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP bar importance plot saved: %s", fname)
    return fname


# ---------------------------------------------------------------------------
# Local explanations (individual applicant)
# ---------------------------------------------------------------------------

def explain_single(
    explainer: shap.Explainer,
    x_single: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
) -> Dict:
    """Compute SHAP explanation for one applicant.

    Parameters
    ----------
    explainer : shap.Explainer
    x_single : np.ndarray
        Single-row feature array (shape: [1, n_features]).
    feature_names : list of str
    top_n : int
        Number of top contributors to return.

    Returns
    -------
    dict with keys:
        - 'base_value'    : float (model expected value / intercept)
        - 'prediction'    : float (predicted probability for this sample)
        - 'contributions' : list of dicts [{feature, value, shap_value}, ...]
                            sorted by |shap_value| descending
        - 'top_increasing': features most strongly pushing probability UP
        - 'top_decreasing': features most strongly pushing probability DOWN
    """
    shap_exp = explainer(x_single)

    # Handle multi-class output
    if shap_exp.values.ndim == 3:
        sv = shap_exp.values[0, :, 1]
        bv = float(shap_exp.base_values[0, 1]) if shap_exp.base_values.ndim > 1 \
            else float(shap_exp.base_values[0])
    else:
        sv = shap_exp.values[0]
        bv = float(shap_exp.base_values[0]) if np.ndim(shap_exp.base_values) > 0 \
            else float(shap_exp.base_values)

    prediction = float(bv + sv.sum())
    prediction = float(np.clip(prediction, 0.0, 1.0))

    contributions = [
        {
            "feature": feature_names[i],
            "feature_value": float(x_single[0, i]) if x_single.ndim == 2 else float(x_single[i]),
            "shap_value": float(sv[i]),
        }
        for i in range(len(feature_names))
    ]
    contributions.sort(key=lambda d: abs(d["shap_value"]), reverse=True)

    top_increasing = [c for c in contributions if c["shap_value"] > 0][:top_n]
    top_decreasing = [c for c in contributions if c["shap_value"] < 0][:top_n]

    return {
        "base_value": bv,
        "prediction": prediction,
        "contributions": contributions[:top_n],
        "top_increasing": top_increasing,
        "top_decreasing": top_decreasing,
    }


def plot_local_waterfall(
    explainer: shap.Explainer,
    x_single: np.ndarray,
    feature_names: List[str],
    applicant_id: str = "applicant",
    save_dir: Path = FIGURES_DIR,
) -> Path:
    """Generate and save a SHAP waterfall plot for one applicant."""
    shap_exp = explainer(x_single)

    if shap_exp.values.ndim == 3:
        exp = shap.Explanation(
            values=shap_exp.values[0, :, 1],
            base_values=float(shap_exp.base_values[0, 1])
            if shap_exp.base_values.ndim > 1 else float(shap_exp.base_values[0]),
            data=shap_exp.data[0],
            feature_names=feature_names,
        )
    else:
        exp = shap.Explanation(
            values=shap_exp.values[0],
            base_values=float(shap_exp.base_values[0]),
            data=shap_exp.data[0],
            feature_names=feature_names,
        )

    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, max_display=12, show=False)
    plt.title(f"SHAP Explanation — {applicant_id}", fontsize=12)
    plt.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = save_dir / f"shap_waterfall_{applicant_id.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP waterfall plot saved: %s", fname)
    return fname
