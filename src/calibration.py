"""
CreditWise — Probability Calibration
=======================================
Evaluates whether the best model's predicted probabilities are well-calibrated
(i.e., a predicted probability of 0.7 corresponds to a 70% observed default rate).

Techniques evaluated
--------------------
* Platt scaling (sigmoid calibration)
* Isotonic regression

Metrics
-------
* Brier score (lower is better — 0 = perfect, 0.25 = random for balanced data)
* Reliability curve (calibration curve)

Academic note
-------------
Calibration improves interpretability of probabilities but may not
improve discrimination metrics (ROC-AUC / PR-AUC). If calibration does
not improve performance, that result is reported honestly — it is not
forced into the final pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from src.config import CALIBRATED_MODEL_FILE, FIGURES_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_model(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "sigmoid",
) -> object:
    """Wrap an already-fitted model with Platt scaling or isotonic regression.

    Parameters
    ----------
    model : fitted estimator
        The already-fitted best model from training.
    X_val : np.ndarray
        Held-out validation/test features (NOT the training set).
    y_val : np.ndarray
        True labels for the validation set.
    method : str
        'sigmoid' (Platt scaling) or 'isotonic'.

    Returns
    -------
    CalibratedClassifierCV
        Fitted calibration wrapper.

    Notes
    -----
    cv='prefit' tells sklearn to use the provided model as-is and only
    fit the calibration layer on X_val / y_val.
    """
    calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_val, y_val)
    logger.info("Calibration complete (%s). Returning calibrated model.", method)
    return calibrated


def compare_calibration(
    model,
    calibrated_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 10,
    save_dir: Path = FIGURES_DIR,
) -> Dict:
    """Compare raw vs calibrated probabilities on the test set.

    Parameters
    ----------
    model : fitted estimator
        The raw (uncalibrated) model.
    calibrated_model : fitted estimator
        The calibrated wrapper from calibrate_model().
    X_test, y_test : arrays
        Held-out test data.
    model_name : str
    n_bins : int
        Number of bins for the calibration curve.
    save_dir : Path

    Returns
    -------
    dict with:
        - 'brier_raw'        : float
        - 'brier_calibrated' : float
        - 'improvement'      : float (positive = calibration helped)
        - 'conclusion'       : str
    """
    prob_raw = model.predict_proba(X_test)[:, 1]
    prob_cal = calibrated_model.predict_proba(X_test)[:, 1]

    brier_raw = brier_score_loss(y_test, prob_raw)
    brier_cal = brier_score_loss(y_test, prob_cal)
    improvement = brier_raw - brier_cal   # positive = calibration helped

    if improvement > 0.001:
        conclusion = (
            f"Calibration improved Brier score by {improvement:.4f} "
            f"({brier_raw:.4f} → {brier_cal:.4f}). Using calibrated model."
        )
    else:
        conclusion = (
            f"Calibration did not meaningfully improve Brier score "
            f"({brier_raw:.4f} → {brier_cal:.4f}). "
            "Raw model probabilities are already well-behaved."
        )

    logger.info(conclusion)

    # Plot calibration curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, prob, label in [
        (axes[0], prob_raw, "Raw"),
        (axes[1], prob_cal, "Calibrated"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=n_bins, strategy="uniform")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly Calibrated")
        ax.plot(mean_pred, frac_pos, marker="o", lw=2, color="#4C72B0", label=label)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"{model_name} — {label} (Brier = {brier_score_loss(y_test, prob):.4f})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Calibration Comparison", fontsize=13)
    plt.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = save_dir / "calibration_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Calibration comparison plot saved: %s", fname)

    return {
        "brier_raw": round(brier_raw, 4),
        "brier_calibrated": round(brier_cal, 4),
        "improvement": round(improvement, 4),
        "conclusion": conclusion,
    }


def save_calibrated_model(calibrated_model, path: Path = CALIBRATED_MODEL_FILE) -> None:
    """Persist the calibrated model with joblib."""
    joblib.dump(calibrated_model, path)
    logger.info("Calibrated model saved: %s", path)
