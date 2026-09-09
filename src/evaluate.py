"""
CreditWise — Model Evaluation
===============================
Computes, prints, and saves all evaluation metrics for every trained model.

Metrics computed
----------------
* Accuracy
* Precision (positive class)
* Recall (positive class)
* F1-score (positive class)
* ROC-AUC
* PR-AUC (Average Precision)
* Confusion matrix
* Brier score (probability quality)

Plots saved
-----------
* Confusion matrices (one per model)
* ROC curves (all models on one plot)
* Precision-Recall curves (all models on one plot)
* Calibration curves (if compare_calibration is called)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURES_DIR, MODEL_COMPARISON_FILE, RESULTS_DIR

logger = logging.getLogger(__name__)

# Consistent colour palette for models
MODEL_COLOURS = {
    "Logistic Regression": "#4C72B0",
    "Random Forest": "#55A868",
    "XGBoost": "#C44E52",
    "LightGBM": "#DD8452",
}


# ---------------------------------------------------------------------------
# Single-model metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict:
    """Compute the full set of classification metrics for one model.

    Parameters
    ----------
    model_name : str
        Human-readable model identifier (used as dict key and in plots).
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels (threshold 0.5 unless changed downstream).
    y_prob : array-like
        Predicted probability of the positive class (class 1).

    Returns
    -------
    dict
        All metrics keyed by metric name.
    """
    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "brier_score": round(brier_score_loss(y_true, y_prob), 4),
    }

    cm = confusion_matrix(y_true, y_pred)
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])

    logger.info(
        "[%s]  ACC=%.4f  AUC=%.4f  PR-AUC=%.4f  Recall=%.4f  F1=%.4f",
        model_name,
        metrics["accuracy"],
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["recall"],
        metrics["f1"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(all_metrics: List[Dict]) -> pd.DataFrame:
    """Convert a list of per-model metric dicts into a display DataFrame.

    Parameters
    ----------
    all_metrics : list of dict
        List returned by calling compute_metrics() for each model.

    Returns
    -------
    pd.DataFrame
        Rows = models, columns = metrics. Sorted by ROC-AUC descending.
    """
    display_cols = [
        "model", "accuracy", "precision", "recall",
        "f1", "roc_auc", "pr_auc", "brier_score",
    ]
    df = pd.DataFrame(all_metrics)[display_cols].sort_values("roc_auc", ascending=False)
    df = df.rename(columns={
        "model": "Model",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "brier_score": "Brier Score",
    })
    return df.reset_index(drop=True)


def save_comparison_table(df: pd.DataFrame, path: Path = MODEL_COMPARISON_FILE) -> None:
    """Save the comparison DataFrame to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Model comparison table saved to %s", path)


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: Path = FIGURES_DIR,
) -> None:
    """Plot and save a labelled confusion matrix for one model.

    Labels clearly identify TP / TN / FP / FN for the positive (default) class.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = [["TN\n(Predicted: No Default\nActual: No Default)",
                "FP\n(Predicted: Default\nActual: No Default)"],
               ["FN\n(Predicted: No Default\nActual: Default)",
                "TP\n(Predicted: Default\nActual: Default)"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted: No Default", "Predicted: Default"],
        yticklabels=["Actual: No Default", "Actual: Default"],
        ax=ax, linewidths=0.5,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = save_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved: %s", fname)


# ---------------------------------------------------------------------------
# ROC curve (all models on one plot)
# ---------------------------------------------------------------------------

def plot_roc_curves(
    results: Dict[str, Dict],
    y_true: np.ndarray,
    save_dir: Path = FIGURES_DIR,
) -> None:
    """Plot ROC curves for all models on a single axis.

    Parameters
    ----------
    results : dict
        Mapping of model_name → {'y_prob': array, 'roc_auc': float, ...}
    y_true : array-like
        True labels (shared across all models — same test set).
    save_dir : Path
        Directory to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_true, res["y_prob"])
        auc = res.get("roc_auc", roc_auc_score(y_true, res["y_prob"]))
        colour = MODEL_COLOURS.get(name, None)
        ax.plot(fpr, tpr, lw=2, label=f"{name}  (AUC = {auc:.4f})", color=colour)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves — Model Comparison", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = save_dir / "roc_curves_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("ROC curves saved: %s", fname)


# ---------------------------------------------------------------------------
# Precision-Recall curve (all models)
# ---------------------------------------------------------------------------

def plot_pr_curves(
    results: Dict[str, Dict],
    y_true: np.ndarray,
    save_dir: Path = FIGURES_DIR,
) -> None:
    """Plot Precision-Recall curves for all models.

    PR-AUC is particularly informative for imbalanced datasets because it
    focuses on the minority (default) class performance.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    baseline = y_true.mean()
    ax.axhline(baseline, color="k", linestyle="--", lw=1,
               label=f"Baseline (class freq = {baseline:.3f})")

    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(y_true, res["y_prob"])
        pr_auc = res.get("pr_auc", average_precision_score(y_true, res["y_prob"]))
        colour = MODEL_COLOURS.get(name, None)
        ax.plot(rec, prec, lw=2, label=f"{name}  (PR-AUC = {pr_auc:.4f})", color=colour)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Model Comparison", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_dir = Path(save_dir)
    fname = save_dir / "pr_curves_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("PR curves saved: %s", fname)


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------

def plot_calibration_curves(
    results: Dict[str, Dict],
    y_true: np.ndarray,
    n_bins: int = 10,
    save_dir: Path = FIGURES_DIR,
) -> None:
    """Plot reliability (calibration) curves for one or more models.

    A well-calibrated model's curve should closely follow the diagonal.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly Calibrated")

    for name, res in results.items():
        prob_true, prob_pred = calibration_curve(
            y_true, res["y_prob"], n_bins=n_bins, strategy="uniform"
        )
        colour = MODEL_COLOURS.get(name, None)
        ax.plot(prob_pred, prob_true, marker="o", lw=2, label=name, color=colour)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_dir = Path(save_dir)
    fname = save_dir / "calibration_curves.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Calibration curves saved: %s", fname)
