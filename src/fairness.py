"""
CreditWise — Fairness Analysis
================================
Computes group-level performance metrics to identify whether the model
exhibits measurable prediction disparities across subgroups.

IMPORTANT LIMITATIONS AND DISCLAIMERS
--------------------------------------
1. This analysis does NOT prove the model is fair or unfair.
2. Disparities measured here are model-level statistical observations.
3. The Give Me Some Credit dataset contains limited demographic attributes.
   We use 'age' (grouped into brackets) and 'NumberOfDependents' as
   available proxy attributes.
4. Neither age nor dependents directly represent protected characteristics
   in all jurisdictions, but age IS a protected attribute in lending
   contexts (Equal Credit Opportunity Act in the USA, for example).
5. These attributes are used ONLY for post-hoc analysis, NOT as model
   input features.
6. Correlation between these proxies and the true protected characteristics
   is unknown — interpret all results with appropriate caution.

Metrics reported (per group)
----------------------------
* Recall (True Positive Rate): how often actual defaults are caught
* False Negative Rate (FNR): defaults the model misses (= 1 - Recall)
* False Positive Rate (FPR): non-defaults incorrectly flagged as risky
* Selection Rate: fraction predicted as high-risk (using chosen threshold)
* ROC-AUC (if group is large enough)
"""

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    roc_auc_score,
)

from src.config import AGE_BINS, AGE_LABELS, DEPENDENTS_BINS, DEPENDENTS_LABELS, FIGURES_DIR

logger = logging.getLogger(__name__)

# Minimum group size to compute AUC meaningfully
MIN_GROUP_SIZE = 100


# ---------------------------------------------------------------------------
# Group metric computation
# ---------------------------------------------------------------------------

def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.Series,
    group_name: str,
    threshold: float = 0.30,
) -> pd.DataFrame:
    """Compute performance metrics separately for each group in *groups*.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    y_prob : array-like
        Predicted probabilities of the positive class.
    groups : pd.Series
        Categorical series of the same length as y_true, with one entry
        per sample indicating its group label.
    group_name : str
        Human-readable attribute name (for column labelling / plots).
    threshold : float
        Probability threshold used to define 'high-risk' for selection rate.

    Returns
    -------
    pd.DataFrame
        One row per group, columns = metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    groups = pd.Series(groups).reset_index(drop=True)

    records = []
    for grp in groups.unique():
        mask = groups == grp
        n = mask.sum()
        yt = y_true[mask]
        yp = y_pred[mask]
        ypr = y_prob[mask]

        if n < 10:
            logger.warning("Group '%s' has only %d samples — skipping.", grp, n)
            continue

        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fnr = fn / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        selection_rate = float((ypr >= threshold).mean())

        auc = np.nan
        if n >= MIN_GROUP_SIZE and len(np.unique(yt)) == 2:
            try:
                auc = roc_auc_score(yt, ypr)
            except Exception:
                pass

        records.append({
            group_name: grp,
            "n_samples": int(n),
            "default_rate": round(float(yt.mean()), 4),
            "recall (TPR)": round(recall, 4) if not np.isnan(recall) else None,
            "FNR": round(fnr, 4) if not np.isnan(fnr) else None,
            "FPR": round(fpr, 4) if not np.isnan(fpr) else None,
            "selection_rate": round(selection_rate, 4),
            "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
        })

    df = pd.DataFrame(records)
    logger.info("Fairness metrics computed for attribute: %s\n%s", group_name, df.to_string())
    return df


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def create_age_groups(age_series: pd.Series) -> pd.Series:
    """Bin continuous age values into categorical brackets."""
    return pd.cut(
        age_series,
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
    ).astype(str)


def create_dependent_groups(dep_series: pd.Series) -> pd.Series:
    """Bin NumberOfDependents into 'No Dependents' / 'Has Dependents'."""
    dep_filled = dep_series.fillna(0)
    return pd.cut(
        dep_filled,
        bins=DEPENDENTS_BINS,
        labels=DEPENDENTS_LABELS,
        right=False,
    ).astype(str)


# ---------------------------------------------------------------------------
# Full fairness report
# ---------------------------------------------------------------------------

def run_fairness_analysis(
    X_test_raw: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.30,
    save_dir=FIGURES_DIR,
) -> Dict[str, pd.DataFrame]:
    """Run fairness analysis for age and dependents groups.

    Parameters
    ----------
    X_test_raw : pd.DataFrame
        Raw (un-engineered, un-scaled) test features so we can access
        'age' and 'NumberOfDependents' for grouping without using them
        as model inputs.
    y_true, y_pred, y_prob : arrays
        Predictions from the best model on the test set.
    threshold : float
        Risk threshold for selection rate computation.
    save_dir : Path

    Returns
    -------
    dict : attribute_name → pd.DataFrame of per-group metrics.
    """
    results = {}

    # --- Age groups ---
    if "age" in X_test_raw.columns:
        age_groups = create_age_groups(X_test_raw["age"])
        df_age = compute_group_metrics(
            y_true, y_pred, y_prob,
            groups=age_groups,
            group_name="Age Group",
            threshold=threshold,
        )
        results["age"] = df_age
        _plot_group_metrics(df_age, "Age Group", save_dir)
    else:
        logger.warning("'age' column not found in X_test_raw — skipping age fairness.")

    # --- Dependents groups ---
    if "NumberOfDependents" in X_test_raw.columns:
        dep_groups = create_dependent_groups(X_test_raw["NumberOfDependents"])
        df_dep = compute_group_metrics(
            y_true, y_pred, y_prob,
            groups=dep_groups,
            group_name="Dependents Group",
            threshold=threshold,
        )
        results["dependents"] = df_dep
        _plot_group_metrics(df_dep, "Dependents Group", save_dir)
    else:
        logger.warning("'NumberOfDependents' not found — skipping dependents fairness.")

    return results


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_group_metrics(df: pd.DataFrame, group_col: str, save_dir) -> None:
    """Bar chart of key fairness metrics across groups."""
    save_dir = __import__("pathlib").Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = ["recall (TPR)", "FNR", "FPR", "selection_rate"]
    available = [c for c in metric_cols if c in df.columns]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        sns.barplot(data=df, x=group_col, y=metric, ax=ax, palette="muted")
        ax.set_title(metric, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle(f"Fairness Metrics by {group_col}", fontsize=13)
    plt.tight_layout()

    slug = group_col.lower().replace(" ", "_")
    fname = save_dir / f"fairness_{slug}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Fairness plot saved: %s", fname)
