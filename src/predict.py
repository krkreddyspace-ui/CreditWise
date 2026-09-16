"""
CreditWise — Inference / Prediction Module
===========================================
Loads the saved preprocessing pipeline and best model from disk, then
produces predictions and risk categorisation for individual applicants.

This module is used by the Streamlit dashboard and the What-If simulator.
It does NOT retrain the model — it only loads and applies saved artefacts.

Important disclaimer (replicated in the UI)
-------------------------------------------
The predicted probability is a model estimate based on historical data.
It does NOT guarantee actual repayment behaviour. This output is intended
as a decision-support tool and NOT as an autonomous lending decision.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config import (
    APP_DISCLAIMER,
    BEST_MODEL_FILE,
    CALIBRATED_MODEL_FILE,
    DECISION_SUPPORT_LABELS,
    FEATURE_LIST_FILE,
    MODEL_METADATA_FILE,
    PREPROCESSING_PIPELINE_FILE,
    RISK_LABELS,
    RISK_THRESHOLDS,
)
from src.feature_engineering import engineer_features

logger = logging.getLogger(__name__)

# Module-level cache so artefacts are loaded only once per session
_artefact_cache: Dict = {}


# ---------------------------------------------------------------------------
# Artefact loading
# ---------------------------------------------------------------------------

def load_artefacts(use_calibrated: bool = False) -> Dict:
    """Load the preprocessing pipeline, model, and feature list from disk.

    Results are cached after the first call.

    Parameters
    ----------
    use_calibrated : bool
        If True, load the calibrated model (if it exists) instead of the
        raw best model.

    Returns
    -------
    dict with keys: 'pipeline', 'model', 'feature_names', 'metadata'

    Raises
    ------
    FileNotFoundError
        If required artefact files are missing (model not yet trained).
    """
    cache_key = "calibrated" if use_calibrated else "raw"
    if cache_key in _artefact_cache:
        return _artefact_cache[cache_key]

    _check_file(PREPROCESSING_PIPELINE_FILE)
    _check_file(FEATURE_LIST_FILE)

    model_file = CALIBRATED_MODEL_FILE if use_calibrated and CALIBRATED_MODEL_FILE.exists() \
        else BEST_MODEL_FILE
    _check_file(model_file)

    pipeline = joblib.load(PREPROCESSING_PIPELINE_FILE)
    model = joblib.load(model_file)

    with open(FEATURE_LIST_FILE) as f:
        feature_names = json.load(f)

    metadata = {}
    if MODEL_METADATA_FILE.exists():
        with open(MODEL_METADATA_FILE) as f:
            metadata = json.load(f)

    artefacts = {
        "pipeline": pipeline,
        "model": model,
        "feature_names": feature_names,
        "metadata": metadata,
    }
    _artefact_cache[cache_key] = artefacts
    logger.info(
        "Artefacts loaded — model: %s | features: %d",
        metadata.get("best_model_name", "unknown"),
        len(feature_names),
    )
    return artefacts


def _check_file(path: Path) -> None:
    """Raise FileNotFoundError with a helpful message if path missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"\n\n  Required artefact not found: {path}\n\n"
            "  You must train the model first:\n"
            "    python -m src.train\n"
        )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_single(
    applicant_data: Dict,
    pipeline: Any = None,
    model: Any = None,
    feature_names: Any = None,
    risk_thresholds: Any = None,
    use_calibrated: bool = True,
) -> Dict:
    """Predict credit risk for a single applicant.

    Parameters
    ----------
    applicant_data : dict
        Keys are original Give Me Some Credit feature names.
    pipeline : sklearn.pipeline.Pipeline, optional
        Preprocessing pipeline. If None, loaded from disk.
    model : fitted model estimator, optional
        Trained model. If None, loaded from disk.
    feature_names : list of str, optional
        Feature names. If None, loaded from disk.
    risk_thresholds : dict, optional
        Risk threshold boundaries.
    use_calibrated : bool
        Use calibrated model if loading from disk.

    Returns
    -------
    dict with keys: default_probability, probability, risk_category, risk_label, decision_support
    """
    if isinstance(pipeline, bool):
        use_calibrated = pipeline
        pipeline = None

    if pipeline is None or model is None or feature_names is None:
        artefacts = load_artefacts(use_calibrated=use_calibrated)
        pipeline = artefacts["pipeline"]
        model = artefacts["model"]
        feature_names = artefacts["feature_names"]
        metadata = artefacts.get("metadata", {})
    else:
        metadata = {}

    if risk_thresholds is None:
        risk_thresholds = RISK_THRESHOLDS

    # Build a single-row DataFrame from the input dict
    input_df = pd.DataFrame([applicant_data])

    # Apply feature engineering
    input_df = engineer_features(input_df)

    # Select and order features exactly as during training
    try:
        X = input_df[feature_names]
    except KeyError as e:
        raise ValueError(
            f"Missing required feature column after feature engineering: {e}"
        ) from e

    # Transform features through pipeline
    X_trans = pipeline.transform(X)

    # Predict default probability (positive class)
    prob = float(model.predict_proba(X_trans)[0, 1])
    prob = max(0.0, min(1.0, prob))  # clip to [0, 1]

    # Assign risk category & decision support
    category = probability_to_risk_category(prob)
    label = RISK_LABELS.get(category, f"{category.capitalize()} Risk")
    decision_support = DECISION_SUPPORT_LABELS.get(category, "")

    return {
        "default_probability": prob,
        "probability": prob,
        "risk_category": category,
        "risk_label": label,
        "raw_category": category,
        "decision_support": decision_support,
        "model_name": metadata.get("best_model_name", type(model).__name__),
        "disclaimer": APP_DISCLAIMER,
    }


def predict_batch(
    df: pd.DataFrame,
    use_calibrated: bool = False,
) -> pd.DataFrame:
    """Predict credit risk for a batch of applicants.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the original Give Me Some Credit feature columns.
    use_calibrated : bool
        Use the calibrated model if available.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with 'default_probability' and 'risk_category'
        columns appended.
    """
    artefacts = load_artefacts(use_calibrated=use_calibrated)
    pipeline = artefacts["pipeline"]
    model = artefacts["model"]
    feature_names = artefacts["feature_names"]

    df_eng = engineer_features(df.copy())
    X = df_eng[feature_names]
    X_tf = pipeline.transform(X)

    probs = model.predict_proba(X_tf)[:, 1]
    probs = np.clip(probs, 0.0, 1.0)

    result_df = df.copy()
    result_df["default_probability"] = probs
    result_df["risk_category"] = [probability_to_risk_category(p) for p in probs]
    result_df["risk_label"] = result_df["risk_category"].map(RISK_LABELS)
    return result_df


# ---------------------------------------------------------------------------
# Risk categorisation
# ---------------------------------------------------------------------------

def probability_to_risk_category(probability: float) -> str:
    """Convert a scalar probability to a risk category string.

    Parameters
    ----------
    probability : float
        Predicted default probability in [0, 1].

    Returns
    -------
    str : 'low' | 'medium' | 'high'

    Raises
    ------
    ValueError
        If probability is outside [0, 1].
    """
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"Probability must be in [0, 1], got {probability}")

    if probability < RISK_THRESHOLDS["low_max"]:
        return "low"
    elif probability < RISK_THRESHOLDS["medium_max"]:
        return "medium"
    else:
        return "high"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_applicant_input(data: Dict) -> Dict:
    """Validate and sanitize raw user input from the Streamlit form.

    Parameters
    ----------
    data : dict
        Raw input values from the UI form (may include strings/None).

    Returns
    -------
    dict
        Cleaned and type-coerced values.

    Raises
    ------
    ValueError
        Describes all validation failures so the UI can display them.
    """
    errors = []

    def _require_non_negative(key, label):
        val = data.get(key)
        try:
            val = float(val)
            if val < 0:
                errors.append(f"{label} must be ≥ 0 (got {val}).")
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
        return val

    def _require_range(key, label, lo, hi):
        val = data.get(key)
        try:
            val = float(val)
            if not (lo <= val <= hi):
                errors.append(f"{label} must be between {lo} and {hi} (got {val}).")
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
        return val

    clean = {
        "RevolvingUtilizationOfUnsecuredLines": _require_non_negative(
            "RevolvingUtilizationOfUnsecuredLines", "Revolving Utilization"
        ),
        "age": _require_range("age", "Age", 18, 110),
        "NumberOfTime30-59DaysPastDueNotWorse": _require_non_negative(
            "NumberOfTime30-59DaysPastDueNotWorse", "Times 30-59 Days Past Due"
        ),
        "DebtRatio": _require_non_negative("DebtRatio", "Debt Ratio"),
        "MonthlyIncome": data.get("MonthlyIncome"),   # None → median-imputed
        "NumberOfOpenCreditLinesAndLoans": _require_non_negative(
            "NumberOfOpenCreditLinesAndLoans", "Open Credit Lines"
        ),
        "NumberOfTimes90DaysLate": _require_non_negative(
            "NumberOfTimes90DaysLate", "Times 90+ Days Late"
        ),
        "NumberRealEstateLoansOrLines": _require_non_negative(
            "NumberRealEstateLoansOrLines", "Real Estate Loans/Lines"
        ),
        "NumberOfTime60-89DaysPastDueNotWorse": _require_non_negative(
            "NumberOfTime60-89DaysPastDueNotWorse", "Times 60-89 Days Past Due"
        ),
        "NumberOfDependents": data.get("NumberOfDependents"),  # None → median-imputed
    }

    # Monthly income: if provided, must be non-negative
    if clean["MonthlyIncome"] is not None:
        try:
            val = float(clean["MonthlyIncome"])
            if val < 0:
                errors.append("Monthly Income must be ≥ 0.")
            clean["MonthlyIncome"] = val
        except (TypeError, ValueError):
            errors.append("Monthly Income must be a number (or left blank).")

    if errors:
        raise ValueError("\n".join(f"• {e}" for e in errors))

    return clean
