"""
CreditWise — Preprocessing Pipeline
=====================================
Builds a scikit-learn Pipeline that transforms raw Give Me Some Credit
features into a clean numerical matrix suitable for all four ML models.

Design principles
-----------------
* The Pipeline is fit ONLY on training data and applied identically to
  test/inference data — preventing data leakage.
* Median imputation is used for numerical variables with missing values.
* Standard scaling is applied selectively: it is included in the pipeline
  so tree-based models can simply ignore it, but Logistic Regression needs
  it. A flag `scale_features` controls this.
* No information from the test set or the target column is used here.

Usage
-----
    from src.preprocessing import build_preprocessing_pipeline, split_data
    from src.data_loader import load_raw_data

    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = build_preprocessing_pipeline(scale_features=True)
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed  = pipeline.transform(X_test)
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    EXPECTED_FEATURE_COLUMNS,
    RANDOM_SEED,
    STRATIFY_SPLIT,
    TARGET_COLUMN,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the raw DataFrame into stratified train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame returned by data_loader.load_raw_data().
    test_size : float
        Fraction of the data to hold out as the test set (default 0.20).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrames / Series
        Features and target for train and test splits.

    Notes
    -----
    Stratification preserves the class distribution in both splits,
    which is especially important for imbalanced credit-risk datasets.
    The test set is NOT used for any model selection or tuning decisions.
    """
    X = df[EXPECTED_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    stratify = y if STRATIFY_SPLIT else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    logger.info(
        "Train/test split: %d train rows | %d test rows  "
        "(default rate — train: %.2f%%  test: %.2f%%)",
        len(X_train), len(X_test),
        y_train.mean() * 100, y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Preprocessing pipeline builder
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(scale_features: bool = True) -> Pipeline:
    """Create a scikit-learn Pipeline for numerical feature preprocessing.

    The pipeline performs two steps:
      1. Median imputation  — handles missing values in MonthlyIncome and
         NumberOfDependents without leaking test-set statistics.
      2. Standard scaling   — required by Logistic Regression; harmless for
         tree-based models. Disable with `scale_features=False` if you want
         to train tree models without scaling for speed.

    Parameters
    ----------
    scale_features : bool
        Whether to apply StandardScaler as the final step.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline ready to call .fit_transform() on training data.
    """
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_features:
        steps.append(("scaler", StandardScaler()))

    pipeline = Pipeline(steps)
    logger.debug(
        "Built preprocessing pipeline: %s",
        " → ".join(s[0] for s in steps),
    )
    return pipeline


# ---------------------------------------------------------------------------
# Convenience: fit and return transformed arrays with feature names
# ---------------------------------------------------------------------------

def fit_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, Pipeline]:
    """Fit *pipeline* on training data and transform both splits.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted preprocessing pipeline from build_preprocessing_pipeline().
    X_train, X_test : pd.DataFrame
        Feature DataFrames for train and test respectively.

    Returns
    -------
    X_train_tf, X_test_tf : np.ndarray
        Transformed arrays.
    pipeline : Pipeline
        The fitted pipeline (to be saved with joblib for inference).
    """
    X_train_tf = pipeline.fit_transform(X_train)
    X_test_tf = pipeline.transform(X_test)

    logger.info(
        "Preprocessing complete: X_train shape %s | X_test shape %s",
        X_train_tf.shape, X_test_tf.shape,
    )
    return X_train_tf, X_test_tf, pipeline


# ---------------------------------------------------------------------------
# Feature-name helper (for SHAP and reporting)
# ---------------------------------------------------------------------------

def get_feature_names(include_engineered: bool = False) -> list:
    """Return the ordered list of feature names after preprocessing.

    The order must match the columns produced by the preprocessing
    pipeline. This list is saved to models/feature_list.json so that
    inference time can reconstruct the correct column order.

    Parameters
    ----------
    include_engineered : bool
        If True, appends engineered feature names after the base set.
        (Populated by feature_engineering.py after engineering is done.)

    Returns
    -------
    list of str
    """
    names = list(EXPECTED_FEATURE_COLUMNS)
    # Engineered features are appended by feature_engineering.get_all_features()
    # at pipeline-build time — this stub keeps the interface consistent.
    return names
