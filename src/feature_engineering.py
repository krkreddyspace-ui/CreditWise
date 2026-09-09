"""
CreditWise — Feature Engineering
==================================
Creates financially justified derived features from the Give Me Some
Credit base columns.

Design rules
------------
* Every feature must have a documented financial or statistical justification.
* No feature is created just to inflate feature count.
* Engineered features are computed from the RAW columns so that the same
  transformation is applied identically at inference time.
* All transformations are implemented as stateless functions — no fitted
  state is required because we use only ratios, log-transforms, and
  simple arithmetic (not statistics computed from the training set).

Features currently implemented
-------------------------------
F1  log_revolving_utilization
    Log-transform of RevolvingUtilizationOfUnsecuredLines.
    Justification: The raw values are right-skewed with extreme outliers
    (values >> 1 are possible due to data quality). A log1p transform
    compresses the tail and makes the distribution more symmetric, helping
    Logistic Regression and reducing the impact of outliers on distance-
    based models.

F2  log_debt_ratio
    Log-transform of DebtRatio.
    Justification: Same as above — DebtRatio has a heavily skewed
    distribution in the raw data.

F3  log_monthly_income
    Log-transform of MonthlyIncome (applied after imputation of NaN).
    Justification: Monthly income is strongly right-skewed. The log scale
    aligns with the economic literature on income distributions.

F4  total_past_due
    Sum of 30-59, 60-89 and 90+ days past-due counts.
    Justification: Delinquency severity is cumulative; combining all three
    buckets gives a single summary measure of payment history.

F5  has_past_due
    Binary indicator: 1 if total_past_due > 0, else 0.
    Justification: Presence vs absence of any delinquency is a strong
    discriminator regardless of count magnitude.

F6  income_per_dependent
    MonthlyIncome / (NumberOfDependents + 1).
    Justification: Available income per household member is a meaningful
    affordability signal. Adding 1 avoids division by zero for borrowers
    with no dependents and ensures non-zero denominator.

F7  credit_line_density
    NumberOfOpenCreditLinesAndLoans / (age - 17).
    Justification: Normalises credit-line count by effective credit age
    (assuming credit starts at 18). Captures credit utilisation relative
    to credit history length.

Note: All features are computed with numpy operations that handle NaN
gracefully — NaN propagation is expected and will be handled by the
median imputer in the preprocessing pipeline.
"""

import logging

import numpy as np
import pandas as pd

from src.config import EXPECTED_FEATURE_COLUMNS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engineered feature names (in append order)
# ---------------------------------------------------------------------------
ENGINEERED_FEATURE_NAMES = [
    "log_revolving_utilization",
    "log_debt_ratio",
    "log_monthly_income",
    "total_past_due",
    "has_past_due",
    "income_per_dependent",
    "credit_line_density",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features to *df* and return the result.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the original Give Me Some Credit columns.
        May contain NaN values — they are preserved and will be handled
        by the downstream imputation step.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the original columns plus engineered features
        appended. The original columns are not modified.
    """
    df = df.copy()

    # F1: Log revolving utilization
    # np.log1p handles 0 correctly; clip negative values to 0 first
    util = df["RevolvingUtilizationOfUnsecuredLines"].clip(lower=0)
    df["log_revolving_utilization"] = np.log1p(util)

    # F2: Log debt ratio
    debt = df["DebtRatio"].clip(lower=0)
    df["log_debt_ratio"] = np.log1p(debt)

    # F3: Log monthly income (NaN safe — log1p(NaN) = NaN → imputed later)
    income = df["MonthlyIncome"].clip(lower=0)
    df["log_monthly_income"] = np.log1p(income)

    # F4: Total past-due count across all severity buckets
    df["total_past_due"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"].clip(lower=0)
        + df["NumberOfTime60-89DaysPastDueNotWorse"].clip(lower=0)
        + df["NumberOfTimes90DaysLate"].clip(lower=0)
    )

    # F5: Binary indicator for any delinquency
    df["has_past_due"] = (df["total_past_due"] > 0).astype(int)

    # F6: Income per dependent (NaN safe)
    n_dep = df["NumberOfDependents"].fillna(0).clip(lower=0)
    df["income_per_dependent"] = df["MonthlyIncome"] / (n_dep + 1)

    # F7: Credit line density (age - 17 is effective credit age)
    credit_age = (df["age"] - 17).clip(lower=1)  # clip to avoid division by zero
    df["credit_line_density"] = df["NumberOfOpenCreditLinesAndLoans"] / credit_age

    logger.debug(
        "Feature engineering complete. Shape: %s → %s",
        (len(df), len(EXPECTED_FEATURE_COLUMNS)),
        df.shape,
    )
    return df


def get_all_feature_names() -> list:
    """Return the full ordered list of features after engineering.

    This is the definitive source of feature names used by the
    preprocessing pipeline, SHAP explainer, and inference module.

    Returns
    -------
    list of str
        Base columns followed by engineered features in append order.
    """
    return list(EXPECTED_FEATURE_COLUMNS) + ENGINEERED_FEATURE_NAMES
