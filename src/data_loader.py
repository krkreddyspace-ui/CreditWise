"""
CreditWise — Data Loader
========================
Responsible for loading the raw Give Me Some Credit dataset, validating
its schema, and returning a clean DataFrame ready for EDA/preprocessing.

This module does NOT perform any feature transformation — that belongs
in preprocessing.py. It only handles IO and structural validation.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    EXPECTED_FEATURE_COLUMNS,
    RAW_DATA_FILE,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_data(filepath: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load the Give Me Some Credit CSV and perform basic structural checks.

    Parameters
    ----------
    filepath : Path
        Path to cs-training.csv (or a compatible dataset).

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with the unnamed index column dropped and columns
        validated against the expected schema.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at *filepath*.
    ValueError
        If required columns are missing from the CSV.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"\n\n  Dataset not found at: {filepath}\n\n"
            "  Action required:\n"
            "  1. Download 'cs-training.csv' from:\n"
            "     https://www.kaggle.com/c/GiveMeSomeCredit/data\n"
            "  2. Place it at the path shown above.\n"
        )

    logger.info("Loading dataset from %s …", filepath)
    df = pd.read_csv(filepath)

    # The raw file has an unnamed leading index column — drop it
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        logger.debug("Dropped unnamed index column(s): %s", unnamed_cols)

    _validate_schema(df)

    logger.info(
        "Dataset loaded: %d rows × %d columns  |  target='%s'",
        len(df), df.shape[1], TARGET_COLUMN,
    )
    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return a dictionary of basic dataset statistics for reporting.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or lightly cleaned DataFrame.

    Returns
    -------
    dict
        Summary statistics including shape, dtypes, missing-value counts,
        duplicate count, and target distribution.
    """
    target_counts = df[TARGET_COLUMN].value_counts().to_dict() if TARGET_COLUMN in df.columns else {}
    total = len(df)

    missing = df.isnull().sum()
    missing = missing[missing > 0].to_dict()

    return {
        "n_rows": total,
        "n_cols": df.shape[1],
        "n_features": df.shape[1] - 1,  # exclude target
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_counts": missing,
        "missing_pct": {k: round(v / total * 100, 2) for k, v in missing.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": target_counts,
        "target_imbalance_ratio": (
            round(target_counts.get(0, 0) / target_counts.get(1, 1), 2)
            if target_counts else None
        ),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply light cleaning based on EDA findings before preprocessing.

    Steps performed
    ---------------
    1. Drop 609 exact duplicate rows (found in EDA).
    2. Replace delinquency-column values of 96 and 98 with NaN — these
       appear to be missing-value codes, not real delinquency counts.
       They will be handled by the median imputer in the pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from load_raw_data().

    Returns
    -------
    pd.DataFrame
        Lightly cleaned DataFrame, ready for preprocessing/splitting.

    Notes
    -----
    * No rows are dropped for outliers (e.g., extreme DebtRatio or
      RevolvingUtilization) — those are handled by log-transforms.
    * The single row with age=0 is NOT dropped here; the input validation
      module will reject age=0 at inference time.
    * This function is applied BEFORE the train/test split so that all
      data splits benefit from the cleaning.
    """
    original_len = len(df)

    # 1. Drop duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    n_dropped = original_len - len(df)
    if n_dropped:
        logger.info("Dropped %d duplicate rows (%d → %d).", n_dropped, original_len, len(df))

    # 2. Replace suspicious delinquency codes with NaN
    delinquency_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    for col in delinquency_cols:
        n_codes = df[col].isin([96, 98]).sum()
        if n_codes:
            df[col] = df[col].replace({96: np.nan, 98: np.nan})
            logger.info("Replaced %d suspicious values (96/98) in '%s' with NaN.", n_codes, col)

    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Assert that all expected columns are present.

    Raises
    ------
    ValueError
        Lists any missing columns so the user knows exactly what is wrong.
    """
    expected = set(EXPECTED_FEATURE_COLUMNS + [TARGET_COLUMN])
    actual = set(df.columns)
    missing = expected - actual

    if missing:
        raise ValueError(
            f"Schema validation failed. Missing columns: {sorted(missing)}\n"
            f"Columns found in CSV: {sorted(actual)}\n"
            "Check that you are using the correct dataset (cs-training.csv)."
        )

    # Warn about any extra unexpected columns (not an error — future-proofing)
    extra = actual - expected
    if extra:
        logger.warning(
            "Unexpected extra columns found (will be ignored during "
            "preprocessing): %s", sorted(extra)
        )
