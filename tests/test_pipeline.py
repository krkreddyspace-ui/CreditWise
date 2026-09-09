"""
CreditWise — Unit Tests
========================
Basic test suite for the CreditWise pipeline.

Run with:
    pytest tests/ -v

These tests verify:
  - probability_to_risk_category produces correct outputs
  - validate_applicant_input catches invalid values
  - feature engineering runs without error on synthetic data
  - preprocessing pipeline produces correct output shape
  - model artefacts can be loaded after training (skipped if not trained yet)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EXPECTED_FEATURE_COLUMNS, RISK_THRESHOLDS
from src.feature_engineering import engineer_features, get_all_feature_names
from src.predict import probability_to_risk_category, validate_applicant_input
from src.preprocessing import build_preprocessing_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_applicant():
    """Valid applicant data dictionary."""
    return {
        "RevolvingUtilizationOfUnsecuredLines": 0.45,
        "age": 42,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.30,
        "MonthlyIncome": 5000.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2,
    }


@pytest.fixture
def sample_dataframe():
    """Small synthetic DataFrame mimicking the Give Me Some Credit schema."""
    n = 50
    rng = np.random.default_rng(42)
    data = {
        "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 1, n),
        "age": rng.integers(25, 75, n),
        "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 5, n),
        "DebtRatio": rng.uniform(0, 2, n),
        "MonthlyIncome": rng.uniform(1000, 10000, n),
        "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, n),
        "NumberOfTimes90DaysLate": rng.integers(0, 3, n),
        "NumberRealEstateLoansOrLines": rng.integers(0, 4, n),
        "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 3, n),
        "NumberOfDependents": rng.integers(0, 5, n),
        "SeriousDlqin2yrs": rng.integers(0, 2, n),
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Risk categorisation tests
# ---------------------------------------------------------------------------

class TestRiskCategorisation:
    def test_low_risk_boundary(self):
        assert probability_to_risk_category(0.0) == "low"
        assert probability_to_risk_category(0.10) == "low"
        assert probability_to_risk_category(RISK_THRESHOLDS["low_max"] - 0.001) == "low"

    def test_medium_risk_boundary(self):
        assert probability_to_risk_category(RISK_THRESHOLDS["low_max"]) == "medium"
        assert probability_to_risk_category(0.45) == "medium"
        assert probability_to_risk_category(RISK_THRESHOLDS["medium_max"] - 0.001) == "medium"

    def test_high_risk_boundary(self):
        assert probability_to_risk_category(RISK_THRESHOLDS["medium_max"]) == "high"
        assert probability_to_risk_category(0.80) == "high"
        assert probability_to_risk_category(1.0) == "high"

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError):
            probability_to_risk_category(-0.01)
        with pytest.raises(ValueError):
            probability_to_risk_category(1.01)

    def test_returns_string(self):
        result = probability_to_risk_category(0.5)
        assert isinstance(result, str)
        assert result in ("low", "medium", "high")


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_valid_input_passes(self, sample_applicant):
        clean = validate_applicant_input(sample_applicant)
        assert clean is not None
        assert isinstance(clean["age"], float)

    def test_negative_age_raises(self, sample_applicant):
        sample_applicant["age"] = -5
        with pytest.raises(ValueError, match="Age"):
            validate_applicant_input(sample_applicant)

    def test_age_too_high_raises(self, sample_applicant):
        sample_applicant["age"] = 150
        with pytest.raises(ValueError, match="Age"):
            validate_applicant_input(sample_applicant)

    def test_negative_income_raises(self, sample_applicant):
        sample_applicant["MonthlyIncome"] = -100
        with pytest.raises(ValueError, match="Monthly Income"):
            validate_applicant_input(sample_applicant)

    def test_negative_debt_ratio_raises(self, sample_applicant):
        sample_applicant["DebtRatio"] = -1.0
        with pytest.raises(ValueError, match="Debt Ratio"):
            validate_applicant_input(sample_applicant)

    def test_none_income_allowed(self, sample_applicant):
        """None monthly income should pass — it will be median-imputed."""
        sample_applicant["MonthlyIncome"] = None
        clean = validate_applicant_input(sample_applicant)
        assert clean["MonthlyIncome"] is None

    def test_none_dependents_allowed(self, sample_applicant):
        sample_applicant["NumberOfDependents"] = None
        clean = validate_applicant_input(sample_applicant)
        assert clean["NumberOfDependents"] is None


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    def test_engineer_returns_dataframe(self, sample_dataframe):
        result = engineer_features(sample_dataframe)
        assert isinstance(result, pd.DataFrame)

    def test_original_columns_preserved(self, sample_dataframe):
        result = engineer_features(sample_dataframe)
        for col in EXPECTED_FEATURE_COLUMNS:
            assert col in result.columns, f"Original column '{col}' missing after engineering"

    def test_engineered_columns_added(self, sample_dataframe):
        result = engineer_features(sample_dataframe)
        expected_new = [
            "log_revolving_utilization",
            "log_debt_ratio",
            "log_monthly_income",
            "total_past_due",
            "has_past_due",
            "income_per_dependent",
            "credit_line_density",
        ]
        for col in expected_new:
            assert col in result.columns, f"Engineered column '{col}' not created"

    def test_no_negative_log_values(self, sample_dataframe):
        result = engineer_features(sample_dataframe)
        for col in ["log_revolving_utilization", "log_debt_ratio", "log_monthly_income"]:
            vals = result[col].dropna()
            assert (vals >= 0).all(), f"{col} contains negative values (log of negative input?)"

    def test_has_past_due_is_binary(self, sample_dataframe):
        result = engineer_features(sample_dataframe)
        unique_vals = set(result["has_past_due"].dropna().unique())
        assert unique_vals.issubset({0, 1})

    def test_feature_count(self):
        all_features = get_all_feature_names()
        assert len(all_features) == len(EXPECTED_FEATURE_COLUMNS) + 7

    def test_row_count_unchanged(self, sample_dataframe):
        result = engineer_features(sample_dataframe)
        assert len(result) == len(sample_dataframe)


# ---------------------------------------------------------------------------
# Preprocessing pipeline tests
# ---------------------------------------------------------------------------

class TestPreprocessingPipeline:
    def test_pipeline_output_shape(self, sample_dataframe):
        from src.feature_engineering import get_all_feature_names
        df_eng = engineer_features(sample_dataframe)
        feature_names = get_all_feature_names()
        X = df_eng[feature_names]

        pipeline = build_preprocessing_pipeline(scale_features=True)
        X_tf = pipeline.fit_transform(X)

        assert X_tf.shape == (len(sample_dataframe), len(feature_names))

    def test_no_nan_after_imputation(self, sample_dataframe):
        """After fit_transform, there should be no NaN values."""
        # Introduce NaN manually
        sample_dataframe.loc[0, "MonthlyIncome"] = np.nan
        sample_dataframe.loc[1, "NumberOfDependents"] = np.nan

        from src.feature_engineering import get_all_feature_names
        df_eng = engineer_features(sample_dataframe)
        feature_names = get_all_feature_names()
        X = df_eng[feature_names]

        pipeline = build_preprocessing_pipeline(scale_features=True)
        X_tf = pipeline.fit_transform(X)

        assert not np.isnan(X_tf).any(), "NaN values remain after imputation"

    def test_pipeline_without_scaling(self, sample_dataframe):
        from src.feature_engineering import get_all_feature_names
        df_eng = engineer_features(sample_dataframe)
        feature_names = get_all_feature_names()
        X = df_eng[feature_names]

        pipeline = build_preprocessing_pipeline(scale_features=False)
        X_tf = pipeline.fit_transform(X)
        # Step names should only include imputer
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0][0] == "imputer"


# ---------------------------------------------------------------------------
# Artefact loading test (skipped if model not trained)
# ---------------------------------------------------------------------------

class TestArtefactLoading:
    def test_load_artefacts_when_available(self):
        from src.config import BEST_MODEL_FILE, PREPROCESSING_PIPELINE_FILE
        if not BEST_MODEL_FILE.exists() or not PREPROCESSING_PIPELINE_FILE.exists():
            pytest.skip("Model artefacts not yet trained — run src/train.py first.")

        from src.predict import load_artefacts
        artefacts = load_artefacts()
        assert "pipeline" in artefacts
        assert "model" in artefacts
        assert "feature_names" in artefacts
        assert len(artefacts["feature_names"]) > 0

    def test_predict_single_probability_in_range(self, sample_applicant):
        from src.config import BEST_MODEL_FILE, PREPROCESSING_PIPELINE_FILE
        if not BEST_MODEL_FILE.exists() or not PREPROCESSING_PIPELINE_FILE.exists():
            pytest.skip("Model artefacts not yet trained — run src/train.py first.")

        from src.predict import predict_single
        result = predict_single(sample_applicant)
        assert 0.0 <= result["default_probability"] <= 1.0
        assert result["risk_category"] in ("low", "medium", "high")
