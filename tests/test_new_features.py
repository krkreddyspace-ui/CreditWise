"""
CreditWise — Advanced Features Test Suite
=========================================
Unit and integration tests for:
- Conformal Prediction & Uncertainty Quantification (ICP)
- Actionable Algorithmic Recourse & Counterfactuals
- Natural Language Underwriter Executive Narrative Generator
- Automated PDF Loan Audit & Decision Dossier Generation
- FastAPI Endpoints for Conformal, Recourse, Narrative, and PDF Dossier
"""

import io
import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.conformal import calculate_conformal_interval
from src.counterfactual import generate_counterfactual_recourse
from src.narrative import generate_executive_narrative
from src.pdf_generator import generate_credit_dossier_pdf
from src.predict import load_artefacts
from src.api import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Conformal Prediction Tests
# ---------------------------------------------------------------------------

def test_conformal_interval_bounds():
    res = calculate_conformal_interval(0.25, confidence_level=0.95)
    assert 0.0 <= res["lower_bound"] <= 0.25
    assert 0.25 <= res["upper_bound"] <= 1.0
    assert res["confidence_percentage"] == 95.0
    assert "certainty_tier" in res
    assert "margin_of_error" in res


def test_conformal_interval_edge_probabilities():
    res_zero = calculate_conformal_interval(0.0)
    assert res_zero["lower_bound"] == 0.0
    assert res_zero["upper_bound"] > 0.0

    res_one = calculate_conformal_interval(1.0)
    assert res_one["upper_bound"] == 1.0
    assert res_one["lower_bound"] < 1.0


# ---------------------------------------------------------------------------
# Counterfactual Recourse Tests
# ---------------------------------------------------------------------------

def test_counterfactual_recourse_for_high_risk():
    artefacts = load_artefacts(use_calibrated=True)
    model = artefacts["model"]
    pipeline = artefacts["pipeline"]

    high_risk_input = {
        "RevolvingUtilizationOfUnsecuredLines": 0.88,
        "age": 34,
        "NumberOfTime30-59DaysPastDueNotWorse": 2,
        "DebtRatio": 0.65,
        "MonthlyIncome": 3200.0,
        "NumberOfOpenCreditLinesAndLoans": 6,
        "NumberOfTimes90DaysLate": 1,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 1,
        "NumberOfDependents": 1
    }

    res = generate_counterfactual_recourse(model, high_risk_input, pipeline, target_threshold=0.30)
    assert res["original_probability"] > 0.30
    assert res["counterfactual_probability"] < res["original_probability"]
    assert res["is_recourse_found"] is True
    assert len(res["actionable_steps"]) > 0
    # Demographic / immutable attributes must remain unchanged
    assert res["modified_profile"]["age"] == 34
    assert res["modified_profile"]["NumberOfDependents"] == 1


def test_counterfactual_already_eligible():
    artefacts = load_artefacts(use_calibrated=True)
    model = artefacts["model"]
    pipeline = artefacts["pipeline"]

    low_risk_input = {
        "RevolvingUtilizationOfUnsecuredLines": 0.05,
        "age": 55,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.15,
        "MonthlyIncome": 12000.0,
        "NumberOfOpenCreditLinesAndLoans": 10,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 2,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 0
    }

    res = generate_counterfactual_recourse(model, low_risk_input, pipeline, target_threshold=0.30)
    assert res["already_eligible"] is True
    assert res["original_probability"] <= 0.30


# ---------------------------------------------------------------------------
# Executive Narrative Tests
# ---------------------------------------------------------------------------

def test_executive_narrative_generation():
    sample_data = {
        "RevolvingUtilizationOfUnsecuredLines": 0.25,
        "age": 45,
        "MonthlyIncome": 7500.0,
        "DebtRatio": 0.30,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "NumberOfTimes90DaysLate": 0
    }
    pred_res = {"probability": 0.045, "risk_category": "Low Risk"}
    conf_res = calculate_conformal_interval(0.045)
    shap_sample = {"RevolvingUtilizationOfUnsecuredLines": -0.15, "total_past_due": -0.45, "DebtRatio": 0.05}

    narrative = generate_executive_narrative(
        sample_data, pred_res, conformal_result=conf_res, shap_contributions=shap_sample
    )

    assert "Favorable" in narrative["executive_headline"] or "Low" in narrative["executive_headline"]
    assert len(narrative["key_strengths"]) > 0
    assert len(narrative["plain_english_summary"]) > 50


# ---------------------------------------------------------------------------
# PDF Dossier Generator Tests
# ---------------------------------------------------------------------------

def test_pdf_dossier_generation():
    sample_data = {
        "RevolvingUtilizationOfUnsecuredLines": 0.30,
        "age": 42,
        "MonthlyIncome": 6500.0,
        "DebtRatio": 0.35,
        "NumberOfOpenCreditLinesAndLoans": 7,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "NumberOfTimes90DaysLate": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1
    }
    pred_res = {"probability": 0.052, "risk_category": "Low Risk"}
    conf_res = calculate_conformal_interval(0.052)
    shap_sample = {"RevolvingUtilizationOfUnsecuredLines": -0.10, "DebtRatio": 0.04}
    narrative = generate_executive_narrative(sample_data, pred_res, conformal_result=conf_res)

    pdf_bytes = generate_credit_dossier_pdf(
        applicant_data=sample_data,
        prediction_result=pred_res,
        conformal_result=conf_res,
        shap_contributions=shap_sample,
        narrative_result=narrative
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")


# ---------------------------------------------------------------------------
# New FastAPI REST Endpoints Tests
# ---------------------------------------------------------------------------

def test_api_conformal_endpoint():
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.25,
        "age": 40,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.30,
        "MonthlyIncome": 6000.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1
    }
    resp = client.post("/conformal", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "lower_bound" in data
    assert "upper_bound" in data
    assert "certainty_tier" in data


def test_api_recourse_endpoint():
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.85,
        "age": 30,
        "NumberOfTime30-59DaysPastDueNotWorse": 2,
        "DebtRatio": 0.60,
        "MonthlyIncome": 3500.0,
        "NumberOfOpenCreditLinesAndLoans": 6,
        "NumberOfTimes90DaysLate": 1,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1
    }
    resp = client.post("/recourse", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "counterfactual_probability_pct" in data
    assert "actionable_steps" in data


def test_api_narrative_endpoint():
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.20,
        "age": 48,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.25,
        "MonthlyIncome": 9000.0,
        "NumberOfOpenCreditLinesAndLoans": 10,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 2,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2
    }
    resp = client.post("/narrative", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "executive_headline" in data
    assert "plain_english_summary" in data


def test_api_pdf_dossier_endpoint():
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.25,
        "age": 40,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.30,
        "MonthlyIncome": 6000.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1
    }
    resp = client.post("/dossier/pdf", json=payload)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF")
