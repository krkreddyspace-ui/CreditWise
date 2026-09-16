"""
CreditWise — REST API Test Suite
=================================
Automated test suite verifying FastAPI REST endpoints:
- GET /health
- POST /predict (single applicant inference)
- POST /predict/batch (batch processing)
- POST /explain (local SHAP breakdown)
"""

import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_root_info():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["pipeline_ready"] is True
    assert data["features_count"] > 0


def test_predict_single_valid():
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.25,
        "age": 42,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.30,
        "MonthlyIncome": 6000.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0
    assert data["risk_category"] in ["Low Risk", "Medium Risk", "High Risk"]
    assert "probability_percentage" in data


def test_predict_single_invalid_age():
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.25,
        "age": 10, # Invalid age < 18
        "DebtRatio": 0.30,
        "MonthlyIncome": 6000.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberRealEstateLoansOrLines": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Pydantic validation error


def test_predict_batch():
    payload = {
        "applicants": [
            {
                "RevolvingUtilizationOfUnsecuredLines": 0.10,
                "age": 50,
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.20,
                "MonthlyIncome": 8500.0,
                "NumberOfOpenCreditLinesAndLoans": 10,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 2,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2
            },
            {
                "RevolvingUtilizationOfUnsecuredLines": 0.85,
                "age": 28,
                "NumberOfTime30-59DaysPastDueNotWorse": 2,
                "DebtRatio": 0.65,
                "MonthlyIncome": 3200.0,
                "NumberOfOpenCreditLinesAndLoans": 5,
                "NumberOfTimes90DaysLate": 1,
                "NumberRealEstateLoansOrLines": 0,
                "NumberOfTime60-89DaysPastDueNotWorse": 1,
                "NumberOfDependents": 0
            }
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2


def test_explain_applicant():
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.45,
        "age": 35,
        "NumberOfTime30-59DaysPastDueNotWorse": 1,
        "DebtRatio": 0.40,
        "MonthlyIncome": 4500.0,
        "NumberOfOpenCreditLinesAndLoans": 6,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "top_contributing_factors" in data
    assert len(data["top_contributing_factors"]) > 0
