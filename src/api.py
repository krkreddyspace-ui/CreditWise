"""
CreditWise — Production REST API Microservice
===============================================
FastAPI microservice providing high-performance REST endpoints for
single-applicant predictions, batch risk assessments, SHAP explanations,
and system health checks.
"""

import logging
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure repo root is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.predict import (
    load_artefacts, predict_single, predict_batch,
    validate_applicant_input, probability_to_risk_category
)
from src.explainability import explain_single
from src.config import RISK_THRESHOLDS, APP_DISCLAIMER

# Initialize FastAPI App
app = FastAPI(
    title="CreditWise REST API",
    description="Explainable AI Framework for Credit Risk Assessment and Loan Decision Support.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Artefact Storage
_artefacts_cache: Dict[str, Any] = {}


def get_system_artefacts() -> Dict[str, Any]:
    """Lazy loader for system artifacts (preprocessing pipeline, model, features)."""
    if not _artefacts_cache:
        try:
            artefacts = load_artefacts(use_calibrated=True)
            _artefacts_cache.update(artefacts)
        except Exception as e:
            raise RuntimeError(f"Failed to load CreditWise model artefacts: {e}")
    return _artefacts_cache


from pydantic import BaseModel, Field, ConfigDict

class ApplicantProfile(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0.0, le=100.0, description="Credit card balance divided by credit limit")
    age: int = Field(..., ge=18, le=110, description="Applicant age in years")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(0, ge=0, le=50, alias="NumberOfTime30-59DaysPastDueNotWorse")
    DebtRatio: float = Field(..., ge=0.0, le=100.0, description="Monthly debt payment divided by gross income")
    MonthlyIncome: Optional[float] = Field(5500.0, ge=0.0, description="Monthly gross income in USD")
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0, le=100, description="Number of open credit lines")
    NumberOfTimes90DaysLate: int = Field(0, ge=0, le=50, description="Number of 90+ days past due occurrences")
    NumberRealEstateLoansOrLines: int = Field(..., ge=0, le=50, description="Number of real estate loans")
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(0, ge=0, le=50, alias="NumberOfTime60-89DaysPastDueNotWorse")
    NumberOfDependents: Optional[int] = Field(1, ge=0, le=20, description="Number of financial dependents")

    def to_raw_dict(self) -> Dict[str, Any]:
        return {
            "RevolvingUtilizationOfUnsecuredLines": self.RevolvingUtilizationOfUnsecuredLines,
            "age": self.age,
            "NumberOfTime30-59DaysPastDueNotWorse": self.NumberOfTime30_59DaysPastDueNotWorse,
            "DebtRatio": self.DebtRatio,
            "MonthlyIncome": self.MonthlyIncome,
            "NumberOfOpenCreditLinesAndLoans": self.NumberOfOpenCreditLinesAndLoans,
            "NumberOfTimes90DaysLate": self.NumberOfTimes90DaysLate,
            "NumberRealEstateLoansOrLines": self.NumberRealEstateLoansOrLines,
            "NumberOfTime60-89DaysPastDueNotWorse": self.NumberOfTime60_89DaysPastDueNotWorse,
            "NumberOfDependents": self.NumberOfDependents,
        }


class SinglePredictionResponse(BaseModel):
    probability: float = Field(..., description="Estimated probability of default (0.0 to 1.0)")
    probability_percentage: float = Field(..., description="Estimated default risk percentage (0.0% to 100.0%)")
    risk_category: str = Field(..., description="Risk category: Low Risk, Medium Risk, or High Risk")
    decision_support: str = Field(..., description="Academic decision support recommendation")
    disclaimer: str = Field(APP_DISCLAIMER)


class BatchPredictionRequest(BaseModel):
    applicants: List[ApplicantProfile]


class BatchPredictionResponse(BaseModel):
    total_processed: int
    predictions: List[SinglePredictionResponse]


class FeatureImpact(BaseModel):
    feature: str
    applicant_value: Any
    shap_value: float
    direction: str


class ExplanationResponse(BaseModel):
    probability: float
    risk_category: str
    top_contributing_factors: List[FeatureImpact]


# Routes
@app.get("/", tags=["System Information"])
def root_info():
    """Returns API overview and documentation links."""
    return {
        "title": "CreditWise Explainable AI REST API",
        "status": "online",
        "documentation": "/docs",
        "disclaimer": APP_DISCLAIMER
    }


@app.get("/health", tags=["System Information"])
def health_check():
    """System health check endpoint verifying model artifact readiness."""
    try:
        arts = get_system_artefacts()
        return {
            "status": "healthy",
            "model_name": arts.get("metadata", {}).get("best_model_name", "XGBoost (Calibrated)"),
            "features_count": len(arts.get("feature_names", [])),
            "pipeline_ready": arts.get("pipeline") is not None
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {e}"
        )


@app.post("/predict", response_model=SinglePredictionResponse, tags=["Inference"])
def predict_applicant_risk(profile: ApplicantProfile):
    """Computes default probability and risk classification for a single loan applicant."""
    try:
        raw_dict = profile.to_raw_dict()
        validated = validate_applicant_input(raw_dict)
        res = predict_single(validated, use_calibrated=True)

        prob = res["default_probability"]
        category = res["risk_label"]
        support = res["decision_support"]

        return SinglePredictionResponse(
            probability=prob,
            probability_percentage=round(prob * 100.0, 2),
            risk_category=category,
            decision_support=support,
            disclaimer=APP_DISCLAIMER
        )
    except Exception as e:
        logger.error("API /predict error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction error: {e}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_applicants(batch: BatchPredictionRequest):
    """Executes high-throughput batch risk assessment for multiple applicants."""
    if not batch.applicants:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Applicant batch list cannot be empty."
        )

    try:
        responses = []
        for applicant in batch.applicants:
            raw_dict = applicant.to_raw_dict()
            validated = validate_applicant_input(raw_dict)
            res = predict_single(validated, use_calibrated=True)
            prob = res["default_probability"]
            category = res["risk_label"]
            support = res["decision_support"]

            responses.append(
                SinglePredictionResponse(
                    probability=prob,
                    probability_percentage=round(prob * 100.0, 2),
                    risk_category=category,
                    decision_support=support,
                    disclaimer=APP_DISCLAIMER
                )
            )

        return BatchPredictionResponse(
            total_processed=len(responses),
            predictions=responses
        )
    except Exception as e:
        logger.error("API /predict/batch error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch processing error: {e}"
        )


@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
def explain_applicant_risk(profile: ApplicantProfile):
    """Generates local SHAP feature contribution breakdown explaining the applicant risk score."""
    try:
        arts = get_system_artefacts()
        raw_dict = profile.to_raw_dict()
        validated = validate_applicant_input(raw_dict)
        pred_res = predict_single(validated, use_calibrated=True)

        shap_exp = explain_single(
            arts["model"],
            validated,
            arts["pipeline"],
            arts["feature_names"]
        )

        shap_vals = shap_exp.get("shap_values", {})
        sorted_items = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)

        impacts = []
        for feat, val in sorted_items[:8]:
            direction = "Increases Risk" if val > 0 else "Decreases Risk"
            user_val = raw_dict.get(feat, "N/A")
            impacts.append(
                FeatureImpact(
                    feature=feat,
                    applicant_value=user_val,
                    shap_value=round(float(val), 4),
                    direction=direction
                )
            )

        return ExplanationResponse(
            probability=pred_res["default_probability"],
            risk_category=pred_res["risk_label"],
            top_contributing_factors=impacts
        )
    except Exception as e:
        logger.error("API /explain error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Explainability error: {e}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
