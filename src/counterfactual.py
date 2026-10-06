"""
CreditWise — Actionable Counterfactual Explanations & Algorithmic Recourse
========================================================================
Generates actionable, minimal-effort recourse modifications for loan applicants
who fall into Medium or High Risk categories to reach Low Risk (<30% default probability).

Distinguishes between actionable/mutable financial variables (credit card utilization,
debt ratio, resolving active late accounts) and immutable demographics (age, dependents).
"""

from typing import Dict, Any, List, Optional, Tuple
import copy
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def generate_counterfactual_recourse(
    model: Any,
    raw_input: Dict[str, Any],
    pipeline: Pipeline,
    target_threshold: float = 0.28,
    max_iterations: int = 25
) -> Dict[str, Any]:
    """Generates an actionable counterfactual profile and step-by-step path to approval.

    Parameters
    ----------
    model : trained ML model (e.g. CalibratedClassifierCV or XGBClassifier)
    raw_input : dict of applicant financial features
    pipeline : scikit-learn preprocessing pipeline
    target_threshold : float, default=0.28 (Target risk ceiling for low risk)
    max_iterations : int, default=25

    Returns
    -------
    dict
        Counterfactual result containing:
        - original_probability
        - target_probability
        - counterfactual_probability
        - is_recourse_found
        - actionable_steps (list of human-readable action items)
        - modified_profile (dict of modified applicant features)
        - delta_table (list of dicts comparing original vs modified values)
    """
    from src.predict import predict_single

    # 1. Base prediction
    base_res = predict_single(raw_input, pipeline=pipeline, model=model)
    base_prob = base_res["probability"]

    # If applicant is already Low Risk, no recourse is needed
    if base_prob <= target_threshold:
        return {
            "original_probability": base_prob,
            "original_probability_pct": round(base_prob * 100, 2),
            "target_probability": target_threshold,
            "counterfactual_probability": base_prob,
            "counterfactual_probability_pct": round(base_prob * 100, 2),
            "risk_reduction_pp": 0.0,
            "is_recourse_found": True,
            "already_eligible": True,
            "actionable_steps": ["Applicant profile already meets low-risk underwriting criteria."],
            "modified_profile": dict(raw_input),
            "delta_table": []
        }

    # 2. Iterative actionable optimization
    current_profile = copy.deepcopy(raw_input)
    steps = []
    
    # Priority Action 1: Resolve all late/past due incidents to 0
    late_30 = int(current_profile.get("NumberOfTime30-59DaysPastDueNotWorse", 0) or 0)
    late_60 = int(current_profile.get("NumberOfTime60-89DaysPastDueNotWorse", 0) or 0)
    late_90 = int(current_profile.get("NumberOfTimes90DaysLate", 0) or 0)

    if late_30 > 0 or late_60 > 0 or late_90 > 0:
        current_profile["NumberOfTime30-59DaysPastDueNotWorse"] = 0
        current_profile["NumberOfTime60-89DaysPastDueNotWorse"] = 0
        current_profile["NumberOfTimes90DaysLate"] = 0
        total_late = late_30 + late_60 + late_90
        steps.append({
            "feature": "Delinquency History",
            "action": f"Cure and resolve {total_late} active delinquent account incident(s) to 0 late records.",
            "impact": "High Positive Impact"
        })

    # Test probability after resolving late incidents
    eval_res = predict_single(current_profile, pipeline=pipeline, model=model)
    current_prob = eval_res["probability"]

    # Priority Action 2: Lower Revolving Utilization
    current_util = float(current_profile.get("RevolvingUtilizationOfUnsecuredLines", 0.5) or 0.5)
    if current_prob > target_threshold and current_util > 0.20:
        # Step down utilization gradually
        target_util_levels = [0.40, 0.25, 0.15, 0.08]
        for t_util in target_util_levels:
            if current_util > t_util:
                current_profile["RevolvingUtilizationOfUnsecuredLines"] = t_util
                eval_res = predict_single(current_profile, pipeline=pipeline, model=model)
                current_prob = eval_res["probability"]
                if current_prob <= target_threshold:
                    steps.append({
                        "feature": "Credit Utilization",
                        "action": f"Pay down credit card balances to reduce revolving utilization from {current_util*100:.1f}% to {t_util*100:.1f}%.",
                        "impact": "Substantial Risk Reduction"
                    })
                    break
        else:
            if current_util > 0.10:
                steps.append({
                    "feature": "Credit Utilization",
                    "action": f"Pay down revolving balances to lower utilization from {current_util*100:.1f}% to 10.0%.",
                    "impact": "Substantial Risk Reduction"
                })
                current_profile["RevolvingUtilizationOfUnsecuredLines"] = 0.10
                eval_res = predict_single(current_profile, pipeline=pipeline, model=model)
                current_prob = eval_res["probability"]

    # Priority Action 3: Debt-to-Income / Debt Ratio Consolidation
    current_debt = float(current_profile.get("DebtRatio", 0.4) or 0.4)
    if current_prob > target_threshold and current_debt > 0.35:
        target_debt_levels = [0.35, 0.25, 0.18]
        for t_debt in target_debt_levels:
            if current_debt > t_debt:
                current_profile["DebtRatio"] = t_debt
                eval_res = predict_single(current_profile, pipeline=pipeline, model=model)
                current_prob = eval_res["probability"]
                if current_prob <= target_threshold:
                    steps.append({
                        "feature": "Debt-to-Income Ratio",
                        "action": f"Consolidate external debts to reduce monthly Debt Ratio from {current_debt:.2f} to {t_debt:.2f}.",
                        "impact": "Moderate Risk Reduction"
                    })
                    break
        else:
            if current_debt > 0.20:
                steps.append({
                    "feature": "Debt-to-Income Ratio",
                    "action": f"Lower monthly Debt Ratio from {current_debt:.2f} to 0.20 through debt restructuring.",
                    "impact": "Moderate Risk Reduction"
                })
                current_profile["DebtRatio"] = 0.20
                eval_res = predict_single(current_profile, pipeline=pipeline, model=model)
                current_prob = eval_res["probability"]

    # Priority Action 4: Monthly Income Enhancement (e.g. Co-signer / Additional verified income)
    current_income = float(current_profile.get("MonthlyIncome", 4000.0) or 4000.0)
    if current_prob > target_threshold:
        boosted_income = current_income * 1.30
        current_profile["MonthlyIncome"] = boosted_income
        eval_res = predict_single(current_profile, pipeline=pipeline, model=model)
        current_prob = eval_res["probability"]
        steps.append({
            "feature": "Verified Income / Co-signer",
            "action": f"Add a co-applicant or verify supplemental income to increase monthly cash flow from ${current_income:,.0f} to ${boosted_income:,.0f}.",
            "impact": "Capacity Enhancement"
        })

    # Construct comparison delta table
    delta_table = []
    for k in raw_input.keys():
        orig_v = raw_input.get(k)
        mod_v = current_profile.get(k)
        if orig_v != mod_v:
            delta_table.append({
                "Feature": k,
                "Original": f"{orig_v:,.2f}" if isinstance(orig_v, (int, float)) else str(orig_v),
                "Target Recourse": f"{mod_v:,.2f}" if isinstance(mod_v, (int, float)) else str(mod_v),
                "Change Direction": "Optimized"
            })

    return {
        "original_probability": base_prob,
        "original_probability_pct": round(base_prob * 100, 2),
        "target_probability": target_threshold,
        "counterfactual_probability": current_prob,
        "counterfactual_probability_pct": round(current_prob * 100, 2),
        "risk_reduction_pp": round((base_prob - current_prob) * 100, 2),
        "is_recourse_found": current_prob <= target_threshold + 0.05,
        "already_eligible": False,
        "actionable_steps": steps,
        "modified_profile": current_profile,
        "delta_table": delta_table
    }
