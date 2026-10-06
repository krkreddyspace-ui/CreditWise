"""
CreditWise — Natural Language Plain-English Underwriter Narrative Generator
==========================================================================
Generates structured executive decision summaries, translating statistical
probabilities, conformal uncertainty intervals, and SHAP local contributions
into actionable, professional credit underwriting narratives.
"""

from typing import Dict, Any, List, Optional


FEATURE_HUMAN_NAMES = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit Card & Revolving Line Utilization",
    "age": "Borrower Age Profile",
    "NumberOfTime30-59DaysPastDueNotWorse": "Short-Term Delinquencies (30–59 Days)",
    "DebtRatio": "Monthly Debt-to-Income Ratio",
    "MonthlyIncome": "Monthly Gross Income",
    "NumberOfOpenCreditLinesAndLoans": "Active Open Credit Lines & Loans",
    "NumberOfTimes90DaysLate": "Severe Delinquencies (90+ Days Late)",
    "NumberRealEstateLoansOrLines": "Mortgage & Real Estate Credit Lines",
    "NumberOfTime60-89DaysPastDueNotWorse": "Mid-Term Delinquencies (60–89 Days)",
    "NumberOfDependents": "Number of Household Dependents",
    "total_past_due": "Cumulative Delinquency History",
    "past_due_severity": "Delinquency Severity Index",
    "income_per_dependent": "Household Income per Dependent",
    "has_past_due": "Prior Delinquency Occurrence",
    "high_utilization": "High Revolving Utilization Flag",
    "has_real_estate": "Homeowner / Real Estate Collateral",
    "debt_to_income_monthly_debt": "Estimated Monthly Debt Burden"
}


def generate_executive_narrative(
    applicant_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    conformal_result: Optional[Dict[str, Any]] = None,
    shap_contributions: Optional[Dict[str, float]] = None,
    counterfactual_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generates a comprehensive underwriter executive narrative report.

    Parameters
    ----------
    applicant_data : dict of input features
    prediction_result : dict from predict_single
    conformal_result : dict from calculate_conformal_interval
    shap_contributions : dict of feature name -> shap value
    counterfactual_result : dict from generate_counterfactual_recourse

    Returns
    -------
    dict
        Structured executive report containing:
        - executive_headline
        - risk_level_description
        - key_strengths (bulleted list)
        - key_risk_drivers (bulleted list)
        - underwriting_recommendation
        - plain_english_summary (full narrative text)
    """
    prob = prediction_result.get("probability", 0.05)
    prob_pct = prob * 100
    risk_cat = prediction_result.get("risk_category", "Low Risk").upper()
    
    # 1. Conformal certainty text
    certainty_str = ""
    if conformal_result:
        lower_p = conformal_result.get("lower_bound_pct", 0.0)
        upper_p = conformal_result.get("upper_bound_pct", 10.0)
        conf_lvl = conformal_result.get("confidence_percentage", 95.0)
        certainty_str = f" Conformal statistical analysis establishes a {conf_lvl:.0f}% confidence interval of [{lower_p:.1f}%, {upper_p:.1f}%]."

    # 2. Executive Headline
    if "LOW" in risk_cat:
        headline = f"Favorable Credit Risk Profile — {prob_pct:.1f}% Estimated Default Probability"
        tone_verdict = "The applicant exhibits strong repayment capacity and disciplined credit management characteristics."
        recommendation = "Standard Processing: Approved for loan decision support subject to standard identity and income documentation verification."
    elif "MEDIUM" in risk_cat:
        headline = f"Moderate Credit Risk Profile — {prob_pct:.1f}% Estimated Default Probability"
        tone_verdict = "The applicant presents balanced financial metrics with specific localized risk flags that warrant manual underwriting review."
        recommendation = "Conditional / Manual Review: Verify debt obligations, examine recent bank statements, and consider structural mitigants (e.g. collateral or co-signer)."
    else:
        headline = f"Elevated Credit Risk Profile — {prob_pct:.1f}% Estimated Default Probability"
        tone_verdict = "The applicant demonstrates elevated default susceptibility driven by heavy credit burden and/or adverse delinquency indicators."
        recommendation = "Enhanced Due Diligence: Standard automated loan approval is not supported. Require senior underwriter review or debt restructuring prior to consideration."

    # 3. Analyze SHAP Strengths and Drivers
    strengths = []
    risk_drivers = []

    if shap_contributions:
        sorted_shap = sorted(shap_contributions.items(), key=lambda x: x[1])
        # Negative SHAP = Decreases default risk (Strengths)
        negative_factors = [item for item in sorted_shap if item[1] < -0.01][:3]
        # Positive SHAP = Increases default risk (Risk Drivers)
        positive_factors = [item for item in sorted_shap if item[1] > 0.01][-3:]
        positive_factors.reverse()

        for feat, val in negative_factors:
            human_name = FEATURE_HUMAN_NAMES.get(feat, feat)
            raw_val = applicant_data.get(feat, "N/A")
            if isinstance(raw_val, float):
                raw_str = f"{raw_val:.2f}"
            else:
                raw_str = str(raw_val)
            strengths.append(f"<b>{human_name}</b> (Recorded: {raw_str}): Exerts a strong downward stabilizing effect on risk ({val:+.3f} SHAP impact).")

        for feat, val in positive_factors:
            human_name = FEATURE_HUMAN_NAMES.get(feat, feat)
            raw_val = applicant_data.get(feat, "N/A")
            if isinstance(raw_val, float):
                raw_str = f"{raw_val:.2f}"
            else:
                raw_str = str(raw_val)
            risk_drivers.append(f"<b>{human_name}</b> (Recorded: {raw_str}): Exerts an upward pressure elevating risk ({val:+.3f} SHAP impact).")

    # Fallbacks if SHAP is empty
    if not strengths:
        income = applicant_data.get("MonthlyIncome", 0)
        if income and income > 5000:
            strengths.append(f"Monthly cash flow capacity of ${income:,.0f} provides debt service cushion.")
        late_30 = applicant_data.get("NumberOfTime30-59DaysPastDueNotWorse", 0)
        if late_30 == 0:
            strengths.append("Zero short-term delinquency incidents in recent billing cycles.")

    if not risk_drivers:
        util = applicant_data.get("RevolvingUtilizationOfUnsecuredLines", 0)
        if util > 0.60:
            risk_drivers.append(f"Revolving credit line utilization of {util*100:.1f}% indicates reliance on credit facilities.")
        debt = applicant_data.get("DebtRatio", 0)
        if debt > 0.50:
            risk_drivers.append(f"Debt-to-income ratio of {debt:.2f} reduces monthly discretionary margin.")

    # 4. Synthesize Full Paragraph
    full_text = (
        f"{tone_verdict}{certainty_str} "
        f"The primary stabilizing assets for this profile include: "
        f"{', '.join([s.replace('<b>', '').replace('</b>', '') for s in strengths]) if strengths else 'consistent baseline financial metrics'}. "
        f"{'Conversely, the main risk escalation factors identified by the AI framework are: ' + ', '.join([r.replace('<b>', '').replace('</b>', '') for r in risk_drivers]) if risk_drivers else 'No acute risk escalators detected.'}"
    )

    return {
        "executive_headline": headline,
        "tone_verdict": tone_verdict,
        "certainty_summary": certainty_str.strip(),
        "key_strengths": strengths,
        "key_risk_drivers": risk_drivers,
        "underwriting_recommendation": recommendation,
        "plain_english_summary": full_text
    }
