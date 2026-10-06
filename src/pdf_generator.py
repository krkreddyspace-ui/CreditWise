"""
CreditWise — Automated PDF Loan Audit & Decision Dossier Generator
=================================================================
Generates institutional-grade, publication-ready PDF Credit Risk Audit Dossiers
containing applicant profile, calibrated default probability, conformal intervals,
SHAP feature breakdowns, and underwriter narratives.
"""

from typing import Dict, Any, Optional
import io
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY


def generate_credit_dossier_pdf(
    applicant_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    conformal_result: Optional[Dict[str, Any]] = None,
    shap_contributions: Optional[Dict[str, float]] = None,
    narrative_result: Optional[Dict[str, Any]] = None,
    counterfactual_result: Optional[Dict[str, Any]] = None,
    applicant_id: str = "CW-APP-2026-0849"
) -> bytes:
    """Generates a styled PDF Credit Risk Dossier in memory and returns bytes.

    Parameters
    ----------
    applicant_data : dict of input features
    prediction_result : dict from predict_single
    conformal_result : dict from calculate_conformal_interval
    shap_contributions : dict of feature name -> shap value
    narrative_result : dict from generate_executive_narrative
    counterfactual_result : dict from generate_counterfactual_recourse
    applicant_id : str reference ID

    Returns
    -------
    bytes
        Raw PDF file bytes suitable for download or persistence.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a")
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2563eb")
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )
    bold_body = ParagraphStyle(
        "BoldBody",
        parent=body_style,
        fontName="Helvetica-Bold"
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748b")
    )

    story = []

    # 1. Header Banner
    eval_date = datetime.datetime.now().strftime("%B %d, %Y - %H:%M:%S UTC")
    header_data = [
        [
            Paragraph("<b>CREDITWISE</b> — AI CREDIT RISK INTELLIGENCE", subtitle_style),
            Paragraph(f"<b>Dossier ID:</b> {applicant_id}", ParagraphStyle("RightAlign", parent=subtitle_style, alignment=TA_RIGHT))
        ],
        [
            Paragraph("Applicant Credit Risk & Explainability Dossier", title_style),
            Paragraph(f"Generated: {eval_date}", ParagraphStyle("RightAlign", parent=body_style, alignment=TA_RIGHT))
        ]
    ]
    t_header = Table(header_data, colWidths=[340, 190])
    t_header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceBefore=2, spaceAfter=12))

    # 2. Executive Decision Summary Banner
    prob = prediction_result.get("probability", 0.05)
    prob_pct = prob * 100
    risk_cat = prediction_result.get("risk_category", "Low Risk")

    cat_color = colors.HexColor("#059669") if "Low" in risk_cat else (colors.HexColor("#d97706") if "Med" in risk_cat else colors.HexColor("#dc2626"))

    conf_text = "N/A"
    if conformal_result:
        conf_text = f"[{conformal_result.get('lower_bound_pct', 0.0):.1f}%, {conformal_result.get('upper_bound_pct', 10.0):.1f}%] @ {conformal_result.get('confidence_percentage', 95):.0f}% Confidence"

    kpi_data = [
        [
            Paragraph("<b>ESTIMATED DEFAULT RISK</b>", body_style),
            Paragraph("<b>RISK CLASSIFICATION</b>", body_style),
            Paragraph("<b>CONFORMAL UNCERTAINTY BAND</b>", body_style)
        ],
        [
            Paragraph(f"<font size='16' color='#0f172a'><b>{prob_pct:.2f}%</b></font>", body_style),
            Paragraph(f"<font size='14' color='{cat_color.hexval()}'><b>{risk_cat.upper()}</b></font>", body_style),
            Paragraph(f"<font size='11' color='#0f172a'><b>{conf_text}</b></font>", body_style)
        ]
    ]
    t_kpi = Table(kpi_data, colWidths=[175, 175, 180])
    t_kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 10))

    # 3. Applicant Financial Profile Table
    story.append(Paragraph("1. Applicant Financial Profile", h2_style))
    
    app_rows = [
        [
            Paragraph("<b>Metric</b>", bold_body), Paragraph("<b>Value</b>", bold_body),
            Paragraph("<b>Metric</b>", bold_body), Paragraph("<b>Value</b>", bold_body)
        ],
        [
            "Borrower Age", f"{applicant_data.get('age', 'N/A')} yrs",
            "Monthly Gross Income", f"${applicant_data.get('MonthlyIncome', 0):,.2f}" if applicant_data.get("MonthlyIncome") is not None else "N/A"
        ],
        [
            "Revolving Utilization", f"{float(applicant_data.get('RevolvingUtilizationOfUnsecuredLines', 0) or 0)*100:.1f}%",
            "Debt-to-Income Ratio", f"{float(applicant_data.get('DebtRatio', 0) or 0):.2f}"
        ],
        [
            "Open Credit Lines", str(applicant_data.get("NumberOfOpenCreditLinesAndLoans", 0)),
            "Real Estate Lines", str(applicant_data.get("NumberRealEstateLoansOrLines", 0))
        ],
        [
            "Delinquencies (30-59d)", str(applicant_data.get("NumberOfTime30-59DaysPastDueNotWorse", 0)),
            "Severe Delinquencies (90+d)", str(applicant_data.get("NumberOfTimes90DaysLate", 0))
        ],
        [
            "Delinquencies (60-89d)", str(applicant_data.get("NumberOfTime60-89DaysPastDueNotWorse", 0)),
            "Number of Dependents", str(applicant_data.get("NumberOfDependents", 0))
        ]
    ]
    t_profile = Table(app_rows, colWidths=[150, 115, 150, 115])
    t_profile.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5)
    ]))
    story.append(t_profile)
    story.append(Spacer(1, 10))

    # 4. Underwriter Executive Narrative
    if narrative_result:
        story.append(Paragraph("2. AI Underwriter Executive Assessment", h2_style))
        story.append(Paragraph(f"<b>Verdict:</b> {narrative_result.get('executive_headline', '')}", bold_body))
        story.append(Spacer(1, 3))
        story.append(Paragraph(narrative_result.get("plain_english_summary", ""), body_style))
        story.append(Spacer(1, 5))
        story.append(Paragraph(f"<b>Recommendation:</b> {narrative_result.get('underwriting_recommendation', '')}", bold_body))
        story.append(Spacer(1, 10))

    # 5. Local SHAP Explainability Breakdown
    if shap_contributions:
        story.append(Paragraph("3. Explainable AI Feature Attribution (SHAP)", h2_style))
        shap_rows = [
            [
                Paragraph("<b>Feature Attribute</b>", bold_body),
                Paragraph("<b>Applicant Value</b>", bold_body),
                Paragraph("<b>SHAP Value</b>", bold_body),
                Paragraph("<b>Risk Direction</b>", bold_body)
            ]
        ]
        sorted_shap = sorted(shap_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
        for feat, val in sorted_shap:
            direction = "Elevates Risk" if val > 0 else "Lowers Risk"
            val_col = colors.HexColor("#dc2626") if val > 0 else colors.HexColor("#059669")
            u_val = applicant_data.get(feat, "N/A")
            u_str = f"{u_val:.2f}" if isinstance(u_val, float) else str(u_val)
            shap_rows.append([
                feat,
                u_str,
                f"{val:+.4f}",
                direction
            ])
        t_shap = Table(shap_rows, colWidths=[200, 110, 110, 110])
        t_shap.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8)
        ]))
        story.append(t_shap)
        story.append(Spacer(1, 10))

    # 6. Actionable Counterfactual Recourse (if available)
    if counterfactual_result and not counterfactual_result.get("already_eligible"):
        story.append(Paragraph("4. Actionable Algorithmic Recourse (Path to Approval)", h2_style))
        steps = counterfactual_result.get("actionable_steps", [])
        if steps:
            for s in steps:
                if isinstance(s, dict):
                    story.append(Paragraph(f"• <b>{s.get('feature', '')}:</b> {s.get('action', '')}", body_style))
                else:
                    story.append(Paragraph(f"• {str(s)}", body_style))
            t_prob = counterfactual_result.get("counterfactual_probability_pct", 25.0)
            story.append(Spacer(1, 3))
            story.append(Paragraph(f"<b>Estimated Post-Recourse Risk:</b> {t_prob:.1f}% (Achieves Low Risk Tier)", bold_body))
            story.append(Spacer(1, 10))

    # 7. Compliance & Signoff Block
    story.append(KeepTogether([
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceBefore=6, spaceAfter=8),
        Paragraph("<b>Institutional Governance & Audit Sign-off</b>", h2_style),
        Paragraph(
            "<b>Compliance Notice:</b> CreditWise is an explainable machine learning decision-support tool. All risk scores, confidence intervals, and SHAP attributions must be reviewed by an authorized human underwriter in adherence with fair lending regulations (ECOA / FCRA).",
            disclaimer_style
        ),
        Spacer(1, 15),
        Table([
            [
                Paragraph("<b>Underwriting Officer Signature:</b> ___________________________", body_style),
                Paragraph("<b>Audit Review Date:</b> ___________________________", body_style)
            ]
        ], colWidths=[265, 265])
    ]))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
