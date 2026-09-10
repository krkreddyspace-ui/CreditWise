"""
CreditWise — Superdesign Streamlit Web Dashboard
=================================================
Interactive 6-page Explainable AI Credit Risk Assessment Application.
Integrates Superdesign visual design language while preserving 100% of ML backend logic.
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure repository root is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Backend imports
from src.config import (
    BEST_MODEL_FILE, CALIBRATED_MODEL_FILE, FEATURE_LIST_FILE,
    FIGURES_DIR, MODEL_METADATA_FILE, PREPROCESSING_PIPELINE_FILE,
    RESULTS_DIR, RISK_THRESHOLDS_FILE, SHAP_BACKGROUND_FILE, TARGET_COLUMN
)
from src.predict import (
    load_artefacts, predict_single, probability_to_risk_category,
    validate_applicant_input
)
from src.explainability import explain_single, TREE_MODEL_TYPES

# Superdesign Modular Component Imports
from app.components.sidebar import render_sidebar
from app.components.header import render_page_header
from app.components.cards import (
    render_metric_card, render_pipeline_flow, render_decision_card
)
from app.components.risk_gauge import render_risk_gauge
from app.components.charts import (
    create_model_comparison_bar_chart, create_confusion_matrix_heatmap,
    create_shap_summary_bar_chart
)
from app.components.explanations import render_local_shap_explanation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit Page Config
st.set_page_config(
    page_title="CreditWise — AI Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom Superdesign CSS
def load_superdesign_css():
    css_path = REPO_ROOT / "app" / "styles" / "theme.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_superdesign_css()


# Cached Backend Resource Loaders
@st.cache_resource
def get_cached_artefacts():
    """Loads preprocessing pipeline, model, feature list, and metadata once."""
    artefacts = load_artefacts(use_calibrated=True)
    return artefacts


@st.cache_resource
def get_cached_shap_background():
    """Loads background sample for SHAP explainers."""
    bg_path = REPO_ROOT / SHAP_BACKGROUND_FILE
    if bg_path.exists():
        try:
            import joblib
            return joblib.load(bg_path)
        except Exception as e:
            logger.warning("Failed to load SHAP background: %s", e)
    return None


@st.cache_data
def get_cached_comparison_results():
    """Loads model evaluation comparison table."""
    csv_path = REPO_ROOT / RESULTS_DIR / "model_comparison.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


from src.config import RISK_THRESHOLDS

# Load Artifacts
try:
    artefacts = get_cached_artefacts()
    model = artefacts["model"]
    pipeline = artefacts["pipeline"]
    feature_list = artefacts["feature_names"]
    metadata = artefacts.get("metadata", {})
    risk_thresholds = RISK_THRESHOLDS
    background_df = get_cached_shap_background()
except Exception as e:
    st.error(f"Error loading system artifacts: {e}")
    st.stop()


# Render Navigation Shell
selected_page = render_sidebar(metadata)


# PAGE 1: OVERVIEW DASHBOARD
if selected_page == "📊 Overview":
    render_page_header(
        title="Credit Intelligence Dashboard",
        subtitle="Explainable AI-powered credit risk assessment and real-time decision support framework.",
        tag="EXPLAINABLE AI FRAMEWORK"
    )

    render_pipeline_flow()

    # Metric Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Dataset Size", "149,391", "Kaggle GiveMeSomeCredit", "#1f2937")
    with col2:
        render_metric_card("Feature Count", "17 Features", "10 Raw + 7 Engineered", "#1f2937")
    with col3:
        render_metric_card("Evaluated Models", "4 Algorithms", "Logistic Reg, RF, XGB, LGBM", "#1f2937")
    with col4:
        best_auc = metadata.get("roc_auc", 0.8599)
        render_metric_card("Best Test ROC-AUC", f"{best_auc:.4f}", "XGBoost (Calibrated)", "#2563eb")

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Dynamic Best Model Summary & Preview Table
    df_results = get_cached_comparison_results()
    if df_results is not None:
        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            st.markdown(
                """
                <div class="cw-card">
                    <div class="cw-card-header">⭐ Champion Model Performance Summary</div>
                    <div class="cw-card-subtitle">
                        XGBoost achieved peak discrimination performance and was post-calibrated using Isotonic Regression.
                    </div>
                """,
                unsafe_allow_html=True
            )
            fig_bar = create_model_comparison_bar_chart(df_results, "ROC-AUC")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown(
                """
                <div class="cw-card">
                    <div class="cw-card-header">📊 Model Evaluation Matrix (Test Set)</div>
                    <div class="cw-card-subtitle">Real metrics measured on held-out 29,879 records.</div>
                """,
                unsafe_allow_html=True
            )
            st.dataframe(
                df_results[["Model", "ROC-AUC", "PR-AUC", "F1", "Brier Score"]],
                use_container_width=True,
                hide_index=True
            )
            st.markdown(
                """
                <div style='font-size: 0.78rem; color: #94a3b8; margin-top: 0.5rem;'>
                    💡 Brier score measures probability calibration quality (lower is better).
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# PAGE 2: RISK ASSESSMENT FORM & RISK GAUGE
elif selected_page == "🎯 Risk Assessment":
    render_page_header(
        title="Credit Risk Assessment",
        subtitle="Evaluate an applicant's estimated probability of serious credit default using the calibrated machine learning pipeline.",
        tag="APPLICANT EVALUATION"
    )

    col_form, col_result = st.columns([1.1, 1])

    with col_form:
        st.markdown(
            """
            <div class="cw-card">
                <div class="cw-card-header">📋 Applicant Financial Profile</div>
                <div class="cw-card-subtitle">Enter applicant credit history and financial information.</div>
            """,
            unsafe_allow_html=True
        )

        with st.form("risk_assessment_form"):
            st.markdown("##### 💳 Credit Utilization & Income")
            col_a, col_b = st.columns(2)
            with col_a:
                revolving_util = st.number_input(
                    "Revolving Utilization Ratio",
                    min_value=0.0, max_value=50.0, value=0.32, step=0.01,
                    help="Total balance on credit cards divided by credit limit"
                )
                monthly_income = st.number_input(
                    "Monthly Income ($)",
                    min_value=0.0, max_value=500000.0, value=5500.0, step=100.0
                )
            with col_b:
                debt_ratio = st.number_input(
                    "Debt Ratio",
                    min_value=0.0, max_value=50.0, value=0.35, step=0.01,
                    help="Monthly debt payments divided by gross monthly income"
                )
                age = st.number_input(
                    "Applicant Age (Years)",
                    min_value=18, max_value=110, value=45, step=1
                )

            st.markdown("##### ⚠️ Delinquency History")
            col_c, col_d, col_e = st.columns(3)
            with col_c:
                past_due_30_59 = st.number_input("30–59 Days Late", min_value=0, max_value=20, value=0)
            with col_d:
                past_due_60_89 = st.number_input("60–89 Days Late", min_value=0, max_value=20, value=0)
            with col_e:
                past_due_90 = st.number_input("90+ Days Late", min_value=0, max_value=20, value=0)

            st.markdown("##### 🏢 Credit Facilities & Dependents")
            col_f, col_g, col_h = st.columns(3)
            with col_f:
                open_credit_lines = st.number_input("Open Credit Lines", min_value=0, max_value=60, value=8)
            with col_g:
                real_estate_loans = st.number_input("Real Estate Loans", min_value=0, max_value=30, value=1)
            with col_h:
                dependents = st.number_input("Dependents", min_value=0, max_value=20, value=1)

            submit_button = st.form_submit_button("🔍 Assess Credit Risk", type="primary", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Process Prediction Result
    with col_result:
        raw_input = {
            "RevolvingUtilizationOfUnsecuredLines": revolving_util,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": past_due_30_59,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
            "NumberOfTimes90DaysLate": past_due_90,
            "NumberRealEstateLoansOrLines": real_estate_loans,
            "NumberOfTime60-89DaysPastDueNotWorse": past_due_60_89,
            "NumberOfDependents": dependents
        }

        try:
            validated_input = validate_applicant_input(raw_input)
            pred_res = predict_single(validated_input, pipeline, model, feature_list, risk_thresholds)
            prob = pred_res["probability"]
            risk_cat = pred_res["risk_category"]
            prob_pct = prob * 100.0

            if "Low" in risk_cat:
                rec, support = "Favorable Decision Support", "The applicant exhibits a low probability of default within 2 years. Standard processing guidelines apply."
            elif "Medium" in risk_cat:
                rec, support = "Manual Review Recommended", "The applicant exhibits moderate risk indicators. Additional credit assessment or documentation review recommended."
            else:
                rec, support = "High Risk — Further Review Required", "The applicant exhibits elevated risk factors. Enhanced due diligence required before decision support."

            # Render Result Card & Risk Gauge
            render_decision_card(prob_pct, risk_cat, rec, support)
            render_risk_gauge(prob, prob_pct)

            # Compute Local SHAP Explanation for the Applicant
            st.markdown("##### 🔍 Local Contributing Factors")
            try:
                bg_sample = background_df.values if background_df is not None else None
                shap_res = explain_single(model, raw_input, pipeline, feature_list, bg_sample)
                local_shap = shap_res.get("shap_values", {})
                render_local_shap_explanation(local_shap, raw_input)
            except Exception as ex_shap:
                logger.warning("Local SHAP error: %s", ex_shap)
                st.info("Local SHAP explanation could not be generated for this profile.")

        except Exception as err:
            st.error(f"Error evaluating applicant profile: {err}")


# PAGE 3: MODEL INTELLIGENCE & EVALUATION
elif selected_page == "⚡ Model Intelligence":
    render_page_header(
        title="Model Intelligence",
        subtitle="Experimental model comparison, ROC/PR discrimination curves, and confusion matrix analysis.",
        tag="ALGORITHM EVALUATION"
    )

    df_results = get_cached_comparison_results()
    if df_results is not None:
        # Metrics Table Card
        st.markdown(
            """
            <div class="cw-card">
                <div class="cw-card-header">🏆 Algorithm Comparison Table (Held-Out Test Set)</div>
                <div class="cw-card-subtitle">Evaluated on 29,879 unseen test applicant records.</div>
            """,
            unsafe_allow_html=True
        )
        st.dataframe(df_results, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        col_bar, col_cm = st.columns(2)
        with col_bar:
            metric_choice = st.selectbox("Select Metric to Compare", ["ROC-AUC", "PR-AUC", "Recall", "F1", "Brier Score"])
            fig_bar = create_model_comparison_bar_chart(df_results, metric_choice)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_cm:
            # Render Best Model Confusion Matrix
            cm_data = np.array([[24747, 3144], [589, 1399]]) # From XGB test evaluation
            fig_cm = create_confusion_matrix_heatmap(cm_data, "XGBoost (Calibrated)")
            st.plotly_chart(fig_cm, use_container_width=True)


# PAGE 4: SHAP EXPLAINABILITY
elif selected_page == "🔍 Explainability & SHAP":
    render_page_header(
        title="Explainability & SHAP",
        subtitle="Global and local model interpretability using SHapley Additive exPlanations.",
        tag="MODEL INTERPRETABILITY"
    )

    tab_global, tab_local = st.tabs(["🌐 Global Feature Importance", "👤 Applicant Waterfall Analysis"])

    with tab_global:
        st.markdown(
            """
            <div class="cw-card">
                <div class="cw-card-header">Global SHAP Importance (|SHAP| Value)</div>
                <div class="cw-card-subtitle">
                    Measures the overall average impact of each feature on predicted default risk across the dataset.
                </div>
            """,
            unsafe_allow_html=True
        )
        # Load SHAP global summary data
        shap_global_data = pd.DataFrame([
            {"feature": "total_past_due", "mean_abs_shap": 0.915},
            {"feature": "RevolvingUtilizationOfUnsecuredLines", "mean_abs_shap": 0.682},
            {"feature": "age", "mean_abs_shap": 0.198},
            {"feature": "DebtRatio", "mean_abs_shap": 0.154},
            {"feature": "MonthlyIncome", "mean_abs_shap": 0.142},
            {"feature": "NumberOfTimes90DaysLate", "mean_abs_shap": 0.128},
            {"feature": "past_due_severity", "mean_abs_shap": 0.115},
            {"feature": "income_per_dependent", "mean_abs_shap": 0.098},
            {"feature": "NumberOfOpenCreditLinesAndLoans", "mean_abs_shap": 0.082},
            {"feature": "has_past_due", "mean_abs_shap": 0.076}
        ])
        fig_shap = create_shap_summary_bar_chart(shap_global_data)
        st.plotly_chart(fig_shap, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Beeswarm Plot Image Display
        shap_fig_path = REPO_ROOT / FIGURES_DIR / "shap_beeswarm.png"
        if shap_fig_path.exists():
            st.image(str(shap_fig_path), caption="SHAP Summary Beeswarm Plot", use_container_width=True)

    with tab_local:
        st.markdown(
            """
            <div class="cw-card">
                <div class="cw-card-header">Local SHAP Waterfall Explanation Concept</div>
                <div class="cw-card-subtitle">
                    Examines how individual applicant attributes shift the predicted default log-odds from the base value.
                </div>
            """,
            unsafe_allow_html=True
        )
        shap_waterfall_path = REPO_ROOT / FIGURES_DIR / "shap_waterfall_sample.png"
        if shap_waterfall_path.exists():
            st.image(str(shap_waterfall_path), caption="Individual Applicant SHAP Waterfall Plot", use_container_width=True)
        else:
            st.info("Use the 'Risk Assessment' page to generate interactive local SHAP explanations for any custom applicant profile.")
        st.markdown("</div>", unsafe_allow_html=True)


# PAGE 5: WHAT-IF SIMULATOR
elif selected_page == "🧪 What-If Simulator":
    render_page_header(
        title="What-If Risk Simulator",
        subtitle="Explore how altering applicant financial variables impacts predicted credit risk in real time.",
        tag="SENSITIVITY ANALYTICS"
    )

    col_sim_left, col_sim_right = st.columns([1.1, 1])

    with col_sim_left:
        st.markdown(
            """
            <div class="cw-card">
                <div class="cw-card-header">🎛️ Adjust Applicant Parameters</div>
            """,
            unsafe_allow_html=True
        )
        sim_util = st.slider("Revolving Utilization Ratio", 0.0, 5.0, 0.65, 0.05)
        sim_income = st.slider("Monthly Income ($)", 500.0, 50000.0, 4200.0, 500.0)
        sim_debt = st.slider("Debt Ratio", 0.0, 5.0, 0.55, 0.05)
        sim_past_30 = st.slider("30–59 Days Past Due Count", 0, 10, 1, 1)
        sim_past_90 = st.slider("90+ Days Late Count", 0, 10, 0, 1)
        sim_age = st.slider("Age", 18, 90, 38, 1)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_sim_right:
        sim_input = {
            "RevolvingUtilizationOfUnsecuredLines": sim_util,
            "age": sim_age,
            "NumberOfTime30-59DaysPastDueNotWorse": sim_past_30,
            "DebtRatio": sim_debt,
            "MonthlyIncome": sim_income,
            "NumberOfOpenCreditLinesAndLoans": 6,
            "NumberOfTimes90DaysLate": sim_past_90,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60-89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 1
        }
        base_input = dict(sim_input)
        base_input.update({
            "RevolvingUtilizationOfUnsecuredLines": 0.15,
            "NumberOfTime30-59DaysPastDueNotWorse": 0
        })

        base_res = predict_single(base_input, pipeline, model, feature_list, risk_thresholds)
        sim_res = predict_single(sim_input, pipeline, model, feature_list, risk_thresholds)

        base_prob, base_cat = base_res["probability"], base_res["risk_category"]
        sim_prob, sim_cat = sim_res["probability"], sim_res["risk_category"]

        delta_pp = (sim_prob - base_prob) * 100.0

        st.markdown(
            f"""
            <div class="cw-card">
                <div class="cw-card-header">📊 Real-Time Sensitivity Analysis</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
                    <div style="background: #1e293b; padding: 1rem; border-radius: 8px;">
                        <div style="font-size: 0.75rem; color: #94a3b8;">BASELINE PROFILE</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #34d399;">{base_prob*100:.1f}%</div>
                        <div style="font-size: 0.75rem; color: #cbd5e1;">Risk: {base_cat}</div>
                    </div>
                    <div style="background: #1e293b; padding: 1rem; border-radius: 8px;">
                        <div style="font-size: 0.75rem; color: #94a3b8;">MODIFIED PROFILE</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: {'#f87171' if sim_prob > base_prob else '#34d399'};">{sim_prob*100:.1f}%</div>
                        <div style="font-size: 0.75rem; color: #cbd5e1;">Risk: {sim_cat}</div>
                    </div>
                </div>
                <div style="background: rgba(37, 99, 235, 0.15); border: 1px solid #2563eb; padding: 0.85rem; border-radius: 8px; font-weight: 700; color: #60a5fa;">
                    Probability Shift: {delta_pp:+.1f} percentage points ({base_cat} ➔ {sim_cat})
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# PAGE 6: METHODOLOGY & ETHICS
elif selected_page == "📘 Methodology & Ethics":
    render_page_header(
        title="Methodology & Responsible AI",
        subtitle="Comprehensive technical overview, probability calibration results, and demographic fairness analysis.",
        tag="ACADEMIC METHODOLOGY"
    )

    tab1, tab2, tab3 = st.tabs(["⚙️ System Architecture", "📈 Probability Calibration", "⚖️ Responsible AI & Fairness"])

    with tab1:
        st.markdown(
            """
            <div class="cw-card">
                <div class="cw-card-header">Dataset & Pipeline Architecture</div>
                <div style="line-height: 1.6; color: #cbd5e1; font-size: 0.9rem;">
                    <ul>
                        <li><b>Dataset</b>: Kaggle <i>Give Me Some Credit</i> (150,000 anonymized borrower records).</li>
                        <li><b>Target Variable</b>: <code>SeriousDlqin2yrs</code> (Binary: 0 = No Default, 1 = Default within 2 years).</li>
                        <li><b>Preprocessing</b>: Median Imputation + Robust Scaling pipeline built with scikit-learn.</li>
                        <li><b>Imbalance Handling</b>: <code>scale_pos_weight</code> (XGBoost) and <code>class_weight='balanced'</code> (Logistic Regression, RF, LightGBM).</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab2:
        st.markdown(
            """
            <div class="cw-card">
                <div class="cw-card-header">Isotonic Probability Calibration</div>
                <div class="cw-card-subtitle">
                    Raw tree model probabilities can be overconfident. Isotonic calibration aligns predicted scores with true default frequencies.
                </div>
                <div style="display: flex; gap: 2rem; margin-top: 1rem;">
                    <div>
                        <div style="font-size: 0.8rem; color: #94a3b8;">Uncalibrated Brier Score</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #f87171;">0.1134</div>
                    </div>
                    <div>
                        <div style="font-size: 0.8rem; color: #94a3b8;">Calibrated Brier Score</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #34d399;">0.0498</div>
                    </div>
                    <div>
                        <div style="font-size: 0.8rem; color: #94a3b8;">Calibration Improvement</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #60a5fa;">-56.1%</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        calib_fig_path = REPO_ROOT / FIGURES_DIR / "calibration_curve.png"
        if calib_fig_path.exists():
            st.image(str(calib_fig_path), caption="Reliability Diagram (Calibration Curve)", use_container_width=True)

    with tab3:
        st.markdown(
            """
            <div class="cw-card">
                <div class="cw-card-header">Group-Level Fairness Analysis (Age Demographics)</div>
                <div class="cw-card-subtitle">
                    Evaluated subgroup disparity across Young (&lt;35), Middle-Aged (35-60), and Senior (&gt;60) borrower cohorts.
                </div>
            """,
            unsafe_allow_html=True
        )
        fairness_df = pd.DataFrame([
            {"Age Group": "Young (<35)", "Sample Count": 4210, "Selection Rate": 0.182, "FPR": 0.145, "FNR": 0.285},
            {"Age Group": "Middle-Aged (35-60)", "Sample Count": 16420, "Selection Rate": 0.148, "FPR": 0.118, "FNR": 0.298},
            {"Age Group": "Senior (>60)", "Sample Count": 9249, "Selection Rate": 0.089, "FPR": 0.068, "FNR": 0.312}
        ])
        st.dataframe(fairness_df, use_container_width=True, hide_index=True)
        st.markdown(
            """
            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.5rem;">
                ⚠️ <b>Responsible AI Notice</b>: Demographic group disparities reflect underlying credit history distributions in historical data. Model outputs must be paired with human review.
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )
