"""
CreditWise — Streamlit Dashboard
===================================
Run with:  streamlit run app/app.py
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

# ── Path setup ──────────────────────────────────────────────────────────────
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    APP_DISCLAIMER, BEST_MODEL_FILE, CALIBRATED_MODEL_FILE,
    FIGURES_DIR, MODEL_COMPARISON_FILE, MODEL_METADATA_FILE,
    PREPROCESSING_PIPELINE_FILE, RISK_LABELS, RISK_THRESHOLDS,
)

logging.basicConfig(level=logging.WARNING)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise — AI Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family:'Inter',sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(175deg,#0f172a 0%,#1a2744 100%);
}
section[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
section[data-testid="stSidebar"] .stRadio label { font-size:0.93rem; }

/* hero strip */
.hero {
    background: linear-gradient(135deg,#0f172a 0%,#1e3a5f 60%,#0f172a 100%);
    border-radius:14px; padding:36px 40px; margin-bottom:24px;
    border:1px solid #1e3a5f;
}
.hero h1 { font-size:2.4rem; font-weight:700; color:#38bdf8; margin:0 0 6px; }
.hero p  { color:#94a3b8; font-size:1.05rem; margin:0; }

/* metric cards */
[data-testid="metric-container"] {
    background:#1e293b; border:1px solid #334155;
    border-radius:12px; padding:14px 18px;
}
[data-testid="metric-container"] label { color:#94a3b8 !important; font-size:0.78rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#f1f5f9 !important; font-size:1.55rem !important; font-weight:600 !important;
}

/* risk badges */
.badge { display:inline-block; padding:5px 18px; border-radius:20px;
         font-weight:700; font-size:0.9rem; letter-spacing:.3px; }
.badge-low    { background:#166534; color:#dcfce7; }
.badge-medium { background:#854d0e; color:#fef9c3; }
.badge-high   { background:#7f1d1d; color:#fee2e2; }

/* info/warning cards */
.card {
    padding:16px 20px; border-radius:10px; margin-bottom:14px;
    border-left:4px solid;
}
.card-info    { background:#0f2744; border-color:#38bdf8; }
.card-warning { background:#1c1206; border-color:#f59e0b; }
.card-success { background:#052e16; border-color:#22c55e; }

/* table header */
thead tr th { background:#1e293b !important; color:#94a3b8 !important; }

/* divider */
hr { border-color:#1e293b; }

/* step items */
.step { display:flex; gap:14px; align-items:flex-start; margin-bottom:10px; }
.step-num {
    background:#1e3a5f; color:#38bdf8; font-weight:700;
    border-radius:50%; width:30px; height:30px; min-width:30px;
    display:flex; align-items:center; justify-content:center; font-size:.85rem;
}
.step-text { color:#cbd5e1; font-size:.93rem; padding-top:4px; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def _load_artefacts():
    try:
        from src.predict import load_artefacts
        return load_artefacts(), None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def _load_shap_background():
    import joblib
    p = PROJECT_ROOT / "models" / "shap_background.joblib"
    return joblib.load(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def _load_comparison():
    if MODEL_COMPARISON_FILE.exists():
        return pd.read_csv(MODEL_COMPARISON_FILE)
    return None


@st.cache_data(show_spinner=False)
def _load_metadata():
    if MODEL_METADATA_FILE.exists():
        with open(MODEL_METADATA_FILE) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def _load_shap_importance():
    p = PROJECT_ROOT / "reports" / "results" / "shap_importance.csv"
    return pd.read_csv(p) if p.exists() else None


def _model_ready():
    return BEST_MODEL_FILE.exists() and PREPROCESSING_PIPELINE_FILE.exists()


def _not_trained_msg():
    st.error("⚠️ Model not yet trained. Run:\n```\npython -m src.train\n```")
    st.stop()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:18px 0 10px'>
      <div style='font-size:2rem'>💳</div>
      <div style='font-size:1.25rem;font-weight:700;color:#38bdf8'>CreditWise</div>
      <div style='font-size:0.75rem;color:#64748b;margin-top:2px'>AI Credit Risk Assessment</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠  Home", "🔍  Risk Assessment", "📊  Model Comparison",
         "🧠  Explainability", "🔧  What-If Simulator", "📖  Methodology"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    meta = _load_metadata()
    if meta:
        st.markdown(f"<small style='color:#475569'>Best model: **{meta.get('best_model_name','—')}**  \nROC-AUC: **{next((m['roc_auc'] for m in meta.get('all_metrics',[]) if m['model']==meta.get('best_model_name')),0):.4f}**</small>", unsafe_allow_html=True)
    st.markdown("<small style='color:#334155'>Academic prototype · Not financial advice</small>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div class="hero">
      <h1>💳 CreditWise</h1>
      <p>An Explainable AI Framework for Intelligent Credit Risk Assessment and Loan Decision Support</p>
    </div>
    """, unsafe_allow_html=True)

    # Status row
    c1, c2, c3, c4 = st.columns(4)
    raw_ok = (PROJECT_ROOT / "data" / "raw" / "cs-training.csv").exists()
    with c1:
        st.metric("Dataset", "✅ Loaded" if raw_ok else "❌ Missing")
    with c2:
        st.metric("Model", "✅ Trained" if _model_ready() else "⚠️ Not trained")
    with c3:
        st.metric("Records", "149,391")
    with c4:
        st.metric("Features", "17  (10 base + 7 engineered)")

    st.markdown("---")
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown("### 🎯 What CreditWise Does")
        st.markdown("""
        CreditWise is an **AI-assisted decision-support system** that estimates the
        probability of credit default for a loan applicant and explains the key factors
        driving that prediction using SHAP.

        It is a college Major Project academic prototype built to demonstrate
        end-to-end machine learning for credit risk analysis.
        """)

        st.markdown("### 🔄 Pipeline")
        steps = [
            ("Data", "Give Me Some Credit dataset — 150K historical borrower records"),
            ("Clean", "Drop 609 duplicates · replace suspicious delinquency codes"),
            ("Engineer", "7 financially justified derived features"),
            ("Split", "80 / 20 stratified train-test split (seed=42)"),
            ("Train", "Logistic Regression · Random Forest · XGBoost · LightGBM"),
            ("Evaluate", "ROC-AUC, PR-AUC, Recall, F1, Brier score on held-out test set"),
            ("Explain", "SHAP global importance + per-applicant local waterfall"),
        ]
        for n, (title, desc) in enumerate(steps, 1):
            st.markdown(f"""<div class="step">
              <div class="step-num">{n}</div>
              <div class="step-text"><b>{title}</b> — {desc}</div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        if meta and meta.get("all_metrics"):
            st.markdown("### 📈 Live Results")
            best_name = meta.get("best_model_name", "")
            for m in sorted(meta["all_metrics"], key=lambda x: x["roc_auc"], reverse=True):
                is_best = m["model"] == best_name
                prefix  = "⭐ " if is_best else "   "
                colour  = "#38bdf8" if is_best else "#64748b"
                st.markdown(
                    f"<div style='background:#1e293b;border-radius:8px;padding:10px 14px;"
                    f"margin-bottom:8px;border-left:3px solid {colour}'>"
                    f"<b style='color:{colour}'>{prefix}{m['model']}</b><br>"
                    f"<small style='color:#94a3b8'>ROC-AUC: <b style='color:#f1f5f9'>{m['roc_auc']:.4f}</b> &nbsp;|&nbsp; "
                    f"Recall: <b style='color:#f1f5f9'>{m['recall']:.4f}</b> &nbsp;|&nbsp; "
                    f"PR-AUC: <b style='color:#f1f5f9'>{m['pr_auc']:.4f}</b></small></div>",
                    unsafe_allow_html=True,
                )

        # Risk thresholds
        st.markdown("### 🎚️ Risk Thresholds")
        for label, lo, hi, cls in [
            ("🟢 Low Risk",    "0.00", "0.30", "badge-low"),
            ("🟡 Medium Risk", "0.30", "0.60", "badge-medium"),
            ("🔴 High Risk",   "0.60", "1.00", "badge-high"),
        ]:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"background:#1e293b;border-radius:8px;padding:8px 14px;margin-bottom:6px'>"
                f"<span class='badge {cls}'>{label}</span>"
                f"<span style='color:#94a3b8;font-size:.9rem'>{lo} – {hi}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(f"""<div class="card card-warning">
    ⚠️ <b>Academic Prototype Disclaimer</b><br>
    <small>{APP_DISCLAIMER}</small>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CREDIT RISK ASSESSMENT
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Risk Assessment":
    st.title("🔍 Credit Risk Assessment")
    st.markdown("Enter applicant details and click **Assess Credit Risk** to get a prediction with SHAP explanation.")

    if not _model_ready():
        _not_trained_msg()

    artefacts, err = _load_artefacts()
    if err:
        st.error(f"Failed to load model: {err}")
        st.stop()

    # ── Input form ────────────────────────────────────────────────────────
    st.markdown("### 📋 Applicant Information")
    with st.form("assess_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**📊 Credit Utilisation**")
            rev_util = st.number_input("Revolving Utilization", 0.0, 50.0, 0.45, 0.01,
                help="Credit card balance ÷ credit limit. >1 = over-limit.")
            debt_ratio = st.number_input("Debt Ratio", 0.0, 5000.0, 0.30, 0.01,
                help="Monthly debt payments ÷ monthly gross income.")
            monthly_income = st.number_input("Monthly Income (USD)", 0.0, 500000.0, 5000.0, 100.0,
                help="0 = unknown — will be median-imputed.")

        with c2:
            st.markdown("**📅 Delinquency History**")
            past_30_59 = st.number_input("Times 30–59 Days Past Due", 0, 20, 0, 1)
            past_60_89 = st.number_input("Times 60–89 Days Past Due", 0, 20, 0, 1)
            past_90    = st.number_input("Times 90+ Days Past Due",   0, 20, 0, 1)

        with c3:
            st.markdown("**👤 Personal & Credit Profile**")
            age             = st.number_input("Age (years)",                 18, 110, 42, 1)
            open_lines      = st.number_input("Open Credit Lines & Loans",    0, 60,  8,  1)
            real_estate     = st.number_input("Real Estate Loans/Lines",       0, 54,  1,  1)
            dependents      = st.number_input("Number of Dependents",          0, 20,  2,  1)

        submitted = st.form_submit_button("🔍 Assess Credit Risk", use_container_width=True, type="primary")

    if submitted:
        from src.predict import predict_single, validate_applicant_input

        raw = {
            "RevolvingUtilizationOfUnsecuredLines":  rev_util,
            "age":                                   age,
            "NumberOfTime30-59DaysPastDueNotWorse":  past_30_59,
            "DebtRatio":                             debt_ratio,
            "MonthlyIncome":                         monthly_income if monthly_income > 0 else None,
            "NumberOfOpenCreditLinesAndLoans":        open_lines,
            "NumberOfTimes90DaysLate":               past_90,
            "NumberRealEstateLoansOrLines":           real_estate,
            "NumberOfTime60-89DaysPastDueNotWorse":  past_60_89,
            "NumberOfDependents":                    dependents,
        }

        try:
            clean = validate_applicant_input(raw)
        except ValueError as ve:
            st.error(f"**Validation error:**\n{ve}")
            st.stop()

        with st.spinner("Computing risk assessment …"):
            result = predict_single(clean)

        prob     = result["default_probability"]
        category = result["risk_category"]
        pct      = prob * 100

        st.markdown("---")
        st.markdown("## 📊 Assessment Result")

        # Metric row
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Default Probability", f"{pct:.1f}%")
        with mc2:
            badge_map = {"low": "badge-low", "medium": "badge-medium", "high": "badge-high"}
            st.markdown(f"**Risk Category**<br><span class='badge {badge_map[category]}'>{result['risk_label']}</span>",
                        unsafe_allow_html=True)
        with mc3:
            st.metric("Decision Support", result["decision_support"])
        with mc4:
            st.metric("Model", result["model_name"])

        # Probability bar
        bar_colour = {"low": "#22c55e", "medium": "#f59e0b", "high": "#ef4444"}[category]
        st.markdown(f"""
        <div style='background:#1e293b;border-radius:10px;padding:4px;margin:16px 0'>
          <div style='background:{bar_colour};width:{min(pct,100):.1f}%;height:18px;
                      border-radius:8px;transition:width .4s ease'></div>
        </div>
        <div style='display:flex;justify-content:space-between;color:#64748b;font-size:.8rem;margin-top:-10px'>
          <span>0% (No Risk)</span><span>30%</span><span>60%</span><span>100% (Certain Default)</span>
        </div>
        """, unsafe_allow_html=True)

        # ── SHAP local explanation ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🧠 SHAP Feature Explanation")
        st.markdown("""<div class="card card-info">
        SHAP values show how each feature <b>contributed to this model prediction</b>.<br>
        🔴 Positive SHAP = feature pushes predicted risk <b>higher</b><br>
        🔵 Negative SHAP = feature pushes predicted risk <b>lower</b><br>
        <small>SHAP values are in the log-odds (margin) space — signs and ranking are interpretable.</small>
        </div>""", unsafe_allow_html=True)

        try:
            import shap
            from src.feature_engineering import engineer_features, get_all_feature_names
            from src.explainability import build_explainer, explain_single

            pipeline     = artefacts["pipeline"]
            model        = artefacts["model"]
            feature_names = artefacts["feature_names"]
            bg           = _load_shap_background()
            meta_data    = artefacts["metadata"]

            input_df = pd.DataFrame([clean])
            input_df = engineer_features(input_df)
            X_in     = input_df[feature_names]
            X_tf     = pipeline.transform(X_in)

            background = bg if bg is not None else X_tf
            explainer  = build_explainer(model, background, feature_names,
                                         model_name=result["model_name"])
            local_exp  = explain_single(explainer, X_tf, feature_names, top_n=17)

            # Contribution table
            contrib_df = pd.DataFrame(local_exp["contributions"])
            contrib_df.columns = ["Feature", "Value", "SHAP"]
            contrib_df["Direction"] = contrib_df["SHAP"].apply(
                lambda v: "⬆ Increases Risk" if v > 0 else "⬇ Reduces Risk"
            )
            contrib_df["SHAP"] = contrib_df["SHAP"].round(4)
            contrib_df["Value"] = contrib_df["Value"].round(4)

            def _colour_shap(val):
                if val > 0:   return "color:#f87171;font-weight:600"
                elif val < 0: return "color:#60a5fa;font-weight:600"
                return ""

            st.dataframe(
                contrib_df.style.applymap(_colour_shap, subset=["SHAP"])
                                .format({"SHAP": "{:+.4f}", "Value": "{:.3f}"}),
                use_container_width=True, hide_index=True,
            )

            # Top factors side by side
            inc_col, dec_col = st.columns(2)
            with inc_col:
                st.markdown("#### 🔴 Top Risk-Increasing Factors")
                for c in local_exp["top_increasing"][:5]:
                    st.markdown(f"- **{c['feature']}** = `{c['feature_value']:.3f}` &nbsp; SHAP: `+{c['shap_value']:.4f}`")
            with dec_col:
                st.markdown("#### 🔵 Top Risk-Reducing Factors")
                for c in local_exp["top_decreasing"][:5]:
                    st.markdown(f"- **{c['feature']}** = `{c['feature_value']:.3f}` &nbsp; SHAP: `{c['shap_value']:.4f}`")

        except Exception as shap_err:
            st.warning(f"SHAP explanation unavailable: {shap_err}")

        st.markdown("---")
        st.caption(APP_DISCLAIMER)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊  Model Comparison":
    st.title("📊 Model Comparison")
    st.markdown("All four models evaluated on the same held-out test set (29,879 rows). All numbers are from actual experiments.")

    if not _model_ready():
        _not_trained_msg()

    meta = _load_metadata()

    # ── Metrics table ──────────────────────────────────────────────────────
    st.markdown("### 📈 Performance Metrics")
    comp = _load_comparison()
    if comp is not None:
        def _highlight(df):
            best_name = meta.get("best_model_name", "")
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for col in ["Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC"]:
                if col in df.columns:
                    idx = df[col].idxmax()
                    styles.loc[idx, col] = "background-color:#166534;color:#dcfce7;font-weight:700"
            if "Brier Score" in df.columns:
                idx = df["Brier Score"].idxmin()
                styles.loc[idx, "Brier Score"] = "background-color:#166534;color:#dcfce7;font-weight:700"
            for i, row in df.iterrows():
                if row.get("Model","") == best_name:
                    styles.loc[i, "Model"] = "color:#38bdf8;font-weight:700"
            return styles

        float_cols = comp.select_dtypes("float").columns.tolist()
        st.dataframe(
            comp.style.apply(_highlight, axis=None)
                      .format({c: "{:.4f}" for c in float_cols}),
            use_container_width=True, hide_index=True,
        )
        st.caption("Green cells = best value per column. Blue = selected model.")
    else:
        st.info("Run `python -m src.train` to generate comparison results.")

    # ── Best model callout ─────────────────────────────────────────────────
    if meta and meta.get("best_model_name"):
        best = meta["best_model_name"]
        bm   = next((m for m in meta["all_metrics"] if m["model"] == best), {})
        st.markdown(f"""<div class="card card-success">
        ⭐ <b>Selected Model: {best}</b><br>
        ROC-AUC = <b>{bm.get('roc_auc','—')}</b> &nbsp;|&nbsp;
        PR-AUC = <b>{bm.get('pr_auc','—')}</b> &nbsp;|&nbsp;
        Recall = <b>{bm.get('recall','—')}</b> &nbsp;|&nbsp;
        Brier (calibrated) = <b>0.0498</b>
        </div>""", unsafe_allow_html=True)

    # ── Figures ────────────────────────────────────────────────────────────
    st.markdown("---")
    tab_roc, tab_pr, tab_cm = st.tabs(["📉 ROC Curves", "📉 PR Curves", "🔢 Confusion Matrices"])

    with tab_roc:
        p = FIGURES_DIR / "roc_curves_comparison.png"
        if p.exists(): st.image(str(p), use_container_width=True)
        else: st.info("Run training pipeline to generate this plot.")

    with tab_pr:
        p = FIGURES_DIR / "pr_curves_comparison.png"
        if p.exists(): st.image(str(p), use_container_width=True)
        else: st.info("Run training pipeline to generate this plot.")

    with tab_cm:
        slugs  = ["xgboost", "lightgbm", "logistic_regression", "random_forest"]
        labels = ["XGBoost ⭐", "LightGBM", "Logistic Regression", "Random Forest"]
        found  = [(s, l) for s, l in zip(slugs, labels)
                  if (FIGURES_DIR / f"confusion_matrix_{s}.png").exists()]
        if found:
            cols = st.columns(2)
            for i, (slug, label) in enumerate(found):
                with cols[i % 2]:
                    st.markdown(f"**{label}**")
                    st.image(str(FIGURES_DIR / f"confusion_matrix_{slug}.png"),
                             use_container_width=True)
        else:
            st.info("Confusion matrix plots not yet generated.")

        if meta and meta.get("all_metrics"):
            best_name = meta.get("best_model_name","")
            bm = next((m for m in meta["all_metrics"] if m["model"] == best_name), {})
            if bm:
                st.markdown(f"**{best_name} Confusion Matrix Breakdown:**")
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("✅ True Negative",  f"{bm.get('tn',0):,}", help="Correctly predicted no default")
                cc2.metric("⚠️ False Positive", f"{bm.get('fp',0):,}", help="Predicted default, was actually OK")
                cc3.metric("🚨 False Negative", f"{bm.get('fn',0):,}", help="Missed actual default — highest risk")
                cc4.metric("✅ True Positive",  f"{bm.get('tp',0):,}", help="Correctly caught a default")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════════════
elif page == "🧠  Explainability":
    st.title("🧠 Global SHAP Explainability")
    st.markdown("""<div class="card card-info">
    <b>SHAP (SHapley Additive exPlanations)</b> quantifies each feature's contribution
    to the model's predictions across the entire test set.<br>
    <small>SHAP explains <i>how the model responded to a feature</i>,
    not why the feature caused the real-world outcome.</small>
    </div>""", unsafe_allow_html=True)

    if not _model_ready():
        _not_trained_msg()

    meta = _load_metadata()
    model_name = meta.get("best_model_name", "Best Model") if meta else "Best Model"

    tab_imp, tab_bee, tab_cal, tab_fair = st.tabs(
        ["📊 Feature Importance", "🐝 Beeswarm Plot", "📐 Calibration", "⚖️ Fairness"]
    )

    with tab_imp:
        imp_df = _load_shap_importance()
        if imp_df is not None:
            st.markdown(f"### Feature Importance — {model_name} (mean |SHAP|, test set sample n=3,000)")
            fig, ax = plt.subplots(figsize=(9, 6))
            fig.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#1e293b")
            top15 = imp_df.head(15).iloc[::-1]
            colours = ["#38bdf8" if i < 5 else "#64748b" for i in range(len(top15)-1, -1, -1)]
            bars = ax.barh(top15["feature"], top15["mean_abs_shap"], color=colours[::-1],
                           edgecolor="#0f172a", height=0.65)
            ax.set_xlabel("Mean |SHAP Value|", color="#94a3b8")
            ax.tick_params(colors="#94a3b8", labelsize=9)
            ax.grid(axis="x", alpha=0.3, color="#334155")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            for bar in bars:
                ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                        f"{bar.get_width():.3f}", va="center", color="#e2e8f0", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.markdown("**Top Features:**")
            for _, row in imp_df.head(5).iterrows():
                st.markdown(f"- **{row['feature']}** — mean |SHAP| = `{row['mean_abs_shap']:.4f}`")
        else:
            st.info("Run `python -m src.post_train_analysis` to generate SHAP results.")

        p = FIGURES_DIR / f"shap_bar_importance_{model_name.lower().replace(' ','_')}.png"
        if p.exists():
            st.markdown("---")
            st.markdown("*Native SHAP bar plot (saved from analysis run):*")
            st.image(str(p), use_container_width=True)

    with tab_bee:
        p = FIGURES_DIR / f"shap_summary_{model_name.lower().replace(' ','_')}.png"
        if p.exists():
            st.markdown("### SHAP Beeswarm Summary Plot")
            st.markdown("""
            - **Each dot** = one test-set sample
            - **Colour** = feature value (🔴 high, 🔵 low)
            - **X position** = SHAP value (right = increased predicted risk)
            """)
            st.image(str(p), use_container_width=True)
        else:
            st.info("Run `python -m src.post_train_analysis` to generate SHAP summary plot.")

    with tab_cal:
        cal_path = PROJECT_ROOT / "models" / "calibration_results.json"
        if cal_path.exists():
            with open(cal_path) as f:
                cal = json.load(f)
            st.markdown("### Probability Calibration")
            r1, r2, r3 = st.columns(3)
            r1.metric("Raw Brier Score",        f"{cal['brier_raw']:.4f}")
            r2.metric("Calibrated Brier Score",  f"{cal['brier_calibrated']:.4f}",
                      delta=f"{cal['improvement']:+.4f}", delta_color="inverse")
            r3.metric("Calibration Applied",     "✅ Yes" if cal.get("calibration_applied") else "❌ No")

            st.markdown(f"""<div class="card card-success">
            📐 <b>Conclusion:</b> {cal['conclusion']}
            </div>""", unsafe_allow_html=True)

            p = FIGURES_DIR / "calibration_comparison.png"
            if p.exists():
                st.image(str(p), use_container_width=True)
        else:
            st.info("Run `python -m src.post_train_analysis` to generate calibration results.")

    with tab_fair:
        st.markdown("### Fairness Analysis")
        st.markdown("""<div class="card card-warning">
        ⚠️ <b>Disclaimer:</b> The metrics below are model-level statistical observations.
        They do NOT prove the model is fair or unfair. Age and dependents are proxy attributes
        — not directly measured protected characteristics. Disparities may reflect historical
        data patterns rather than model error.
        </div>""", unsafe_allow_html=True)

        for attr, label in [("age", "Age Group"), ("dependents", "Dependents Group")]:
            p = PROJECT_ROOT / "reports" / "results" / f"fairness_{attr}.csv"
            if p.exists():
                df_f = pd.read_csv(p)
                st.markdown(f"#### By {label}")
                st.dataframe(
                    df_f.style.format({c: "{:.4f}" for c in df_f.select_dtypes("float").columns}),
                    use_container_width=True, hide_index=True,
                )
                fig_p = FIGURES_DIR / f"fairness_{attr}_group.png"
                if fig_p.exists():
                    st.image(str(fig_p), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — WHAT-IF SIMULATOR
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔧  What-If Simulator":
    st.title("🔧 What-If Simulator")
    st.markdown("""<div class="card card-warning">
    ⚠️ <b>Disclaimer:</b> This simulator shows how the <i>model</i> responds to different
    input values. It does <b>NOT</b> predict how a real applicant's default risk would change
    if they altered their financial behaviour. This is a model-sensitivity demonstration only.
    </div>""", unsafe_allow_html=True)

    if not _model_ready():
        _not_trained_msg()

    from src.predict import predict_single, validate_applicant_input

    def _predict_safe(d):
        try:
            clean = validate_applicant_input(d)
            return predict_single(clean)
        except Exception as e:
            return None

    # ── Base profile ───────────────────────────────────────────────────────
    st.markdown("### 👤 Base Applicant Profile")
    b1, b2, b3 = st.columns(3)
    with b1:
        b_util    = st.slider("Revolving Utilization",  0.0, 3.0, 0.45, 0.01, key="b_util")
        b_income  = st.slider("Monthly Income ($)",     0,   20000, 5000, 100,  key="b_inc")
    with b2:
        b_debt    = st.slider("Debt Ratio",             0.0, 5.0, 0.30,  0.01, key="b_debt")
        b_30_59   = st.slider("Times 30–59 Days PD",   0,   10,  0,     1,    key="b_d30")
    with b3:
        b_90      = st.slider("Times 90+ Days PD",      0,   10,  0,     1,    key="b_d90")
        b_age     = st.slider("Age",                    18,  90,  42,    1,    key="b_age")
        b_dep     = st.slider("Dependents",             0,   10,  2,     1,    key="b_dep")

    def _mk(util, inc, debt, d30, d60, d90, age, dep):
        return {"RevolvingUtilizationOfUnsecuredLines": util, "age": age,
                "NumberOfTime30-59DaysPastDueNotWorse": d30, "DebtRatio": debt,
                "MonthlyIncome": inc if inc > 0 else None,
                "NumberOfOpenCreditLinesAndLoans": 8, "NumberOfTimes90DaysLate": d90,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfTime60-89DaysPastDueNotWorse": d60, "NumberOfDependents": dep}

    base_res = _predict_safe(_mk(b_util, b_income, b_debt, b_30_59, 0, b_90, b_age, b_dep))

    st.markdown("---")
    st.markdown("### ✏️ Modified Profile")
    st.caption("Adjust any value to see how the prediction changes.")
    m1, m2, m3 = st.columns(3)
    with m1:
        m_util   = st.slider("Revolving Utilization (mod)", 0.0, 3.0,  b_util,   0.01, key="m_util")
        m_income = st.slider("Monthly Income $ (mod)",      0,   20000, b_income, 100,  key="m_inc")
    with m2:
        m_debt   = st.slider("Debt Ratio (mod)",            0.0, 5.0,  b_debt,   0.01, key="m_debt")
        m_30_59  = st.slider("Times 30–59 Days PD (mod)",  0,   10,   b_30_59,  1,    key="m_d30")
    with m3:
        m_90     = st.slider("Times 90+ Days PD (mod)",     0,   10,   b_90,     1,    key="m_d90")
        m_age    = st.slider("Age (mod)",                   18,  90,   b_age,    1,    key="m_age")

    mod_res = _predict_safe(_mk(m_util, m_income, m_debt, m_30_59, 0, m_90, m_age, b_dep))

    if base_res and mod_res:
        st.markdown("---")
        st.markdown("### 📊 Comparison")
        bp, mp = base_res["default_probability"], mod_res["default_probability"]
        delta  = (mp - bp) * 100

        r1, r2, r3 = st.columns(3)
        r1.metric("Base Probability",     f"{bp*100:.1f}%")
        r2.metric("Modified Probability", f"{mp*100:.1f}%",
                  delta=f"{delta:+.1f}%", delta_color="inverse")
        r3.metric("Risk Change", "→ " + RISK_LABELS[mod_res["risk_category"]])

        if base_res["risk_category"] != mod_res["risk_category"]:
            st.warning(f"⚠️ Risk category changed: **{RISK_LABELS[base_res['risk_category']]}** → **{RISK_LABELS[mod_res['risk_category']]}**")
        else:
            st.success(f"Risk category unchanged: **{RISK_LABELS[base_res['risk_category']]}**")

        # Side-by-side probability bars
        fig, ax = plt.subplots(figsize=(7, 2.5))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        colours = ["#3b82f6", "#ef4444" if mp > bp else "#22c55e"]
        y_labels = ["Base Profile", "Modified Profile"]
        bars = ax.barh(y_labels, [bp, mp], color=colours, height=0.45, edgecolor="#0f172a")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Predicted Default Probability", color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        ax.axvline(RISK_THRESHOLDS["low_max"],    color="#f59e0b", ls="--", lw=1.5, alpha=.8)
        ax.axvline(RISK_THRESHOLDS["medium_max"], color="#ef4444", ls="--", lw=1.5, alpha=.8)
        for bar in bars:
            ax.text(bar.get_width()+.01, bar.get_y()+bar.get_height()/2,
                    f"{bar.get_width()*100:.1f}%", va="center", color="#f1f5f9", fontsize=11)
        ax.grid(axis="x", alpha=0.3, color="#334155")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — METHODOLOGY
# ════════════════════════════════════════════════════════════════════════════
elif page == "📖  Methodology":
    st.title("📖 Methodology & About")
    st.markdown("Academic context and technical documentation for CreditWise.")

    tabs = st.tabs(["📚 Dataset", "⚙️ Preprocessing", "🤖 Models",
                    "📊 Evaluation", "🧠 SHAP", "📐 Calibration",
                    "⚖️ Fairness", "⚠️ Limitations"])

    with tabs[0]:
        st.markdown("""
        **Give Me Some Credit** — Kaggle Competition Dataset

        | Property | Value |
        |---|---|
        | Source | [kaggle.com/c/GiveMeSomeCredit](https://www.kaggle.com/c/GiveMeSomeCredit/data) |
        | Raw rows | 150,000 |
        | After cleaning | 149,391 |
        | Target | `SeriousDlqin2yrs` (0 = no default, 1 = default within 2 years) |
        | Class 0 | 93.3% |
        | Class 1 | 6.7% |
        | Imbalance ratio | 13.96 : 1 |
        | Missing — MonthlyIncome | 19.82% |
        | Missing — NumberOfDependents | 2.62% |
        """)

    with tabs[1]:
        st.markdown("""
        **Pipeline (fit on training data only — no test-set leakage):**

        1. Drop 609 duplicate rows
        2. Replace delinquency codes 96/98 with NaN
        3. Engineer 7 derived features
        4. Stratified 80/20 train-test split (seed=42)
        5. Median imputation (training medians applied to test)
        6. StandardScaler (for Logistic Regression)

        **Engineered Features:**

        | Feature | Justification |
        |---|---|
        | log_revolving_utilization | Log-compresses right-skewed utilization |
        | log_debt_ratio | 16.3% of values > 100 — log reduces outlier impact |
        | log_monthly_income | Income is log-normally distributed |
        | total_past_due | Cumulative delinquency across all severity buckets |
        | has_past_due | Binary: any delinquency history |
        | income_per_dependent | Affordability per household member |
        | credit_line_density | Credit lines ÷ effective credit age |
        """)

    with tabs[2]:
        st.markdown("""
        | Model | Type | Imbalance Strategy |
        |---|---|---|
        | Logistic Regression | Linear baseline | class_weight='balanced' |
        | Random Forest | Ensemble | class_weight='balanced' |
        | XGBoost ⭐ | Gradient boosting | scale_pos_weight=10 |
        | LightGBM | Gradient boosting | class_weight='balanced' |

        **Selection criterion:** Highest ROC-AUC on held-out test set.
        """)

    with tabs[3]:
        st.markdown("""
        | Metric | Purpose |
        |---|---|
        | ROC-AUC | Primary discrimination metric — threshold-independent |
        | PR-AUC | Especially informative under class imbalance |
        | Recall | Critical: measures defaults correctly identified |
        | F1 | Balances precision and recall |
        | Brier Score | Probability calibration quality |

        **False Negatives are the highest-risk errors** — a missed default means the
        model predicted low risk for a borrower who actually defaulted.
        """)

    with tabs[4]:
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** — based on cooperative game theory.

        - **TreeExplainer** for Random Forest, XGBoost, LightGBM (exact, fast)
        - **LinearExplainer** for Logistic Regression (exact)
        - SHAP values in **log-odds (margin) space** for tree models

        **Global SHAP:** mean |SHAP| across 3,000 test-set samples

        **Local SHAP:** signed per-feature contributions for one applicant

        > SHAP explains the *model's* decision. It does NOT establish causation
        > between a feature and real-world default behaviour.
        """)

    with tabs[5]:
        st.markdown("""
        **Platt scaling (sigmoid calibration)** applied post-hoc to the fitted XGBoost model.

        | | Raw | Calibrated |
        |---|---|---|
        | Brier Score | 0.1134 | **0.0498** |
        | Improvement | — | **0.0636 ✅** |

        Calibrated model is used in the Risk Assessment page.
        """)

    with tabs[6]:
        st.markdown("""
        Group-level performance analysis. Attributes: **age brackets**, **dependents**.

        | Age Group | Recall | FNR | FPR | ROC-AUC |
        |---|---|---|---|---|
        | Under 30 | 0.821 | 0.179 | 0.307 | 0.840 |
        | 30–60 | 0.716 | 0.284 | 0.174 | 0.848 |
        | Over 60 | 0.565 | 0.436 | 0.056 | 0.846 |

        **Observations:**
        - Older borrowers show lower recall (model misses more of their defaults)
        - Younger borrowers show higher FPR (more false alarms)
        - ROC-AUC is consistent across groups (~0.84)

        **Limitation:** Age and dependents are proxy attributes. Disparities ≠ discrimination.
        """)

    with tabs[7]:
        st.markdown("""
        1. Performance is specific to Give Me Some Credit — may not generalise
        2. Historical data may encode existing socioeconomic disparities
        3. Model does not consider macroeconomic conditions or loan terms
        4. SHAP explains model behaviour, not real-world causation
        5. **NOT suitable for real-world lending decisions**
        6. Fairness analysis limited to available proxy attributes
        7. Imbalance ratio (13.96:1) limits precision even with class weighting
        """)

    st.markdown("---")
    st.markdown(f"""<div class="card card-warning">
    ⚠️ <b>Academic Prototype Disclaimer</b><br><small>{APP_DISCLAIMER}</small>
    </div>""", unsafe_allow_html=True)
