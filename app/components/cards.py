"""
CreditWise — Cards & Metric Containers Component
=================================================
Renders styled cards, summary stats, decision banners, and pipeline flows.
"""

import streamlit as st
from typing import Dict, Any, List


def render_metric_card(label: str, value: str, subtext: str = "", border_color: str = "#1f2937") -> None:
    """Renders a single metric card container."""
    st.markdown(
        f"""
        <div style="background-color: #111827; border: 1px solid {border_color}; border-radius: 12px; padding: 1.1rem 1.25rem; height: 100%;">
            <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 0.4rem;">
                {label}
            </div>
            <div style="font-size: 1.75rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em; line-height: 1.2;">
                {value}
            </div>
            {f'<div style="font-size: 0.78rem; color: #64748b; margin-top: 0.35rem;">{subtext}</div>' if subtext else ''}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_pipeline_flow() -> None:
    """Renders the 7-stage ML pipeline visual flowchart in Superdesign style."""
    st.markdown(
        """
        <div style="background-color: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem;">
            <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #60a5fa; margin-bottom: 0.8rem;">
                End-to-End Decision Support Pipeline
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.4rem;">
                <div style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.82rem; color: #f8fafc; font-weight: 600;">
                    <span style="background: #2563eb; width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.7rem;">1</span> Data Ingestion
                </div>
                <span style="color: #475569;">➔</span>
                <div style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.82rem; color: #f8fafc; font-weight: 600;">
                    <span style="background: #2563eb; width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.7rem;">2</span> Preprocessing
                </div>
                <span style="color: #475569;">➔</span>
                <div style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.82rem; color: #f8fafc; font-weight: 600;">
                    <span style="background: #2563eb; width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.7rem;">3</span> Feature Eng.
                </div>
                <span style="color: #475569;">➔</span>
                <div style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.82rem; color: #f8fafc; font-weight: 600;">
                    <span style="background: #2563eb; width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.7rem;">4</span> Imbalance ML
                </div>
                <span style="color: #475569;">➔</span>
                <div style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.82rem; color: #f8fafc; font-weight: 600;">
                    <span style="background: #2563eb; width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.7rem;">5</span> Calibration
                </div>
                <span style="color: #475569;">➔</span>
                <div style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.82rem; color: #f8fafc; font-weight: 600;">
                    <span style="background: #2563eb; width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.7rem;">6</span> SHAP Explain
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_decision_card(prob_pct: float, risk_category: str, recommendation: str, decision_support: str) -> None:
    """Renders the risk assessment prediction result card."""
    category_colors = {
        "LOW": {"bg": "rgba(16, 185, 129, 0.12)", "border": "#059669", "text": "#34d399", "badge": "cw-badge-low"},
        "MEDIUM": {"bg": "rgba(245, 158, 11, 0.12)", "border": "#d97706", "text": "#fbbf24", "badge": "cw-badge-medium"},
        "HIGH": {"bg": "rgba(239, 68, 68, 0.12)", "border": "#dc2626", "text": "#f87171", "badge": "cw-badge-high"}
    }
    style = category_colors.get(risk_category.upper(), category_colors["MEDIUM"])

    st.markdown(
        f"""
        <div style="background-color: {style['bg']}; border: 1.5px solid {style['border']}; border-radius: 14px; padding: 1.5rem; margin-top: 1rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
                <span style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8;">
                    ESTIMATED DEFAULT PROBABILITY
                </span>
                <span class="cw-badge {style['badge']}">
                    {risk_category} RISK
                </span>
            </div>
            <div style="font-size: 3rem; font-weight: 900; color: {style['text']}; letter-spacing: -0.03em; line-height: 1;">
                {prob_pct:.1f}%
            </div>
            <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.2rem;">
                    Decision Support: {recommendation}
                </div>
                <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.4;">
                    {decision_support}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
