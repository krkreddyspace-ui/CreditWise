"""
CreditWise — Superdesign Sidebar Component
============================================
Renders brand header, navigation menu, system status badges, and academic disclaimer.
"""

import streamlit as st
from typing import Dict, Any


def render_sidebar(model_metadata: Dict[str, Any]) -> str:
    """Renders the left sidebar navigation shell."""
    with st.sidebar:
        # Brand & Logo Header
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1.25rem 0; border-bottom: 1px solid #1f2937; margin-bottom: 1.25rem;">
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <div style="background: linear-gradient(135deg, #2563eb, #06b6d4); width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 800; color: #fff; font-size: 1.1rem; box-shadow: 0 4px 10px rgba(37, 99, 235, 0.4);">
                        CW
                    </div>
                    <div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em; line-height: 1.1;">
                            CreditWise
                        </div>
                        <div style="font-size: 0.65rem; font-weight: 600; color: #60a5fa; letter-spacing: 0.08em; text-transform: uppercase;">
                            AI CREDIT INTELLIGENCE
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Navigation Options
        page = st.radio(
            "NAVIGATION",
            options=[
                "📊 Overview",
                "🎯 Risk Assessment",
                "⚡ Model Intelligence",
                "🔍 Explainability & SHAP",
                "🧪 What-If Simulator",
                "📘 Methodology & Ethics"
            ],
            index=0,
            label_visibility="visible"
        )

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        # System Status Badges
        best_model_name = model_metadata.get("best_model_name", "XGBoost (Calibrated)")
        n_test = model_metadata.get("test_set_size", 29879)

        st.markdown(
            f"""
            <div style="background-color: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 0.85rem; margin-bottom: 1rem;">
                <div style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 0.5rem;">
                    SYSTEM STATUS
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.4rem;">
                    <span style="font-size: 0.78rem; color: #e2e8f0;">Active Model</span>
                    <span style="font-size: 0.72rem; background: rgba(37, 99, 235, 0.2); color: #60a5fa; padding: 2px 6px; border-radius: 4px; font-weight: 600;">{best_model_name}</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.4rem;">
                    <span style="font-size: 0.78rem; color: #e2e8f0;">Dataset</span>
                    <span style="font-size: 0.72rem; color: #34d399; font-weight: 600;">GiveMeSomeCredit</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 0.78rem; color: #e2e8f0;">Test Evaluation</span>
                    <span style="font-size: 0.72rem; color: #cbd5e1; font-weight: 600;">{n_test:,} samples</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Academic Disclaimer Footer
        st.markdown(
            """
            <div style="padding-top: 0.75rem; border-top: 1px solid #1f2937; font-size: 0.7rem; color: #64748b; line-height: 1.4;">
                <strong style="color: #94a3b8;">Academic Prototype Notice</strong><br/>
                CreditWise is an explainable decision-support prototype. Estimates do not constitute financial advice or automated lending decisions.
            </div>
            """,
            unsafe_allow_html=True
        )

        return page
