"""
CreditWise — Local SHAP Explanation Component
===============================================
Renders applicant-level SHAP breakdown, directional impact indicators,
and key contributing factors.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List


def render_local_shap_explanation(shap_dict: Dict[str, float], raw_features: Dict[str, Any]) -> None:
    """Renders local feature contribution table with visual direction indicators."""
    if not shap_dict:
        st.info("Local SHAP values unavailable for this prediction.")
        return

    # Sort by absolute contribution
    sorted_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    rows = []
    for feat, val in sorted_items[:8]:
        direction = "⬆️ Increases Risk" if val > 0 else "⬇️ Decreases Risk"
        badge_style = "color: #f87171; font-weight: 600;" if val > 0 else "color: #34d399; font-weight: 600;"
        user_val = raw_features.get(feat, "N/A")

        rows.append(
            f"""
            <tr style="border-bottom: 1px solid #1f2937;">
                <td style="padding: 0.6rem; color: #f8fafc; font-weight: 600;">{feat}</td>
                <td style="padding: 0.6rem; color: #cbd5e1;">{user_val}</td>
                <td style="padding: 0.6rem; color: {'#f87171' if val > 0 else '#34d399'}; font-weight: 700;">{val:+.4f}</td>
                <td style="padding: 0.6rem; {badge_style}">{direction}</td>
            </tr>
            """
        )

    table_html = f"""
    <div style="background-color: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 1.25rem; margin-top: 1rem;">
        <div style="font-size: 0.85rem; font-weight: 700; color: #60a5fa; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.8rem;">
            Individual Applicant SHAP Contribution Breakdown
        </div>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
            <thead>
                <tr style="border-bottom: 2px solid #374151; color: #94a3b8; text-align: left;">
                    <th style="padding: 0.5rem;">Feature</th>
                    <th style="padding: 0.5rem;">Applicant Value</th>
                    <th style="padding: 0.5rem;">SHAP Value</th>
                    <th style="padding: 0.5rem;">Directional Impact</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        <div style="margin-top: 0.8rem; font-size: 0.75rem; color: #64748b;">
            💡 Positive SHAP values push the prediction toward Higher Default Risk; negative values push toward Lower Risk.
        </div>
    </div>
    """

    st.markdown(table_html, unsafe_allow_html=True)
