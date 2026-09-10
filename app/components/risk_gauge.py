"""
CreditWise — Risk Gauge Component
==================================
Renders an interactive horizontal risk probability gauge (0% - 100%)
with color bands (Low: 0-30%, Medium: 30-60%, High: 60-100%) and marker pin.
"""

import streamlit as st


def render_risk_gauge(prob: float, prob_pct: float) -> None:
    """Renders the HTML/CSS horizontal risk probability gauge bar."""
    # Constrain probability between 0 and 1
    clamped_prob = max(0.0, min(1.0, float(prob)))
    position_pct = clamped_prob * 100.0

    st.markdown(
        f"""
        <div style="background-color: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 1.25rem; margin-top: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                <span style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8;">
                    Risk Spectrum Gauge
                </span>
                <span style="font-size: 0.85rem; font-weight: 700; color: #f8fafc;">
                    {prob_pct:.1f}% Default Risk
                </span>
            </div>
            
            <div style="position: relative; width: 100%; height: 16px; border-radius: 8px; background: linear-gradient(to right, #10b981 0%, #10b981 30%, #f59e0b 30%, #f59e0b 60%, #ef4444 60%, #ef4444 100%); margin: 1.2rem 0 0.6rem 0; box-shadow: inset 0 2px 4px rgba(0,0,0,0.6);">
                <div style="position: absolute; top: -7px; left: {position_pct:.2f}%; width: 28px; height: 28px; background-color: #ffffff; border: 3.5px solid #0b0f19; border-radius: 50%; transform: translateX(-50%); box-shadow: 0 0 12px rgba(255,255,255,0.8); z-index: 10;">
                </div>
            </div>

            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 600; color: #64748b;">
                <span style="color: #34d399;">🟢 LOW (0.0% – 30.0%)</span>
                <span style="color: #fbbf24;">🟡 MEDIUM (30.0% – 60.0%)</span>
                <span style="color: #f87171;">🔴 HIGH (60.0% – 100.0%)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
