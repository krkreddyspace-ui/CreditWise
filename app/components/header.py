"""
CreditWise — Header Component
==============================
Renders page headers, hero banners, and breadcrumbs in Superdesign style.
"""

import streamlit as st


def render_page_header(title: str, subtitle: str, tag: str = "EXPLAINABLE AI FRAMEWORK") -> None:
    """Renders a page hero banner with superdesign dark typography."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #1f2937;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; color: #3b82f6; text-transform: uppercase; margin-bottom: 0.25rem;">
                {tag}
            </div>
            <h1 style="font-size: 1.85rem; font-weight: 800; color: #f8fafc; margin: 0 0 0.4rem 0; letter-spacing: -0.02em;">
                {title}
            </h1>
            <div style="font-size: 0.95rem; color: #94a3b8; font-weight: 400; max-width: 800px; line-height: 1.5;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
