"""
CreditWise — Plotly Dark Theme Charts Component
================================================
Renders dark enterprise Plotly charts for ROC-AUC curves, PR-AUC curves,
Model Metrics comparisons, and SHAP feature importance.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any


DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#111827",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    margin=dict(l=40, r=40, t=40, b=40),
    xaxis=dict(gridcolor="#1f2937", zerolinecolor="#374151"),
    yaxis=dict(gridcolor="#1f2937", zerolinecolor="#374151"),
)


def create_model_comparison_bar_chart(df_results: pd.DataFrame, metric: str = "ROC-AUC") -> go.Figure:
    """Creates a dark Plotly bar chart comparing model performance."""
    df_sorted = df_results.sort_values(by=metric, ascending=True)
    
    colors = ["#2563eb" if m != df_sorted[metric].max() else "#06b6d4" for m in df_sorted[metric]]

    fig = go.Figure(
        go.Bar(
            x=df_sorted[metric],
            y=df_sorted["Model"],
            orientation="h",
            marker=dict(color=colors, cornerradius=6),
            text=df_sorted[metric].apply(lambda x: f"{x:.4f}"),
            textposition="outside"
        )
    )
    
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=f"Model Comparison — {metric}", font=dict(size=14, color="#f8fafc")),
        xaxis_title=metric,
        yaxis_title="",
        height=320,
    )
    return fig


def create_confusion_matrix_heatmap(cm: np.ndarray, model_name: str = "XGBoost") -> go.Figure:
    """Creates a dark Plotly confusion matrix heatmap with high-contrast color palette."""
    tn, fp, fn, tp = cm.ravel()
    total = float(np.sum(cm)) if np.sum(cm) > 0 else 1.0

    annotations = [
        [
            f"<b>True Neg (TN)</b><br><span style='font-size:1.15rem; color:#f8fafc;'>{tn:,}</span><br><span style='font-size:0.75rem; color:#94a3b8;'>({tn/total*100:.1f}%)</span>",
            f"<b>False Pos (FP)</b><br><span style='font-size:1.15rem; color:#f8fafc;'>{fp:,}</span><br><span style='font-size:0.75rem; color:#94a3b8;'>({fp/total*100:.1f}%)</span>"
        ],
        [
            f"<b>False Neg (FN)</b><br><span style='font-size:1.15rem; color:#f8fafc;'>{fn:,}</span><br><span style='font-size:0.75rem; color:#94a3b8;'>({fn/total*100:.1f}%)</span>",
            f"<b>True Pos (TP)</b><br><span style='font-size:1.15rem; color:#f8fafc;'>{tp:,}</span><br><span style='font-size:0.75rem; color:#94a3b8;'>({tp/total*100:.1f}%)</span>"
        ]
    ]

    # Log-transformed intensity to prevent extreme class imbalance from washing out minority cells
    z_log = np.log1p(cm)

    # Custom high-contrast dark fintech colorscale (Slate -> Navy -> Royal Blue -> Azure)
    dark_fintech_colorscale = [
        [0.0, "#1e293b"],   # Slate dark (low count)
        [0.35, "#1e3a8a"],  # Deep blue
        [0.70, "#1d4ed8"],  # Medium royal blue
        [1.0, "#3b82f6"]    # Bright blue (high count)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_log,
            x=["Predicted Non-Default", "Predicted Default"],
            y=["Actual Non-Default", "Actual Default"],
            colorscale=dark_fintech_colorscale,
            showscale=False,
            text=annotations,
            texttemplate="%{text}",
            textfont=dict(family="Inter, sans-serif", size=12, color="#ffffff"),
            xgap=6,
            ygap=6,
            hoverinfo="none"
        )
    )
    
    layout_opts = dict(DARK_LAYOUT)
    layout_opts.update(
        title=dict(text=f"Confusion Matrix — {model_name}", font=dict(size=14, color="#f8fafc")),
        height=350,
        xaxis=dict(
            title=dict(text="Predicted Label", font=dict(size=12, color="#cbd5e1")),
            tickfont=dict(size=11, color="#e2e8f0"),
            gridcolor="rgba(0,0,0,0)",
            side="bottom"
        ),
        yaxis=dict(
            title=dict(text="Actual Label", font=dict(size=12, color="#cbd5e1")),
            tickfont=dict(size=11, color="#e2e8f0"),
            gridcolor="rgba(0,0,0,0)",
            autorange="reversed"  # Places Actual Non-Default at top, Actual Default at bottom
        )
    )
    fig.update_layout(**layout_opts)
    return fig


def create_shap_summary_bar_chart(shap_df: pd.DataFrame) -> go.Figure:
    """Creates a horizontal bar chart of top SHAP global feature importances."""
    df_sorted = shap_df.head(10).sort_values(by="mean_abs_shap", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df_sorted["mean_abs_shap"],
            y=df_sorted["feature"],
            orientation="h",
            marker=dict(
                color=df_sorted["mean_abs_shap"],
                colorscale="Viridis",
                showscale=False,
                cornerradius=6
            ),
            text=df_sorted["mean_abs_shap"].apply(lambda x: f"{x:.3f}"),
            textposition="outside"
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Top 10 Global SHAP Feature Importance (|SHAP|)", font=dict(size=14, color="#f8fafc")),
        xaxis_title="Mean |SHAP Value| (Impact on Default Risk)",
        yaxis_title="",
        height=380,
    )
    return fig
