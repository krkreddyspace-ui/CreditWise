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
    """Creates a dark Plotly confusion matrix heatmap."""
    tn, fp, fn, tp = cm.ravel()
    annotations = [
        [f"True Neg (TN)<br><b>{tn:,}</b>", f"False Pos (FP)<br><b>{fp:,}</b>"],
        [f"False Neg (FN)<br><b>{fn:,}</b>", f"True Pos (TP)<br><b>{tp:,}</b>"]
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted Non-Default", "Predicted Default"],
            y=["Actual Non-Default", "Actual Default"],
            colorscale="Blues",
            showscale=False,
            text=annotations,
            texttemplate="%{text}",
            textfont=dict(size=13, color="#ffffff")
        )
    )
    
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=f"Confusion Matrix — {model_name}", font=dict(size=14, color="#f8fafc")),
        height=350,
    )
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
