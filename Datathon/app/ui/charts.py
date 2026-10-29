"""
Standardized Enterprise Plotly Charts.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

CHART_FONT = dict(family="Inter, sans-serif", size=12, color="#475569")
CHART_MARGINS = dict(l=16, r=16, t=20, b=20)
CHART_CONFIG = {"displayModeBar": False, "responsive": True}

def render_probability_distribution(df: pd.DataFrame):
    fig = px.histogram(
        df,
        x="Conversion Probability",
        nbins=24,
        color="Intent Tier",
        color_discrete_map={
            "High Intent": "#dc2626",
            "Medium Intent": "#f59e0b",
            "Low Intent": "#94a3b8"
        },
        opacity=0.85
    )
    fig.update_layout(
        font=CHART_FONT,
        margin=CHART_MARGINS,
        height=280,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Conversion Likelihood", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="Sessions", showgrid=True, gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def render_intent_donut(df: pd.DataFrame):
    counts = df["Intent Tier"].value_counts().reset_index()
    counts.columns = ["Intent Tier", "Count"]
    fig = px.pie(
        counts,
        names="Intent Tier",
        values="Count",
        color="Intent Tier",
        color_discrete_map={
            "High Intent": "#dc2626",
            "Medium Intent": "#f59e0b",
            "Low Intent": "#cbd5e1"
        },
        hole=0.58
    )
    fig.update_layout(
        font=CHART_FONT,
        margin=CHART_MARGINS,
        height=280,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1)
    )
    return fig

def render_inactivity_scatter(df: pd.DataFrame):
    fig = px.scatter(
        df,
        x="Recency",
        y="Conversion Probability",
        color="Intent Tier",
        color_discrete_map={
            "High Intent": "#dc2626",
            "Medium Intent": "#f59e0b",
            "Low Intent": "#94a3b8"
        },
        opacity=0.75,
        labels={"Recency": "Days Inactive (Recency)", "Conversion Probability": "Probability"}
    )
    try:
        valid_mask = df["Recency"].notnull() & df["Conversion Probability"].notnull()
        if valid_mask.sum() > 3:
            x_clean = df.loc[valid_mask, "Recency"].values
            y_clean = df.loc[valid_mask, "Conversion Probability"].values
            poly_coefs = np.polyfit(x_clean, y_clean, 2)
            poly_fn = np.poly1d(poly_coefs)
            x_grid = np.linspace(x_clean.min(), x_clean.max(), 100)
            y_grid = np.clip(poly_fn(x_grid), 0.0, 1.0)
            fig.add_scatter(
                x=x_grid,
                y=y_grid,
                mode="lines",
                name="Cohort Trend Curve",
                line=dict(color="#0f172a", width=2, dash="dot")
            )
    except Exception:
        pass

    fig.update_layout(
        font=CHART_FONT,
        margin=CHART_MARGINS,
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def render_spend_scatter(df: pd.DataFrame):
    fig = px.scatter(
        df,
        x="Monetary_log",
        y="DistinctProducts_log",
        size="Conversion Probability",
        color="Intent Tier",
        color_discrete_map={
            "High Intent": "#dc2626",
            "Medium Intent": "#f59e0b",
            "Low Intent": "#94a3b8"
        },
        labels={"Monetary_log": "Lifetime Spend Index (log)", "DistinctProducts_log": "Catalog Variety (log)"}
    )
    fig.update_layout(
        font=CHART_FONT,
        margin=CHART_MARGINS,
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def render_confusion_matrix_heatmap(cm: dict):
    fig = px.imshow(
        [[cm['TN'], cm['FP']], [cm['FN'], cm['TP']]],
        labels=dict(x="Predicted Classification (0.50)", y="Actual Observed Outcome", color="Sessions"),
        x=["Negative (No Purchase)", "Positive (Purchase)"],
        y=["Negative (No Purchase)", "Positive (Purchase)"],
        text_auto=True,
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        font=CHART_FONT,
        margin=CHART_MARGINS,
        height=300,
        coloraxis_showscale=False
    )
    return fig

def render_sensitivity_curve(th_df: pd.DataFrame, selected_cutoff: float):
    fig = px.line(
        th_df,
        x="Threshold",
        y=["Precision", "Recall", "F1-Score"],
        markers=True,
        color_discrete_sequence=["#2563eb", "#10b981", "#f59e0b"]
    )
    fig.add_vline(x=selected_cutoff, line_dash="dash", line_color="#0f172a", annotation_text=f"Selected: {selected_cutoff:.2f}")
    fig.update_layout(
        font=CHART_FONT,
        margin=CHART_MARGINS,
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
