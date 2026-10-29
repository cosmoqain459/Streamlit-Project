"""
Retail Purchase & Conversion Intelligence Platform
Enterprise Decision Platform for Session Scoring, Behavioral Analytics,
Outcome Evaluation, Model Governance, and Grounded Multi-Turn Conversational Copilot.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore", message=".*use_container_width.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Load UI Components & Styles ──
from ui.styles import inject_custom_css
from ui.header import render_top_header
from ui.sidebar import render_sidebar
from ui.components import (
    render_kpi_card, render_insight_strip, render_session_detail_panel
)
from ui.charts import (
    render_probability_distribution, render_intent_donut,
    render_inactivity_scatter, render_spend_scatter,
    render_confusion_matrix_heatmap, render_sensitivity_curve
)

# ── Prediction, Evaluation, Analytics Engines ──
from predict import load_metrics, load_schema, predict_batch
from analytics_engine import (
    compute_executive_kpis, compute_behavior_profiles,
    compute_geographic_breakdown, assign_business_segments,
    generate_structured_facts
)
from evaluation_engine import (
    is_evaluation_dataset, predict_evaluation_dataset,
    build_evaluation_facts, calculate_threshold_sensitivity
)
from ai_insights import synthesize_ai_insights
from rag_copilot import RAGExecutiveCopilot

# ── Page Configuration ──
st.set_page_config(
    page_title="Retail Intelligence | Enterprise Decision Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Inject Modern SaaS Design Tokens & Chrome Removal ──
inject_custom_css()

# ── Load Model Governance Metadata ──
try:
    metrics = load_metrics()
    schema = load_schema()
except Exception as e:
    st.error(f"Error loading model artifacts: {str(e)}")
    st.stop()

# Track active navigation state if updated programmatically
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Dashboard"

# ── Render Section-Grouped Sidebar ──
current_is_eval = st.session_state.get("is_eval_active", False)
selected_view, input_df = render_sidebar(metrics, current_is_eval)

if selected_view != st.session_state.active_nav:
    st.session_state.active_nav = selected_view

# Handle Missing Data / Empty State
if input_df is None:
    st.markdown("""
    <div class="chart-card" style="text-align:center; padding: 48px 24px;">
        <div style="font-size:2rem; margin-bottom:12px;">◈</div>
        <div style="font-size:1.15rem; font-weight:700; color:#0f172a; margin-bottom:6px;">No Session Dataset Loaded</div>
        <div style="font-size:0.85rem; color:#64748b; max-width:440px; margin:0 auto 20px auto;">
            Select a sample dataset from the left sidebar or upload an e-commerce session CSV to generate purchase conversion intelligence.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Dual-Mode Inference & Data Preparation ──
is_eval = is_evaluation_dataset(input_df)
st.session_state.is_eval_active = is_eval

try:
    if is_eval:
        predictions_df, eval_summary = predict_evaluation_dataset(input_df)
        eval_facts = build_evaluation_facts(predictions_df, eval_summary)
        notifications = eval_summary.get("notifications", [])
    else:
        predictions_df, notifications = predict_batch(input_df)
        eval_summary = None
        eval_facts = None
except Exception as e:
    st.error(f"Inference Pipeline Error: {str(e)}")
    st.stop()

# Deterministic Analytics & Segmentation
segmented_df = assign_business_segments(predictions_df)
kpis = compute_executive_kpis(predictions_df, metrics)
structured_facts = generate_structured_facts(predictions_df, metrics)

# ── Top Global SaaS Header ──
render_top_header(len(predictions_df), is_eval, roc_auc=metrics.get("roc_auc", 0.793))

# Notifications (Non-blocking)
if notifications and not is_eval:
    for note in notifications:
        st.caption(f"ℹ️ {note}")


# ══════════════════════════════════════════════════════════════
# VIEW 1: DASHBOARD (EXECUTIVE DECISION PLATFORM)
# ══════════════════════════════════════════════════════════════
if st.session_state.active_nav == "Dashboard":
    d_h1, d_h2 = st.columns([3.5, 1.5])
    with d_h1:
        st.markdown('<div class="page-title">Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Monitor purchase intent, conversion behavior, and model performance.</div>', unsafe_allow_html=True)
    with d_h2:
        st.markdown(
            f'<div style="text-align:right; font-size:0.78rem; color:#64748b; margin-top:4px;">'
            f'<strong>Online Shoppers</strong> • {len(predictions_df):,} sessions<br>'
            f'<span class="badge {"badge-mode-b" if is_eval else "badge-mode-a"}">{"● Evaluation Mode" if is_eval else "● Prediction Mode"}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── 4 COMPACT KPI CARDS ──
    col1, col2, col3, col4 = st.columns(4)
    if is_eval:
        m = eval_summary["metrics"]
        with col1:
            render_kpi_card("Conversion Rate", f"{m['actual_conversion_rate']}%", f"{m['actual_conversions']} confirmed buyer sessions", card_theme="green")
        with col2:
            render_kpi_card("Predicted Buyers", f"{kpis['predicted_conversions']}", f"{kpis['conversion_rate_50']}% model flag rate (0.50 cutoff)", card_theme="blue")
        with col3:
            render_kpi_card("Live F1-Score", f"{m['f1']:.3f}", f"Holdout benchmark: {metrics.get('f1', 0.704):.3f}", card_theme="slate")
        with col4:
            render_kpi_card("Brier Calibration", f"{m['brier_score']:.4f}", f"ROC-AUC: {m['roc_auc']:.3f}", card_theme="coral")
    else:
        with col1:
            render_kpi_card("High Intent", f"{kpis['high_intent_count']}", f"{kpis['high_intent_pct']}% of sessions (Score ≥ {kpis['high_cutoff']:.2f})", card_theme="coral")
        with col2:
            render_kpi_card("Avg Probability", f"{kpis['avg_probability']}%", "Average cohort model score", card_theme="blue")
        with col3:
            render_kpi_card("Medium Intent", f"{kpis['medium_intent_count']}", f"{kpis['medium_intent_pct']}% of sessions (Score ≥ {kpis['medium_cutoff']:.2f})", card_theme="slate")
        with col4:
            render_kpi_card("Low Intent", f"{kpis['low_intent_count']}", f"{kpis['low_intent_pct']}% of sessions", card_theme="slate")

    # ── PRIMARY INSIGHT CARD ──
    ins_col1, ins_col2 = st.columns([4.2, 1])
    with ins_col1:
        render_insight_strip(
            "Key Insight",
            f"Shoppers in the High-Intent tier exhibit an average inactivity lapse of <strong>{kpis.get('avg_recency', 45):.0f} days</strong> "
            f"compared to extended inactivity among lower tiers. Inactivity lapse and historical spend drive over 50% of model discrimination."
        )
    with ins_col2:
        st.write("")
        if st.button("Explore Predictions →", key="btn_dash_to_preds", use_container_width=True):
            st.session_state.active_nav = "Predictions"
            st.rerun()

    # ── CORE 2x2 ANALYTICS GRID ──
    grid_r1_c1, grid_r1_c2 = st.columns([1.2, 1])
    with grid_r1_c1:
        st.markdown(
            '<div class="chart-card">'
            '<div class="chart-card-title">Conversion Probability</div>'
            '<div class="chart-card-sub">Distribution of model scores across active sessions</div>',
            unsafe_allow_html=True
        )
        fig_dist = render_probability_distribution(predictions_df)
        st.plotly_chart(fig_dist, config={"displayModeBar": False}, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with grid_r1_c2:
        st.markdown(
            '<div class="chart-card">'
            '<div class="chart-card-title">Session Intent Distribution</div>'
            '<div class="chart-card-sub">How sessions are distributed across model-score tiers</div>',
            unsafe_allow_html=True
        )
        fig_donut = render_intent_donut(predictions_df)
        st.plotly_chart(fig_donut, config={"displayModeBar": False}, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    grid_r2_c1, grid_r2_c2 = st.columns([1.2, 1])
    with grid_r2_c1:
        geo_df = compute_geographic_breakdown(predictions_df)
        if not geo_df.empty:
            st.markdown(
                '<div class="chart-card">'
                '<div class="chart-card-title">Conversion Likelihood by Market</div>'
                '<div class="chart-card-sub">Average model score across the highest-volume markets</div>',
                unsafe_allow_html=True
            )
            fig_geo = px.bar(
                geo_df.head(6),
                x="CountryBucket",
                y="Avg_Conversion_Probability",
                color="Avg_Conversion_Probability",
                color_continuous_scale="Blues",
                text=geo_df.head(6)["Avg_Conversion_Probability"].apply(lambda x: f"{x:.1f}%"),
                labels={"CountryBucket": "Market", "Avg_Conversion_Probability": "Avg Likelihood %"}
            )
            fig_geo.update_layout(
                height=260,
                margin=dict(l=16, r=16, t=10, b=20),
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(showgrid=True, gridcolor="#f1f5f9")
            )
            st.plotly_chart(fig_geo, config={"displayModeBar": False}, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with grid_r2_c2:
        st.markdown(
            '<div class="chart-card">'
            '<div class="chart-card-title">Catalog Exploration vs Likelihood</div>'
            '<div class="chart-card-sub">Relationship between distinct products viewed and model score</div>',
            unsafe_allow_html=True
        )
        fig_spend = render_spend_scatter(predictions_df)
        st.plotly_chart(fig_spend, config={"displayModeBar": False}, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── COMMERCIAL OPPORTUNITIES / SEGMENTS SUMMARY ──
    st.markdown(
        '<div class="chart-card">'
        '<div class="chart-card-title">Commercial Micro-Segments & Action Matrix</div>'
        '<div class="chart-card-sub">Actionable customer clusters based on RFM profile and conversion intent</div>',
        unsafe_allow_html=True
    )
    seg_counts = segmented_df["Business Segment"].value_counts().reset_index()
    seg_counts.columns = ["Business Segment", "Sessions"]
    seg_counts["Share"] = (seg_counts["Sessions"] / len(segmented_df) * 100).round(1).astype(str) + "%"

    proto_map = {
        "VIP High-Value Champions": "Immediate checkout concierge, exclusive preview access",
        "Active Consideration Nudge": "Automated abandoned browse reminder, 10% voucher expiring in 48h",
        "At-Risk Warm Leads": "Personalized re-engagement email featuring top-rated catalog items",
        "Lapsed High-Spenders": "High-priority winback campaign with tailored loyalty reward"
    }
    seg_counts["Recommended Action Protocol"] = seg_counts["Business Segment"].map(proto_map).fillna("Standard engagement")
    st.dataframe(seg_counts, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── TOP HIGH-INTENT SESSIONS PREVIEW ──
    st.markdown(
        '<div class="chart-card">'
        '<div class="chart-card-title">Top High-Opportunity Sessions</div>'
        '<div class="chart-card-sub">Sessions exhibiting the highest conversion likelihood for immediate engagement</div>',
        unsafe_allow_html=True
    )
    top_preview = predictions_df.sort_values("Conversion Probability", ascending=False).head(5)
    prev_cols = ["Conversion Probability", "Intent Tier", "Recency", "Frequency_log", "Monetary_log", "CountryBucket"]
    if is_eval and "Prediction Result" in top_preview.columns:
        prev_cols.insert(2, "Prediction Result")

    st.dataframe(
        top_preview[prev_cols].style.format({
            "Conversion Probability": "{:.1%}",
            "Recency": "{:.0f}",
            "Frequency_log": "{:.2f}",
            "Monetary_log": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# VIEW 2: PREDICTIONS (ANALYTICAL WORKSPACE)
# ══════════════════════════════════════════════════════════════
elif st.session_state.active_nav == "Predictions":
    st.markdown('<div class="page-title">Predictions</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Explore session-level purchase intent and identify high-opportunity sessions.</div>', unsafe_allow_html=True)

    # Filter Toolbar
    with st.container():
        st.markdown('<div class="chart-card" style="padding:12px 16px; margin-bottom:14px;">', unsafe_allow_html=True)
        if is_eval:
            f_col1, f_col2, f_col3, f_col4 = st.columns([1.2, 1.2, 1.2, 1.4])
        else:
            f_col1, f_col2, f_col3 = st.columns([1.5, 1.5, 2])

        with f_col1:
            tier_filter = st.selectbox("Intent Tier:", ["All", "High Intent", "Medium Intent", "Low Intent"])
        with f_col2:
            countries = ["All"] + sorted(predictions_df["CountryBucket"].unique().tolist()) if "CountryBucket" in predictions_df.columns else ["All"]
            country_filter = st.selectbox("Market Region:", countries)

        if is_eval:
            with f_col3:
                res_filter = st.selectbox("Classification Outcome:", ["All", "Only Errors", "Correct Purchase", "False Alarm", "Missed Conversion", "Correct Non-Purchase"])
            with f_col4:
                p_min = st.slider("Min Probability:", 0.0, 1.0, 0.0, 0.05)
        else:
            with f_col3:
                p_min = st.slider("Min Probability:", 0.0, 1.0, 0.0, 0.05)
        st.markdown('</div>', unsafe_allow_html=True)

    # Filter Application
    view_df = predictions_df.copy()
    if tier_filter != "All":
        view_df = view_df[view_df["Intent Tier"] == tier_filter]
    if country_filter != "All":
        view_df = view_df[view_df["CountryBucket"] == country_filter]
    view_df = view_df[view_df["Conversion Probability"] >= p_min]

    if is_eval:
        if res_filter == "Only Errors":
            view_df = view_df[view_df["Prediction Result"].isin(["False Alarm", "Missed Conversion"])]
        elif res_filter != "All":
            view_df = view_df[view_df["Prediction Result"] == res_filter]

    st.caption(f"Showing **{len(view_df):,}** of {len(predictions_df):,} total sessions.")

    # Data Table
    if is_eval:
        show_cols = ["Conversion Probability", "Intent Tier", "Prediction Result", "Actual Label", "Predicted Label", "Recency", "Frequency_log", "Monetary_log", "DistinctProducts_log", "CountryBucket"]
    else:
        show_cols = ["Conversion Probability", "Intent Tier", "Recency", "Frequency_log", "Monetary_log", "DistinctProducts_log", "Tenure", "CountryBucket"]
    avail_cols = [c for c in show_cols if c in view_df.columns]

    fmt_dict = {
        "Conversion Probability": "{:.1%}",
        "Recency": "{:.0f}",
        "Tenure": "{:.0f}",
        "Frequency_log": "{:.2f}",
        "Monetary_log": "{:.2f}",
        "DistinctProducts_log": "{:.2f}"
    }
    st.dataframe(
        view_df[avail_cols].style.format({k: v for k, v in fmt_dict.items() if k in avail_cols}),
        use_container_width=True,
        height=320
    )

    st.markdown("---")

    # Session Deep-Dive Inspector
    if len(view_df) > 0:
        st.markdown('<div class="chart-card-title">Session Detail Inspector</div>', unsafe_allow_html=True)
        session_opts = list(range(len(view_df)))
        selected_row_idx = st.selectbox(
            "Select Session to Inspect:",
            options=session_opts,
            format_func=lambda i: f"Session #{i+1} — Likelihood: {view_df.iloc[i]['Conversion Probability']:.1%} [{view_df.iloc[i]['Intent Tier']}]" + (f" ({view_df.iloc[i]['Prediction Result']})" if is_eval else "")
        )
        row = view_df.iloc[selected_row_idx]
        render_session_detail_panel(row, is_eval, metrics.get("feature_importances", {}))
    else:
        st.info("No sessions match the current filter selection.")


# ══════════════════════════════════════════════════════════════
# VIEW 3: BEHAVIOR ANALYTICS (WORKSPACE)
# ══════════════════════════════════════════════════════════════
elif st.session_state.active_nav == "Behavior Analytics":
    st.markdown('<div class="page-title">Behavior Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Understand behavioral signals driving purchase intent across active cohorts.</div>', unsafe_allow_html=True)

    b_col1, b_col2 = st.columns([1.2, 1])
    with b_col1:
        st.markdown(
            '<div class="chart-card">'
            '<div class="chart-card-title">Inactivity Lapse (Recency) vs Conversion Probability</div>'
            '<div class="chart-card-sub">Empirical observation: Likelihood drops sharply as inactivity exceeds 60 days</div>',
            unsafe_allow_html=True
        )
        fig_rec = render_inactivity_scatter(predictions_df)
        st.plotly_chart(fig_rec, config={"displayModeBar": False}, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with b_col2:
        st.markdown(
            '<div class="chart-card">'
            '<div class="chart-card-title">Customer Lifetime Spend vs Catalog Variety</div>'
            '<div class="chart-card-sub">Bubble size reflects individual session conversion probability</div>',
            unsafe_allow_html=True
        )
        fig_spend = render_spend_scatter(predictions_df)
        st.plotly_chart(fig_spend, config={"displayModeBar": False}, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="chart-card">'
        '<div class="chart-card-title">Aggregated Behavioral Metrics by Intent Tier</div>'
        '<div class="chart-card-sub">Mean and median behavioral indicators across High, Medium, and Low intent cohorts</div>',
        unsafe_allow_html=True
    )
    profiles_df = compute_behavior_profiles(predictions_df)
    st.dataframe(profiles_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# VIEW 4: EVALUATION (OUTCOME EVALUATION CONSOLE)
# ══════════════════════════════════════════════════════════════
elif st.session_state.active_nav == "Evaluation":
    st.markdown('<div class="page-title">Ground-Truth Outcome Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Rigorous ML validation against observed purchase outcomes, error diagnostics, and decision boundary calibration.</div>', unsafe_allow_html=True)

    if not is_eval:
        st.markdown("""
        <div class="chart-card" style="padding:32px 24px; text-align:center;">
            <div style="font-size:1.5rem; margin-bottom:8px;">🎯</div>
            <div style="font-weight:700; font-size:1.1rem; color:#0f172a; margin-bottom:6px;">Evaluation Mode Inactive</div>
            <div style="font-size:0.85rem; color:#64748b; max-width:500px; margin:0 auto 16px auto;">
                The current dataset contains no ground-truth <code>Revenue</code> column.
                To inspect the Confusion Matrix, Error Profiling, and Decision Threshold Sensitivity, select the preloaded evaluation batch from the sidebar.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        m = eval_summary["metrics"]
        cm = eval_summary["confusion_matrix"]
        err = eval_summary["error_analysis"]

        # 4 Evaluation KPIs
        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        with e_col1:
            render_kpi_card("Accuracy", f"{m['accuracy']:.1%}", f"{cm['correct_total']} of {m['total_evaluated']} correct", card_theme="green")
        with e_col2:
            render_kpi_card("Live F1-Score", f"{m['f1']:.3f}", f"Precision: {m['precision']:.1%} | Recall: {m['recall']:.1%}", card_theme="blue")
        with e_col3:
            render_kpi_card("Actual vs Predicted", f"{cm['TP'] + cm['FN']} / {kpis['predicted_conversions']}", f"{m['actual_conversion_rate']}% confirmed conversion rate", card_theme="slate")
        with e_col4:
            render_kpi_card("Brier Calibration", f"{m['brier_score']:.4f}", f"Live ROC-AUC: {m['roc_auc']:.3f}", card_theme="coral")

        st.write("")

        ev_tab1, ev_tab2, ev_tab3 = st.tabs(["Confusion Matrix & Benchmark", "Behavioral Error Profiling", "Decision Threshold Explorer"])

        with ev_tab1:
            cm_col1, cm_col2 = st.columns([1.1, 1])
            with cm_col1:
                st.markdown(
                    '<div class="chart-card">'
                    '<div class="chart-card-title">Confusion Matrix (0.50 Decision Cutoff)</div>'
                    '<div class="chart-card-sub">Live cohort classification breakdown</div>',
                    unsafe_allow_html=True
                )
                fig_cm = render_confusion_matrix_heatmap(cm)
                st.plotly_chart(fig_cm, config={"displayModeBar": False}, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with cm_col2:
                st.markdown(
                    '<div class="chart-card">'
                    '<div class="chart-card-title">Live Cohort vs Holdout Benchmark</div>'
                    '<div class="chart-card-sub">Validation metrics against benchmark test split</div>',
                    unsafe_allow_html=True
                )
                benchmark_data = [
                    {"Metric": "ROC-AUC", "Live Cohort": f"{m['roc_auc']:.3f}", "Holdout Benchmark": f"{metrics.get('roc_auc', 0.793):.3f}"},
                    {"Metric": "F1-Score", "Live Cohort": f"{m['f1']:.3f}", "Holdout Benchmark": f"{metrics.get('f1', 0.704):.3f}"},
                    {"Metric": "Precision", "Live Cohort": f"{m['precision']:.1%}", "Holdout Benchmark": f"{metrics.get('precision', 0.674):.1%}"},
                    {"Metric": "Recall", "Live Cohort": f"{m['recall']:.1%}", "Holdout Benchmark": f"{metrics.get('recall', 0.736):.1%}"},
                    {"Metric": "Accuracy", "Live Cohort": f"{m['accuracy']:.1%}", "Holdout Benchmark": f"{metrics.get('accuracy', 0.705):.1%}"}
                ]
                st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Business Semantics
            st.markdown(f"""
            <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin-top:10px;">
                <div class="kpi-card kpi-card-green" style="padding:10px 12px;">
                    <div class="kpi-label">Correct Purchases (TP)</div>
                    <div class="kpi-val" style="font-size:1.3rem; color:#16a34a;">{cm['TP']}</div>
                    <div class="kpi-sub">Accurate high-intent capture</div>
                </div>
                <div class="kpi-card kpi-card-slate" style="padding:10px 12px;">
                    <div class="kpi-label">Correct Non-Purchases (TN)</div>
                    <div class="kpi-val" style="font-size:1.3rem;">{cm['TN']}</div>
                    <div class="kpi-sub">Avoided marketing spend waste</div>
                </div>
                <div class="kpi-card kpi-card-slate" style="padding:10px 12px; border-top-color:#d97706;">
                    <div class="kpi-label">False Alarms (FP)</div>
                    <div class="kpi-val" style="font-size:1.3rem; color:#d97706;">{cm['FP']}</div>
                    <div class="kpi-sub">High intent, checkout drop-off</div>
                </div>
                <div class="kpi-card kpi-card-coral" style="padding:10px 12px;">
                    <div class="kpi-label">Missed Conversions (FN)</div>
                    <div class="kpi-val" style="font-size:1.3rem; color:#dc2626;">{cm['FN']}</div>
                    <div class="kpi-sub">Unexpected buyers</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with ev_tab2:
            st.markdown(
                '<div class="chart-card">'
                '<div class="chart-card-title">Behavioral Error Profiling (Non-Causal Comparison)</div>'
                '<div class="chart-card-sub">Compare behavioral signals between false alarms and missed conversions</div>',
                unsafe_allow_html=True
            )
            missed_prof = err.get("missed_conversions", {})
            alarm_prof = err.get("false_alarms", {})
            correct_prof = err.get("correct_predictions", {})

            err_table_data = [
                {
                    "Segment": "Missed Conversions (FN)",
                    "Volume": missed_prof.get("count", 0),
                    "Avg Inactivity (Days)": missed_prof.get("avg_recency", 0),
                    "Avg Lifetime Spend Index": missed_prof.get("avg_spend", 0),
                    "Avg Model Score": f"{missed_prof.get('avg_prob', 0)}%"
                },
                {
                    "Segment": "False Alarms (FP)",
                    "Volume": alarm_prof.get("count", 0),
                    "Avg Inactivity (Days)": alarm_prof.get("avg_recency", 0),
                    "Avg Lifetime Spend Index": alarm_prof.get("avg_spend", 0),
                    "Avg Model Score": f"{alarm_prof.get('avg_prob', 0)}%"
                },
                {
                    "Segment": "Correct Classifications (TP+TN)",
                    "Volume": correct_prof.get("count", 0),
                    "Avg Inactivity (Days)": correct_prof.get("avg_recency", 0),
                    "Avg Lifetime Spend Index": correct_prof.get("avg_spend", 0),
                    "Avg Model Score": f"{correct_prof.get('avg_prob', 0)}%"
                }
            ]
            st.dataframe(pd.DataFrame(err_table_data), use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with ev_tab3:
            st.markdown(
                '<div class="chart-card">'
                '<div class="chart-card-title">Decision Threshold Explorer</div>'
                '<div class="chart-card-sub">Explore operational decision boundaries without altering production model weights. Note: Production intent cutoffs remain unchanged.</div>',
                unsafe_allow_html=True
            )
            y_true_eval = predictions_df["Actual Outcome"].values
            y_prob_eval = predictions_df["Conversion Probability"].values
            th_df = calculate_threshold_sensitivity(y_true_eval, y_prob_eval)

            sim_cutoff = st.slider("Operational Decision Threshold:", 0.20, 0.80, 0.50, 0.05)
            fig_sens = render_sensitivity_curve(th_df, sim_cutoff)
            st.plotly_chart(fig_sens, config={"displayModeBar": False}, use_container_width=True)
            st.dataframe(th_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# VIEW 5: MODEL INTELLIGENCE (GOVERNANCE & EXPLAINABILITY)
# ══════════════════════════════════════════════════════════════
elif st.session_state.active_nav == "Model Intelligence":
    st.markdown('<div class="page-title">Model Governance & Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Production model architecture, holdout benchmark performance, and feature impact contributions.</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_kpi_card("ROC-AUC", f"{metrics.get('roc_auc', 0.793):.3f}", "Holdout discrimination", card_theme="blue")
    with m2:
        render_kpi_card("F1-Score", f"{metrics.get('f1', 0.704):.3f}", "Balanced metric", card_theme="green")
    with m3:
        render_kpi_card("Precision", f"{metrics.get('precision', 0.674):.1%}", "Buyer reliability", card_theme="slate")
    with m4:
        render_kpi_card("Recall", f"{metrics.get('recall', 0.736):.1%}", "Buyer coverage", card_theme="slate")
    with m5:
        render_kpi_card("Accuracy", f"{metrics.get('accuracy', 0.705):.1%}", "Overall hit rate", card_theme="slate")

    st.write("")

    gov_col1, gov_col2 = st.columns([1.2, 1])
    with gov_col1:
        st.markdown(
            '<div class="chart-card">'
            '<div class="chart-card-title">Feature Importance (Business Aggregated)</div>'
            '<div class="chart-card-sub">Historical predictive contribution of behavioral factors</div>',
            unsafe_allow_html=True
        )
        raw_imp = metrics.get("feature_importances", {})
        prefix_map = {
            "CountryBucket_": "Geographic Region",
            "Recency": "Inactivity Lapse (Recency)",
            "Frequency_log": "Order Frequency",
            "Monetary_log": "Lifetime Spend Level",
            "DistinctProducts_log": "Product Catalog Variety",
            "Tenure": "Customer Relationship Tenure"
        }
        agg_imp = {}
        for feat, score in raw_imp.items():
            matched = False
            for prefix, label in prefix_map.items():
                if feat == prefix or feat.startswith(prefix):
                    agg_imp[label] = agg_imp.get(label, 0.0) + score
                    matched = True
                    break
            if not matched:
                agg_imp[feat] = agg_imp.get(feat, 0.0) + score

        imp_df = pd.DataFrame([{"Feature": k, "Contribution": v} for k, v in agg_imp.items()]).sort_values("Contribution", ascending=True)
        fig_imp = px.bar(
            imp_df,
            x="Contribution",
            y="Feature",
            orientation="h",
            color="Contribution",
            color_continuous_scale="Blues",
            text=imp_df["Contribution"].apply(lambda x: f"{x:.1%}")
        )
        fig_imp.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=20),
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9")
        )
        st.plotly_chart(fig_imp, config={"displayModeBar": False}, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with gov_col2:
        st.markdown(
            '<div class="chart-card">'
            '<div class="chart-card-title">Production Governance & Threshold Contracts</div>'
            '<div class="chart-card-sub">Deterministic specifications and invariance guarantees</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"""
        - **Model Family**: `{schema.get('model_type', 'RandomForestClassifier')}`
        - **Target Outcome**: `{schema.get('target', 'Revenue')}` (Purchase Conversion)
        - **High Intent Cutoff**: `≥ {metrics.get('high_intent_cutoff', 0.7281):.4f}` (80th percentile of holdout scores)
        - **Medium Intent Cutoff**: `≥ {metrics.get('medium_intent_cutoff', 0.5164):.4f}` (50th percentile of holdout scores)
        - **Deterministic Invariance**: Fixed cutoffs are established once during holdout benchmarking and never drift per batch.
        """)
        st.info("💡 Feature importance reflects historical predictive association and does not imply unilateral causality.")
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# VIEW 6: AI COPILOT (INTEGRATED ENTERPRISE ASSISTANT)
# ══════════════════════════════════════════════════════════════
elif st.session_state.active_nav == "AI Copilot":
    cop_h1, cop_h2 = st.columns([4, 1])
    with cop_h1:
        st.markdown('<div class="page-title">Executive AI Copilot</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="page-subtitle">Grounded conversational intelligence backed by deterministic cohort facts. '
            f'<span class="badge badge-system">Context: {len(predictions_df):,} active sessions ({ "Evaluation Mode" if is_eval else "Prediction Mode" })</span></div>',
            unsafe_allow_html=True
        )
    with cop_h2:
        st.write("")
        if st.button("🗑️ Reset Chat", use_container_width=True):
            st.session_state.copilot_messages = []
            st.rerun()

    # LLM Settings & Provider Selection
    with st.expander("⚙️ Reasoning Engine & LLM Settings (Google Gemini / DeepSeek)", expanded=False):
        c_p1, c_p2 = st.columns([1, 2])
        with c_p1:
            copilot_provider = st.selectbox(
                "Reasoning Engine:",
                options=["auto", "gemini", "deepseek", "local"],
                format_func=lambda x: {
                    "auto": "Auto (Best Available)",
                    "gemini": "Google Gemini 1.5 Flash",
                    "deepseek": "DeepSeek V3 Chat",
                    "local": "Local Deterministic Engine"
                }.get(x, x),
                index=0,
                key="copilot_provider_select"
            )
        with c_p2:
            copilot_api_key = st.text_input(
                "API Key (Optional for Gemini / DeepSeek):",
                type="password",
                placeholder="AIzaSy... or sk-...",
                help="Enter your Gemini or DeepSeek API key to enable strategic LLM reasoning beyond fixed templates.",
                key="copilot_api_key_input"
            )

    copilot = RAGExecutiveCopilot(predictions_df, metrics, schema, evaluation_facts=eval_facts)

    if "copilot_messages" not in st.session_state or not isinstance(st.session_state.copilot_messages, list) or len(st.session_state.copilot_messages) == 0:
        mode_desc = "Evaluation Mode with ground truth Revenue" if is_eval else "Prediction Mode"
        greeting = (
            f"Hello! I am your **Retail Intelligence Copilot**.\\n\\n"
            f"I have indexed **{len(predictions_df):,} sessions** currently loaded in **{mode_desc}**.\\n\\n"
            f"You can ask me about conversion health, commercial retention strategies, predictive drivers, or specific failure patterns."
        )
        st.session_state.copilot_messages = [
            {
                "role": "assistant",
                "content": greeting,
                "citations": [f"Indexed Sessions: {len(predictions_df)}", f"Active Mode: {'Evaluation' if is_eval else 'Prediction'}"]
            }
        ]

    # Suggested Inquiries at top
    st.markdown("**Suggested Inquiries:**")
    q_cols = st.columns(4)
    selected_chip = None

    if is_eval:
        if q_cols[0].button("🎯 Evaluation Health", use_container_width=True):
            selected_chip = "Summarize the evaluation results against actual outcomes."
        if q_cols[1].button("❌ Missed Conversions", use_container_width=True):
            selected_chip = "Why were conversions missed and what were their behavioral signals?"
        if q_cols[2].button("⚠️ False Alarm Profiling", use_container_width=True):
            selected_chip = "What caused false alarms and how does that impact marketing?"
        if q_cols[3].button("⚖️ Threshold Trade-offs", use_container_width=True):
            selected_chip = "What happens if we lower or raise the decision threshold?"
    else:
        if q_cols[0].button("📊 Cohort Diagnostic", use_container_width=True):
            selected_chip = "Summarize the conversion health of this cohort."
        if q_cols[1].button("🎯 Retention Strategy", use_container_width=True):
            selected_chip = "What retention marketing strategy should we deploy?"
        if q_cols[2].button("🔍 Driver Analysis", use_container_width=True):
            selected_chip = "What behavioral factors drive conversions most strongly?"
        if q_cols[3].button("⚠️ Inactive Shoppers", use_container_width=True):
            selected_chip = "What are the risks and patterns among low intent shoppers?"

    # Chronological Top-to-Bottom Conversation Stream
    chat_box = st.container()
    with chat_box:
        for msg in st.session_state.copilot_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("citations"):
                    citations_html = "".join([f'<span class="citation-chip">{c}</span>' for c in msg["citations"]])
                    st.markdown(f'<div style="margin-top:6px;">{citations_html}</div>', unsafe_allow_html=True)

    # Bottom spacer so pinned chat input never covers the last message
    st.markdown('<div class="chat-bottom-spacer"></div>', unsafe_allow_html=True)

    # Pinned Input at Bottom
    user_typed = st.chat_input("Ask a question about conversion signals, customer segments, or evaluation metrics...")
    active_prompt = selected_chip or user_typed

    if active_prompt:
        st.session_state.copilot_messages.append({"role": "user", "content": active_prompt})
        with st.spinner("Synthesizing cohort evidence & reasoning..."):
            res = copilot.query(
                active_prompt,
                chat_history=st.session_state.copilot_messages,
                provider=st.session_state.get("copilot_provider_select", "auto"),
                api_key=st.session_state.get("copilot_api_key_input", None)
            )
            st.session_state.copilot_messages.append({
                "role": "assistant",
                "content": res["answer"],
                "citations": res.get("citations", [])
            })
        st.rerun()
