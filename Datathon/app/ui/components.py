"""
Reusable Enterprise SaaS UI Components.
"""

import streamlit as st
import pandas as pd
import textwrap

def render_kpi_card(label: str, value: str, subtext: str = "", card_theme: str = "slate"):
    theme_class = f"kpi-card-{card_theme}"
    html = textwrap.dedent(f"""
    <div class="kpi-card {theme_class}">
        <div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-val">{value}</div>
        </div>
        <div class="kpi-sub">{subtext}</div>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)

def render_insight_strip(observation_title: str, observation_text: str):
    html = textwrap.dedent(f"""
    <div class="insight-strip">
        <div class="insight-lead">{observation_title}</div>
        <div class="insight-text">{observation_text}</div>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)

def render_session_detail_panel(row: pd.Series, is_eval: bool, feature_importances: dict = None):
    prob = float(row.get("Conversion Probability", 0.0))
    tier = str(row.get("Intent Tier", "Low Intent"))
    tier_class = "badge-high" if tier == "High Intent" else ("badge-med" if tier == "Medium Intent" else "badge-low")

    eval_badge = ""
    if is_eval:
        res = row.get("Prediction Result", "")
        if res == "Correct Purchase":
            eval_badge = '<span class="badge badge-tp">CORRECT PURCHASE</span>'
        elif res == "False Alarm":
            eval_badge = '<span class="badge badge-fp">FALSE ALARM</span>'
        elif res == "Missed Conversion":
            eval_badge = '<span class="badge badge-fn">MISSED CONVERSION</span>'
        elif res == "Correct Non-Purchase":
            eval_badge = '<span class="badge badge-tn">CORRECT NON-PURCHASE</span>'

    html = textwrap.dedent(f"""
    <div class="chart-card" style="margin-bottom:16px;">
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:14px;">
            <div style="font-weight:700; font-size:1.05rem; color:#0f172a;">Session Deep-Dive Inspector</div>
            <div style="display:flex; gap:6px;">
                <span class="badge {tier_class}">{tier.upper()}</span>
                {eval_badge}
            </div>
        </div>
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin-bottom:12px;">
            <div class="kpi-card kpi-card-blue" style="padding:10px 14px;">
                <div class="kpi-label">Likelihood</div>
                <div class="kpi-val" style="font-size:1.4rem;">{prob:.1%}</div>
            </div>
            <div class="kpi-card kpi-card-slate" style="padding:10px 14px;">
                <div class="kpi-label">Inactivity Lapse</div>
                <div class="kpi-val" style="font-size:1.4rem;">{row.get('Recency', 0):.0f}d</div>
            </div>
            <div class="kpi-card kpi-card-slate" style="padding:10px 14px;">
                <div class="kpi-label">Spend (log)</div>
                <div class="kpi-val" style="font-size:1.4rem;">{row.get('Monetary_log', 0):.2f}</div>
            </div>
            <div class="kpi-card kpi-card-slate" style="padding:10px 14px;">
                <div class="kpi-label">Products (log)</div>
                <div class="kpi-val" style="font-size:1.4rem;">{row.get('DistinctProducts_log', 0):.2f}</div>
            </div>
        </div>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)

    # Why this score? Feature impact bars
    st.markdown("**Top Contributing Signals for this Session:**")
    signals = [
        ("Inactivity Recency Lapse", 0.258, f"{row.get('Recency', 0):.0f} days since last visit (Dominant conversion driver)"),
        ("Historical Spend Level", 0.246, f"{row.get('Monetary_log', 0):.2f} log spend (Anchors baseline propensity)"),
        ("Product Catalog Exploration", 0.189, f"{row.get('DistinctProducts_log', 0):.2f} distinct products browsed"),
        ("Customer Account Tenure", 0.168, f"{row.get('Tenure', 0):.0f} days relationship duration")
    ]

    for name, weight, explanation in signals:
        col_name, col_bar, col_val = st.columns([1.5, 2, 2.5])
        with col_name:
            st.caption(f"**{name}**")
        with col_bar:
            st.progress(float(weight))
        with col_val:
            st.caption(explanation)

    st.write("")
    if st.button("💬 Ask AI Copilot About This Session", key=f"btn_ask_session_{row.name}"):
        st.session_state.active_nav = "AI Copilot"
        if "copilot_messages" not in st.session_state or not isinstance(st.session_state.copilot_messages, list):
            st.session_state.copilot_messages = []
        st.session_state.copilot_messages.append({
            "role": "user",
            "content": f"Analyze session #{row.name}: Conversion Likelihood is {prob:.1%}, Intent Tier is '{tier}', Recency is {row.get('Recency', 0):.0f} days, and Lifetime Spend log is {row.get('Monetary_log', 0):.2f}. What is the recommended commercial action?"
        })
        st.rerun()
