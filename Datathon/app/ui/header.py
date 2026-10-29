"""
Top Global SaaS Header Component.
"""

import streamlit as st
import textwrap

def render_top_header(session_count: int, is_eval: bool, roc_auc: float = 0.793):
    mode_badge = (
        '<span class="badge badge-mode-b">● Evaluation Mode</span>'
        if is_eval else
        '<span class="badge badge-mode-a">● Prediction Mode</span>'
    )
    html = textwrap.dedent(f"""
    <div class="saas-header">
        <div class="header-brand">
            <div class="brand-icon">◈</div>
            <div>
                <div class="brand-title">Retail Purchase & Conversion Intelligence</div>
                <div class="brand-sub">Enterprise Decision Support Platform</div>
            </div>
        </div>
        <div class="header-badges">
            <span class="badge badge-system">Cohort: {session_count:,} sessions</span>
            {mode_badge}
            <span class="badge badge-system">● Model Ready (ROC-AUC {roc_auc:.3f})</span>
        </div>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)
