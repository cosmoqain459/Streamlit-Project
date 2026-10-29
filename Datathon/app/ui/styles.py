"""
Global SaaS Design Tokens, Typography, and Chrome Removal Styles.
Inspired by modern enterprise analytics platforms (Linear, Stripe, Vercel).
"""

import streamlit as st
import textwrap

def inject_custom_css():
    css = textwrap.dedent("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Typography & Canvas */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0f172a;
        background-color: #f8fafc;
        -webkit-font-smoothing: antialiased;
    }

    /* ── Hide Unnecessary Streamlit Prototype Chrome ── */
    #MainMenu { visibility: hidden !important; display: none !important; }
    footer { visibility: hidden !important; display: none !important; }
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.5rem !important;
        z-index: 10 !important;
    }
    header[data-testid="stHeader"] button[title*="Deploy"],
    header[data-testid="stHeader"] div[class*="deploy"] {
        display: none !important;
    }
    [data-testid="stToolbar"] { visibility: hidden !important; display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stStatusWidget"] { visibility: hidden !important; display: none !important; }

    /* Streamlit Main View Container */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1400px !important;
    }

    /* Top Global SaaS Header */
    .saas-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 20px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .header-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-icon {
        width: 32px;
        height: 32px;
        background: #0f172a;
        color: #ffffff;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 15px;
    }
    .brand-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.015em;
        margin: 0;
        line-height: 1.2;
    }
    .brand-sub {
        font-size: 0.76rem;
        color: #64748b;
        margin: 0;
    }
    .header-badges {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Status & Semantic Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        white-space: nowrap;
    }
    .badge-mode-a {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .badge-mode-b {
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
    }
    .badge-system {
        background-color: #f8fafc;
        color: #475569;
        border: 1px solid #e2e8f0;
    }
    .badge-high {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .badge-med {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .badge-low {
        background-color: #f8fafc;
        color: #475569;
        border: 1px solid #e2e8f0;
    }
    .badge-tp {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .badge-tn {
        background-color: #f8fafc;
        color: #334155;
        border: 1px solid #e2e8f0;
    }
    .badge-fp {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .badge-fn {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }

    /* Executive KPI Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        border-top: 3px solid #0f172a;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-card-green { border-top-color: #16a34a; }
    .kpi-card-coral { border-top-color: #dc2626; }
    .kpi-card-blue  { border-top-color: #2563eb; }
    .kpi-card-slate { border-top-color: #64748b; }

    .kpi-label {
        font-size: 0.70rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .kpi-val {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.025em;
        line-height: 1.15;
        margin-bottom: 4px;
    }
    .kpi-sub {
        font-size: 0.76rem;
        color: #64748b;
        line-height: 1.3;
    }

    /* Page Typography */
    .page-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin: 0 0 3px 0;
    }
    .page-subtitle {
        font-size: 0.86rem;
        color: #64748b;
        margin-bottom: 18px;
        line-height: 1.4;
    }

    /* Insight Strip Component */
    .insight-strip {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 13px 18px;
        margin-bottom: 20px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .insight-lead {
        font-weight: 700;
        color: #2563eb;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 3px;
    }
    .insight-text {
        font-size: 0.87rem;
        color: #1e293b;
        line-height: 1.5;
        margin: 0;
    }

    /* Chart & Container Cards */
    .chart-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 18px 12px 18px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        margin-bottom: 18px;
    }
    .chart-card-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .chart-card-sub {
        font-size: 0.76rem;
        color: #64748b;
        margin-bottom: 8px;
    }

    /* ── Professional Sidebar Styling ── */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .sidebar-brand-box {
        padding: 4px 0 14px 0;
        border-bottom: 1px solid #f1f5f9;
        margin-bottom: 14px;
    }
    .sidebar-section-header {
        font-size: 0.67rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #94a3b8;
        font-weight: 700;
        margin: 14px 0 6px 4px;
    }

    /* Sidebar Nav Buttons Styling */
    section[data-testid="stSidebar"] .stButton button {
        border-radius: 6px !important;
        padding: 6px 10px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        transition: all 0.12s ease !important;
        margin-bottom: 2px !important;
    }
    section[data-testid="stSidebar"] .stButton button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid transparent !important;
        color: #475569 !important;
    }
    section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
        background: #f8fafc !important;
        border-color: #e2e8f0 !important;
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        border-left: 3px solid #0f172a !important;
    }

    /* Chat Message Polish */
    [data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 14px 18px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    }
    [data-testid="stChatMessageContent"] {
        font-size: 0.9rem !important;
        line-height: 1.55 !important;
        color: #0f172a !important;
    }
    .citation-chip {
        display: inline-block;
        background: #f1f5f9;
        color: #334155;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 6px;
        margin-top: 6px;
        border: 1px solid #cbd5e1;
    }
    .chat-bottom-spacer {
        height: 80px;
        width: 100%;
    }
    </style>
    """)
    st.markdown(css, unsafe_allow_html=True)
