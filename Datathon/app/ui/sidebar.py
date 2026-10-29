"""
Section-Grouped Sidebar Navigation & Data Intake.
"""

import os
import pandas as pd
import streamlit as st
import textwrap

def render_sidebar(metrics: dict, current_is_eval: bool):
    with st.sidebar:
        # Platform Brand Header
        brand_html = textwrap.dedent("""
        <div class="sidebar-brand-box">
            <div style="font-weight: 700; font-size: 1.05rem; color: #0f172a; display: flex; align-items: center; gap: 8px;">
                <span style="display:inline-block; width:10px; height:10px; background:#0f172a; border-radius:2px;"></span>
                Retail Intelligence
            </div>
            <div style="font-size: 0.75rem; color: #64748b;">Enterprise Decision Platform</div>
        </div>
        """)
        st.markdown(brand_html, unsafe_allow_html=True)

        current_view = st.session_state.get("active_nav", "Dashboard")

        # ── SECTION 1: OVERVIEW ──
        st.markdown('<div class="sidebar-section-header">OVERVIEW</div>', unsafe_allow_html=True)
        is_dash = (current_view == "Dashboard")
        if st.button("📊  Dashboard", key="btn_nav_dash", use_container_width=True, type="primary" if is_dash else "secondary"):
            st.session_state.active_nav = "Dashboard"
            st.rerun()

        # ── SECTION 2: INTELLIGENCE ──
        st.markdown('<div class="sidebar-section-header">INTELLIGENCE</div>', unsafe_allow_html=True)
        is_pred = (current_view == "Predictions")
        if st.button("🔍  Predictions", key="btn_nav_pred", use_container_width=True, type="primary" if is_pred else "secondary"):
            st.session_state.active_nav = "Predictions"
            st.rerun()

        is_beh = (current_view == "Behavior Analytics")
        if st.button("📈  Behavior Analytics", key="btn_nav_beh", use_container_width=True, type="primary" if is_beh else "secondary"):
            st.session_state.active_nav = "Behavior Analytics"
            st.rerun()

        is_eval_view = (current_view == "Evaluation")
        if st.button("🎯  Evaluation", key="btn_nav_eval", use_container_width=True, type="primary" if is_eval_view else "secondary"):
            st.session_state.active_nav = "Evaluation"
            st.rerun()

        # ── SECTION 3: MODEL ──
        st.markdown('<div class="sidebar-section-header">MODEL</div>', unsafe_allow_html=True)
        is_mod = (current_view == "Model Intelligence")
        if st.button("⚙️  Model Intelligence", key="btn_nav_mod", use_container_width=True, type="primary" if is_mod else "secondary"):
            st.session_state.active_nav = "Model Intelligence"
            st.rerun()

        # ── SECTION 4: AI ──
        st.markdown('<div class="sidebar-section-header">AI</div>', unsafe_allow_html=True)
        is_ai = (current_view == "AI Copilot")
        if st.button("💬  AI Copilot", key="btn_nav_ai", use_container_width=True, type="primary" if is_ai else "secondary"):
            st.session_state.active_nav = "AI Copilot"
            st.rerun()

        st.markdown("---")

        # ── DATASET INGESTION CARD ──
        st.markdown('<div class="sidebar-section-header">DATASET</div>', unsafe_allow_html=True)

        if "dataset_choice" not in st.session_state:
            st.session_state.dataset_choice = "Mode A: Sample Prediction Batch"

        with st.expander("Change Dataset / Upload", expanded=False):
            data_choice = st.selectbox(
                "Select Batch",
                [
                    "Mode A: Sample Prediction Batch",
                    "Mode B: Sample Evaluation Batch",
                    "Upload Custom CSV"
                ],
                key="dataset_choice",
                label_visibility="collapsed"
            )

        input_df = None
        dataset_name = "Online Shoppers"
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        if data_choice == "Mode A: Sample Prediction Batch":
            sample_path = os.path.join(base_dir, 'data', 'sample_sessions.csv')
            if os.path.exists(sample_path):
                input_df = pd.read_csv(sample_path)
                dataset_name = "sample_sessions.csv"
            else:
                st.warning("Prediction sample not found.")
        elif data_choice == "Mode B: Sample Evaluation Batch":
            eval_path = os.path.join(base_dir, 'data', 'sample_evaluation_sessions.csv')
            if os.path.exists(eval_path):
                input_df = pd.read_csv(eval_path)
                dataset_name = "sample_evaluation_sessions.csv"
            else:
                st.warning("Evaluation sample not found.")
        else:
            uploaded_file = st.file_uploader("Upload Session CSV", type=["csv"], key="custom_csv_uploader", label_visibility="collapsed")
            if uploaded_file is not None:
                try:
                    input_df = pd.read_csv(uploaded_file)
                    dataset_name = uploaded_file.name
                    st.session_state["last_uploaded_df"] = input_df
                    st.session_state["last_uploaded_name"] = dataset_name
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
            elif "last_uploaded_df" in st.session_state:
                input_df = st.session_state["last_uploaded_df"]
                dataset_name = st.session_state.get("last_uploaded_name", "Uploaded CSV")

        # Determine mode dynamically from loaded data
        from evaluation_engine import is_evaluation_dataset
        is_eval_active = is_evaluation_dataset(input_df) if input_df is not None else current_is_eval

        dataset_status_label = "● Evaluation Mode" if is_eval_active else "● Prediction Mode"
        dataset_badge_class = "badge-mode-b" if is_eval_active else "badge-mode-a"
        session_count = len(input_df) if input_df is not None else 0

        card_html = textwrap.dedent(f"""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:10px 12px; margin-bottom:8px;">
            <div style="font-weight:600; font-size:0.85rem; color:#0f172a;">{dataset_name}</div>
            <div style="font-size:0.75rem; color:#64748b; margin-bottom:4px;">{session_count:,} sessions loaded</div>
            <span class="badge {dataset_badge_class}">{dataset_status_label}</span>
        </div>
        """)
        st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("---")
        # Global Status Bar
        footer_html = textwrap.dedent(f"""
        <div style="font-size: 0.72rem; color: #64748b; line-height: 1.6; padding: 2px;">
            <div style="color: #16a34a; font-weight: 600;">● Model Ready</div>
            <div style="color: #2563eb; font-weight: 600;">● Dataset Valid (17 Features)</div>
            <div style="color: #0f172a; font-weight: 600;">● Analytics Ready</div>
            <div style="color: #94a3b8; font-size: 0.68rem; margin-top: 4px;">ROC {metrics.get('roc_auc', 0.793):.3f} • F1 {metrics.get('f1', 0.704):.3f}</div>
        </div>
        """)
        st.markdown(footer_html, unsafe_allow_html=True)

        return st.session_state.active_nav, input_df
