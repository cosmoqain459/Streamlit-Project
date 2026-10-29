"""
Automated Test Suite for the Retail Purchase & Conversion Intelligence Platform
Verifies:
1. Artifact contracts and schema validity.
2. Single-row and batch inference.
3. Unseen category handling.
4. Missing feature validation.
5. Target column exclusion.
6. Deterministic Analytics Engine and Segmentation.
7. AI Insights factual synthesis.
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))

from predict import (
    load_metrics, load_schema, predict_batch, validate_and_prepare_input
)
from analytics_engine import (
    compute_executive_kpis, assign_business_segments, generate_structured_facts
)
from ai_insights import (
    generate_deterministic_insights, synthesize_ai_insights
)

ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app', 'artifacts'))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app', 'data'))

def test_artifacts_exist_and_conform():
    """Verify all contract files exist and have non-empty valid JSON."""
    schema_path = os.path.join(ARTIFACTS_DIR, 'schema.json')
    metrics_path = os.path.join(ARTIFACTS_DIR, 'metrics.json')

    assert os.path.exists(schema_path), "schema.json missing"
    assert os.path.exists(metrics_path), "metrics.json missing"

    with open(schema_path, 'r') as f:
        schema = json.load(f)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    assert schema.get("target") in ["Revenue", "Churned"]
    assert "numeric_features" in schema
    assert "categorical_features" in schema
    assert "roc_auc" in metrics
    assert "f1" in metrics
    assert "high_intent_cutoff" in metrics
    assert "medium_intent_cutoff" in metrics
    assert metrics["high_intent_cutoff"] > metrics["medium_intent_cutoff"]

def test_single_row_and_batch_inference():
    """Verify inference runs on sample batch and single row with valid outputs."""
    sample_path = os.path.join(DATA_DIR, 'sample_sessions.csv')
    model_path = os.path.join(ARTIFACTS_DIR, 'model.pkl')

    if not os.path.exists(sample_path) or not os.path.exists(model_path):
        pytest.skip("Artifacts or sample data not yet downloaded.")

    df = pd.read_csv(sample_path)
    res_df, notes = predict_batch(df)

    assert "Conversion Probability" in res_df.columns
    assert "Intent Tier" in res_df.columns
    assert len(res_df) == len(df)
    assert res_df["Conversion Probability"].between(0.0, 1.0).all()
    assert set(res_df["Intent Tier"].unique()).issubset({"High Intent", "Medium Intent", "Low Intent"})

    # Single row inference test
    single_res, _ = predict_batch(df.iloc[[0]])
    assert len(single_res) == 1
    assert 0.0 <= single_res["Conversion Probability"].iloc[0] <= 1.0
    assert single_res["Intent Tier"].iloc[0] in ["High Intent", "Medium Intent", "Low Intent"]

def test_unseen_categorical_value():
    """Verify unexpected categorical values do not crash the pipeline."""
    sample_path = os.path.join(DATA_DIR, 'sample_sessions.csv')
    model_path = os.path.join(ARTIFACTS_DIR, 'model.pkl')

    if not os.path.exists(sample_path) or not os.path.exists(model_path):
        pytest.skip("Artifacts not yet downloaded.")

    df = pd.read_csv(sample_path).head(5).copy()
    df["CountryBucket"] = "Alien_Country_999"

    res_df, _ = predict_batch(df)
    assert len(res_df) == 5
    assert not res_df["Conversion Probability"].isnull().any()

def test_missing_feature_validation_error():
    """Verify that missing required features raise a clear descriptive ValueError."""
    sample_path = os.path.join(DATA_DIR, 'sample_sessions.csv')
    if not os.path.exists(sample_path):
        pytest.skip("Sample data missing.")

    df = pd.read_csv(sample_path).drop(columns=["Recency"])
    schema = load_schema()

    with pytest.raises(ValueError, match="Missing required columns.*Recency"):
        validate_and_prepare_input(df, schema)

def test_target_exclusion_notification():
    """Verify accidental inclusion of target column is stripped and reported."""
    sample_path = os.path.join(DATA_DIR, 'sample_sessions.csv')
    if not os.path.exists(sample_path):
        pytest.skip("Sample data missing.")

    df = pd.read_csv(sample_path).copy()
    df["Revenue"] = [1] * len(df)
    schema = load_schema()

    clean_df, notes = validate_and_prepare_input(df, schema)
    assert "Revenue" not in clean_df.columns
    assert any("Target column 'Revenue' detected and ignored" in note for note in notes)

def test_deterministic_analytics_and_ai_insights():
    """Verify analytics engine produces valid facts and AI insights synthesize cleanly."""
    sample_path = os.path.join(DATA_DIR, 'sample_sessions.csv')
    model_path = os.path.join(ARTIFACTS_DIR, 'model.pkl')

    if not os.path.exists(sample_path) or not os.path.exists(model_path):
        pytest.skip("Artifacts missing.")

    df = pd.read_csv(sample_path)
    scored_df, _ = predict_batch(df)
    metrics = load_metrics()

    # Test KPIs
    kpis = compute_executive_kpis(scored_df, metrics)
    assert kpis["total_sessions"] == len(df)
    assert "high_intent_count" in kpis
    assert "avg_probability" in kpis

    # Test Segmentation
    segmented_df = assign_business_segments(scored_df)
    assert "Business Segment" in segmented_df.columns
    assert segmented_df["Business Segment"].nunique() > 0

    # Test Structured Facts & AI Synthesis
    facts = generate_structured_facts(scored_df, metrics)
    assert "kpis" in facts
    assert "primary_model_drivers" in facts

    insights = synthesize_ai_insights(facts)
    assert "executive_summary" in insights
    assert len(insights["executive_summary"]) >= 3
    assert "recommended_actions" in insights
    assert len(insights["recommended_actions"]) >= 3


from evaluation_engine import (
    is_evaluation_dataset, prepare_evaluation_data,
    predict_evaluation_dataset, calculate_threshold_sensitivity,
    build_evaluation_facts
)
from rag_copilot import RAGExecutiveCopilot

def test_evaluation_dataset_detection():
    """Verify evaluation dataset detector correctly distinguishes Mode A vs Mode B."""
    eval_path = os.path.join(DATA_DIR, 'sample_evaluation_sessions.csv')
    pred_path = os.path.join(DATA_DIR, 'sample_sessions.csv')

    df_eval = pd.read_csv(eval_path)
    df_pred = pd.read_csv(pred_path)

    assert is_evaluation_dataset(df_eval) is True
    assert is_evaluation_dataset(df_pred) is False

def test_evaluation_target_isolation():
    """Verify target isolation ensures Revenue never leaks into the feature matrix."""
    eval_path = os.path.join(DATA_DIR, 'sample_evaluation_sessions.csv')
    df = pd.read_csv(eval_path)
    schema = load_schema()

    X_df, y_true = prepare_evaluation_data(df, schema)

    assert "Revenue" not in X_df.columns
    assert len(X_df) == len(df)
    assert len(y_true) == len(df)
    assert set(y_true.unique()).issubset({0, 1})

def test_predict_evaluation_dataset_and_metrics():
    """Verify complete evaluation pipeline outputs valid metrics, confusion matrix, and result tags."""
    eval_path = os.path.join(DATA_DIR, 'sample_evaluation_sessions.csv')
    df = pd.read_csv(eval_path)

    evaluated_df, summary = predict_evaluation_dataset(df)

    assert "Actual Outcome" in evaluated_df.columns
    assert "Predicted Outcome" in evaluated_df.columns
    assert "Prediction Result" in evaluated_df.columns

    valid_results = {"Correct Purchase", "Correct Non-Purchase", "False Alarm", "Missed Conversion"}
    assert set(evaluated_df["Prediction Result"].unique()).issubset(valid_results)

    m = summary["metrics"]
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["recall"] <= 1.0
    assert 0.0 <= m["f1"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["brier_score"] <= 1.0

    cm = summary["confusion_matrix"]
    assert cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"] == len(df)
    assert cm["correct_total"] == cm["TP"] + cm["TN"]
    assert cm["error_total"] == cm["FP"] + cm["FN"]

def test_threshold_sensitivity_simulation():
    """Verify exploratory threshold simulation computes valid metrics across cutoffs."""
    eval_path = os.path.join(DATA_DIR, 'sample_evaluation_sessions.csv')
    df = pd.read_csv(eval_path)

    evaluated_df, _ = predict_evaluation_dataset(df)
    y_true = evaluated_df["Actual Outcome"].values
    y_prob = evaluated_df["Conversion Probability"].values

    th_df = calculate_threshold_sensitivity(y_true, y_prob)

    assert len(th_df) >= 5
    assert "Threshold" in th_df.columns
    assert "Precision" in th_df.columns
    assert "Recall" in th_df.columns
    assert "F1-Score" in th_df.columns
    assert (th_df["Precision"].between(0.0, 1.0)).all()
    assert (th_df["Recall"].between(0.0, 1.0)).all()

def test_rag_copilot_conversational_and_evaluation_queries():
    """Verify RAG Copilot answers both general and evaluation queries with citations and chat history."""
    eval_path = os.path.join(DATA_DIR, 'sample_evaluation_sessions.csv')
    df = pd.read_csv(eval_path)
    metrics = load_metrics()
    schema = load_schema()

    evaluated_df, summary = predict_evaluation_dataset(df)
    eval_facts = build_evaluation_facts(evaluated_df, summary)

    copilot = RAGExecutiveCopilot(evaluated_df, metrics, schema, evaluation_facts=eval_facts)

    # Mode B evaluation query
    res_missed = copilot.query("Why were conversions missed?")
    assert "Missed Conversions" in res_missed["answer"]
    assert len(res_missed["citations"]) > 0

    res_fp = copilot.query("Tell me about false alarms and marketing impact")
    assert "False Alarms" in res_fp["answer"]
    assert len(res_fp["citations"]) > 0

    # Conversational follow-up query with history
    history = [
        {"role": "user", "content": "What retention strategy should we deploy?"},
        {"role": "assistant", "content": "VIP Conversion Sprint for high intent shoppers."}
    ]
    res_followup = copilot.query("What about the second action for medium intent?", chat_history=history)
    assert len(res_followup["citations"]) > 0
