"""
Prediction & Inference Engine
Loads pre-trained model artifacts, validates session schema, runs inference,
and assigns deterministic intent tiers based on fixed holdout thresholds.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')

@st.cache_resource
def load_model(artifacts_dir=ARTIFACTS_DIR):
    model_path = os.path.join(artifacts_dir, 'model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please place trained model artifacts in {artifacts_dir}."
        )
    return joblib.load(model_path)

@st.cache_resource
def load_encoder(artifacts_dir=ARTIFACTS_DIR):
    encoder_path = os.path.join(artifacts_dir, 'encoder.pkl')
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder file not found at {encoder_path}.")
    return joblib.load(encoder_path)

@st.cache_resource
def load_scaler(artifacts_dir=ARTIFACTS_DIR):
    scaler_path = os.path.join(artifacts_dir, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}.")
    return joblib.load(scaler_path)

@st.cache_resource
def load_metrics(artifacts_dir=ARTIFACTS_DIR):
    metrics_path = os.path.join(artifacts_dir, 'metrics.json')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}.")
    with open(metrics_path, 'r') as f:
        return json.load(f)

@st.cache_resource
def load_schema(artifacts_dir=ARTIFACTS_DIR):
    schema_path = os.path.join(artifacts_dir, 'schema.json')
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found at {schema_path}.")
    with open(schema_path, 'r') as f:
        return json.load(f)

def validate_and_prepare_input(df: pd.DataFrame, schema: dict):
    """
    Validates that required features exist, strips any accidental target column,
    and formats data types.
    Returns: (cleaned_df, notifications_list)
    """
    notifications = []
    df = df.copy()

    # Target exclusion handling (Revenue is primary target, generic variants handled safely)
    possible_targets = ["Revenue", "revenue", "Churned", "target", "y"]
    for t in possible_targets:
        if t in df.columns:
            df = df.drop(columns=[t])
            notifications.append(f"Target column '{t}' detected and ignored during prediction.")

    numeric_cols = schema["numeric_features"]
    categorical_cols = schema["categorical_features"]

    # Check for missing required features
    all_required = numeric_cols + categorical_cols
    missing_cols = [col for col in all_required if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns in dataset: {', '.join(missing_cols)}.\n"
            f"Expected numeric features: {', '.join(numeric_cols)}\n"
            f"Expected categorical features: {', '.join(categorical_cols)}"
        )

    # Cast numeric columns to float
    for col in numeric_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            raise ValueError(f"Could not convert numeric column '{col}' to numeric: {str(e)}")

    if df[numeric_cols].isnull().any().any():
        null_counts = df[numeric_cols].isnull().sum()
        bad_cols = null_counts[null_counts > 0].to_dict()
        notifications.append(f"Warning: Found missing values in numeric columns: {bad_cols}. Imputing median.")
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Cast categorical columns to string
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    return df, notifications

def predict_batch(df: pd.DataFrame, artifacts_dir=ARTIFACTS_DIR):
    """
    Takes raw input DataFrame, validates schema, applies preprocessing,
    and returns DataFrame augmented with 'Conversion Probability' and 'Intent Tier'.
    """
    model = load_model(artifacts_dir)
    encoder = load_encoder(artifacts_dir)
    scaler = load_scaler(artifacts_dir)
    metrics = load_metrics(artifacts_dir)
    schema = load_schema(artifacts_dir)

    # Validate and clean input
    clean_df, notifications = validate_and_prepare_input(df, schema)

    numeric_cols = schema["numeric_features"]
    categorical_cols = schema["categorical_features"]

    # Preprocessing
    X_num = scaler.transform(clean_df[numeric_cols])
    X_cat = encoder.transform(clean_df[categorical_cols])
    X_proc = np.hstack([X_num, X_cat])

    # Inference: Conversion Probability
    probs = model.predict_proba(X_proc)[:, 1]

    # Assign Intent Tiers using FIXED holdout model score cutoffs from metrics.json
    high_cutoff = metrics.get("high_intent_cutoff", 0.7281)
    medium_cutoff = metrics.get("medium_intent_cutoff", 0.5164)

    def assign_tier(p):
        if p >= high_cutoff:
            return "High Intent"
        elif p >= medium_cutoff:
            return "Medium Intent"
        else:
            return "Low Intent"

    tiers = [assign_tier(p) for p in probs]

    # Build result dataframe
    result_df = df.copy()
    result_df["Conversion Probability"] = np.round(probs, 4)
    result_df["Intent Tier"] = tiers

    return result_df, notifications
