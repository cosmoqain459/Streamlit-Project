"""
Evaluation Engine (Dual-Mode Operation)
Handles target isolation, ground-truth outcome benchmarking, confusion matrix calculation,
error profiling (Missed Conversions vs False Alarms), and decision threshold sensitivity.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss, confusion_matrix
)
from predict import predict_batch, load_schema, load_metrics

def is_evaluation_dataset(df: pd.DataFrame) -> bool:
    """Checks if the dataset contains the actual ground-truth Revenue target."""
    target_names = ["Revenue", "revenue", "Churned", "churned"]
    return any(t in df.columns for t in target_names)

def get_target_column_name(df: pd.DataFrame) -> str:
    """Returns the exact name of the target column in the dataframe."""
    for t in ["Revenue", "revenue", "Churned", "churned"]:
        if t in df.columns:
            return t
    return "Revenue"

def prepare_evaluation_data(df: pd.DataFrame, schema: dict) -> Tuple[pd.DataFrame, pd.Series]:
    """
    CRITICAL ISOLATION FUNCTION:
    Extracts the actual outcome target and ensures it NEVER enters the feature matrix.
    Returns: (features_df, actual_target_series)
    """
    target_col = get_target_column_name(df)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in evaluation dataset.")

    # Convert binary outcomes to integer (1 = Purchase, 0 = No Purchase)
    y_raw = df[target_col]
    if y_raw.dtype == bool or y_raw.dtype == object:
        y_true = y_raw.astype(str).str.lower().isin(["true", "1", "purchase", "yes"]).astype(int)
    else:
        y_true = y_raw.fillna(0).astype(int)

    # Strictly drop target from the feature matrix
    X_df = df.drop(columns=[target_col]).copy()
    return X_df, y_true

def predict_evaluation_dataset(df: pd.DataFrame, artifacts_dir=None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs existing production model on the isolated feature set,
    compares predictions against observed Revenue, and classifies every session.
    """
    schema = load_schema(artifacts_dir) if artifacts_dir else load_schema()
    X_df, y_true = prepare_evaluation_data(df, schema)

    # Inference using existing production pipeline
    scored_df, notifications = predict_batch(X_df, artifacts_dir) if artifacts_dir else predict_batch(X_df)

    # Evaluation Classification at 0.50 decision threshold
    probs = scored_df["Conversion Probability"].values
    y_pred = (probs >= 0.50).astype(int)

    # Prediction Result Classification
    results = []
    for actual, pred in zip(y_true, y_pred):
        if actual == 1 and pred == 1:
            results.append("Correct Purchase")
        elif actual == 0 and pred == 0:
            results.append("Correct Non-Purchase")
        elif actual == 0 and pred == 1:
            results.append("False Alarm")
        else:
            results.append("Missed Conversion")

    evaluated_df = scored_df.copy()
    evaluated_df["Actual Outcome"] = y_true.values
    evaluated_df["Actual Label"] = ["Purchased" if y == 1 else "No Purchase" for y in y_true]
    evaluated_df["Predicted Outcome"] = y_pred
    evaluated_df["Predicted Label"] = ["Purchased" if p == 1 else "No Purchase" for p in y_pred]
    evaluated_df["Prediction Result"] = results

    # Compute complete metrics and confusion matrix
    eval_metrics = calculate_evaluation_metrics(y_true.values, y_pred, probs)
    cm_dict = calculate_confusion_matrix(y_true.values, y_pred)
    error_analysis = analyze_errors(evaluated_df)
    threshold_df = calculate_threshold_sensitivity(y_true.values, probs)

    eval_summary = {
        "metrics": eval_metrics,
        "confusion_matrix": cm_dict,
        "error_analysis": error_analysis,
        "threshold_sensitivity": threshold_df,
        "notifications": notifications
    }

    return evaluated_df, eval_summary

def calculate_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Calculates comprehensive evaluation metrics with robust edge-case handling."""
    total = len(y_true)
    if total == 0:
        return {}

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Safe ROC-AUC calculation (requires at least 2 distinct classes)
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5
    else:
        auc = 0.5

    # Probability quality metrics
    brier = float(brier_score_loss(y_true, y_prob))
    try:
        # Clip probabilities to avoid log(0)
        p_clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
        ll = float(log_loss(y_true, p_clipped))
    except Exception:
        ll = 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "total_evaluated": total,
        "actual_conversion_rate": round(float(np.mean(y_true)) * 100, 2)
    }

def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculates confusion matrix components with counts and percentages."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP_pct": float(round(tp / total * 100, 1)) if total > 0 else 0.0,
        "TN_pct": float(round(tn / total * 100, 1)) if total > 0 else 0.0,
        "FP_pct": float(round(fp / total * 100, 1)) if total > 0 else 0.0,
        "FN_pct": float(round(fn / total * 100, 1)) if total > 0 else 0.0,
        "correct_total": int(tp + tn),
        "correct_pct": float(round((tp + tn) / total * 100, 1)) if total > 0 else 0.0,
        "error_total": int(fp + fn),
        "error_pct": float(round((fp + fn) / total * 100, 1)) if total > 0 else 0.0
    }

def calculate_threshold_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
) -> pd.DataFrame:
    """
    Simulates different classification thresholds for exploratory analysis.
    DOES NOT modify production intent thresholds or model weights.
    """
    rows = []
    total = len(y_true)

    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        p = float(precision_score(y_true, preds, zero_division=0))
        r = float(recall_score(y_true, preds, zero_division=0))
        f = float(f1_score(y_true, preds, zero_division=0))
        acc = float(accuracy_score(y_true, preds))
        flagged = int(np.sum(preds))

        rows.append({
            "Threshold": th,
            "Precision": round(p, 3),
            "Recall": round(r, 3),
            "F1-Score": round(f, 3),
            "Accuracy": round(acc, 3),
            "Flagged Purchases": flagged,
            "Flagged %": round(flagged / total * 100, 1) if total > 0 else 0
        })

    return pd.DataFrame(rows)

def analyze_errors(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Profiles behavioral patterns of Missed Conversions (FN) and False Alarms (FP).
    Avoids causal claims; strictly presents comparative observational evidence.
    """
    missed = df[df["Prediction Result"] == "Missed Conversion"]
    alarms = df[df["Prediction Result"] == "False Alarm"]
    correct = df[df["Prediction Result"].isin(["Correct Purchase", "Correct Non-Purchase"])]

    def profile_subset(subset_df):
        if len(subset_df) == 0:
            return {"count": 0, "avg_prob": 0, "avg_recency": 0, "avg_spend": 0, "top_countries": []}

        top_c = subset_df["CountryBucket"].value_counts().head(3).to_dict() if "CountryBucket" in subset_df.columns else {}
        return {
            "count": len(subset_df),
            "avg_prob": round(float(subset_df["Conversion Probability"].mean() * 100), 1),
            "avg_recency": round(float(subset_df["Recency"].mean()), 1) if "Recency" in subset_df.columns else 0,
            "avg_spend": round(float(subset_df["Monetary_log"].mean()), 2) if "Monetary_log" in subset_df.columns else 0,
            "avg_freq": round(float(subset_df["Frequency_log"].mean()), 2) if "Frequency_log" in subset_df.columns else 0,
            "top_countries": top_c
        }

    return {
        "missed_conversions": profile_subset(missed),
        "false_alarms": profile_subset(alarms),
        "correct_predictions": profile_subset(correct)
    }

def build_evaluation_facts(evaluated_df: pd.DataFrame, eval_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Packages deterministic structured evaluation facts to be fed into the AI Copilot.
    """
    m = eval_summary["metrics"]
    cm = eval_summary["confusion_matrix"]
    err = eval_summary["error_analysis"]

    return {
        "evaluation_active": True,
        "total_evaluated": m.get("total_evaluated", 0),
        "accuracy": m.get("accuracy", 0),
        "precision": m.get("precision", 0),
        "recall": m.get("recall", 0),
        "f1_score": m.get("f1", 0),
        "roc_auc": m.get("roc_auc", 0),
        "brier_score": m.get("brier_score", 0),
        "actual_conversion_rate": m.get("actual_conversion_rate", 0),
        "correct_purchases_tp": cm.get("TP", 0),
        "correct_non_purchases_tn": cm.get("TN", 0),
        "false_alarms_fp": cm.get("FP", 0),
        "missed_conversions_fn": cm.get("FN", 0),
        "accuracy_pct": cm.get("correct_pct", 0),
        "missed_conversions_profile": err.get("missed_conversions", {}),
        "false_alarms_profile": err.get("false_alarms", {})
    }
