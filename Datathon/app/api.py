"""
Enterprise Prediction REST API (FastAPI)
Provides production microservice endpoints for real-time inference,
counterfactual simulation, model governance metrics, and health checks.
Interactive Swagger documentation available at: http://localhost:8000/docs
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

# Ensure app directory is on path
_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Internal prediction utilities
from rag_copilot import RAGExecutiveCopilot
from evaluation_engine import predict_evaluation_dataset, build_evaluation_facts
from predict import (
    load_model, load_encoder, load_scaler, load_metrics, load_schema,
    predict_batch
)

app = FastAPI(
    title="Retail Purchase & Conversion Intelligence API",
    description="Enterprise Machine Learning Microservice for Real-Time Purchase Intent Prediction & Counterfactual Simulation",
    version="1.0.0"
)

# ── Pydantic Request Models ──
class ShopperSession(BaseModel):
    Recency: float = Field(..., description="Days since last transaction (e.g. 15.0)")
    Frequency_log: float = Field(..., description="Log1p of purchase frequency (e.g. 2.19)")
    Monetary_log: float = Field(..., description="Log1p of total lifetime spend (e.g. 7.31)")
    DistinctProducts_log: float = Field(..., description="Log1p of distinct products viewed (e.g. 4.12)")
    Tenure: float = Field(..., description="Customer relationship tenure in days (e.g. 380.0)")
    CountryBucket: str = Field(default="United Kingdom", description="Country/Region code")

class BatchPredictionRequest(BaseModel):
    sessions: List[ShopperSession]

class LabeledShopperSession(ShopperSession):
    Revenue: int = Field(..., description="Observed outcome: 1 for purchase, 0 for no purchase")

class EvaluationRequest(BaseModel):
    sessions: List[LabeledShopperSession]


class CopilotChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[Dict[str, str]]] = None
    provider: Optional[str] = "auto"
    api_key: Optional[str] = None
    dataset_mode: Optional[str] = "prediction"

class CounterfactualSimulationRequest(BaseModel):
    baseline_session: ShopperSession
    simulated_recency: Optional[float] = Field(None, description="Proposed new recency after intervention")
    simulated_frequency_log: Optional[float] = Field(None, description="Proposed new frequency")
    simulated_monetary_log: Optional[float] = Field(None, description="Proposed new spend")

# ── Endpoints ──

@app.get("/health", tags=["System"])
def health_check():
    """Returns microservice operational status and artifact readiness."""
    try:
        metrics = load_metrics()
        schema = load_schema()
        return {
            "status": "healthy",
            "model_type": schema.get("model_type", "RandomForestClassifier"),
            "target": schema.get("target", "Revenue"),
            "holdout_roc_auc": metrics.get("roc_auc", 0.7925),
            "holdout_f1": metrics.get("f1", 0.7037)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Artifacts not ready: {str(e)}")

@app.get("/metrics", tags=["Governance"])
def get_metrics():
    """Returns holdout performance metrics, fixed cutoffs, and feature importances."""
    try:
        return load_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema", tags=["Governance"])
def get_schema():
    """Returns the formal input feature contract."""
    try:
        return load_schema()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", tags=["Inference"])
def predict_sessions(request: BatchPredictionRequest):
    """
    Scores an array of shopper sessions and returns Conversion Probabilities and Intent Tiers.
    """
    try:
        records = [s.model_dump() for s in request.sessions]
        input_df = pd.DataFrame(records)
        scored_df, notes = predict_batch(input_df)

        results = []
        for _, row in scored_df.iterrows():
            prob = float(row["Conversion Probability"])
            tier = str(row["Intent Tier"])
            results.append({
                "Conversion Probability": prob,
                "probability": prob,
                "Intent Tier": tier,
                "intent_tier": tier,
                "Session Features": {
                    "Recency": float(row["Recency"]),
                    "Frequency_log": float(row["Frequency_log"]),
                    "Monetary_log": float(row["Monetary_log"]),
                    "CountryBucket": str(row["CountryBucket"])
                }
            })

        return {
            "total_scored": len(results),
            "predictions": results,
            "notifications": notes
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

@app.post("/evaluate", tags=["Evaluation Mode"])
def evaluate_sessions(request: EvaluationRequest):
    """
    Evaluates an array of labeled sessions against ground-truth Revenue outcomes.
    Returns confusion matrix, accuracy, precision, recall, F1, ROC-AUC, Brier score,
    and individual prediction result classifications.
    """
    try:
        from evaluation_engine import predict_evaluation_dataset
        records = [s.model_dump() for s in request.sessions]
        input_df = pd.DataFrame(records)
        evaluated_df, summary = predict_evaluation_dataset(input_df)

        results = []
        for _, row in evaluated_df.iterrows():
            prob = float(row["Conversion Probability"])
            tier = str(row["Intent Tier"])
            results.append({
                "Conversion Probability": prob,
                "probability": prob,
                "Intent Tier": tier,
                "intent_tier": tier,
                "Actual Outcome": int(row["Actual Outcome"]),
                "Predicted Outcome": int(row["Predicted Outcome"]),
                "Prediction Result": str(row["Prediction Result"])
            })

        return {
            "total_evaluated": len(results),
            "metrics": summary["metrics"],
            "confusion_matrix": summary["confusion_matrix"],
            "error_analysis": summary["error_analysis"],
            "evaluations": results,
            "scored_sessions": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Evaluation error: {str(e)}")

@app.post("/simulate", tags=["Prescriptive Intelligence"])
def simulate_intervention(request: CounterfactualSimulationRequest):
    """
    Performs counterfactual 'What-If' analysis by testing proposed behavioral shifts.
    """
    try:
        base_dict = request.baseline_session.model_dump()
        sim_dict = base_dict.copy()

        if request.simulated_recency is not None:
            sim_dict["Recency"] = request.simulated_recency
        if request.simulated_frequency_log is not None:
            sim_dict["Frequency_log"] = request.simulated_frequency_log
        if request.simulated_monetary_log is not None:
            sim_dict["Monetary_log"] = request.simulated_monetary_log

        compare_df = pd.DataFrame([base_dict, sim_dict])
        scored_df, _ = predict_batch(compare_df)

        p_base = float(scored_df.iloc[0]["Conversion Probability"])
        t_base = str(scored_df.iloc[0]["Intent Tier"])

        p_sim = float(scored_df.iloc[1]["Conversion Probability"])
        t_sim = str(scored_df.iloc[1]["Intent Tier"])

        delta_prob = round(p_sim - p_base, 4)

        return {
            "baseline": {"probability": p_base, "intent_tier": t_base},
            "simulated": {"probability": p_sim, "intent_tier": t_sim},
            "delta_probability": delta_prob,
            "impact_summary": f"Probability shifted by {delta_prob:+.1%} ({t_base} -> {t_sim})"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Simulation error: {str(e)}")


@app.post("/copilot/chat", tags=["Conversational Intelligence"])
def copilot_chat(request: CopilotChatRequest):
    """
    Executive AI Copilot endpoint powered by Google Gemini and DeepSeek LLMs.
    Strictly grounded in deterministic cohort quantiles and evaluation facts.
    """
    try:
        base_dir = os.path.dirname(__file__)
        metrics = load_metrics()
        schema = load_schema()

        if request.dataset_mode == "evaluation":
            sample_path = os.path.join(base_dir, "data", "sample_evaluation_sessions.csv")
            if os.path.exists(sample_path):
                df = pd.read_csv(sample_path)
                preds_df, eval_summary = predict_evaluation_dataset(df)
                eval_facts = build_evaluation_facts(preds_df, eval_summary)
            else:
                preds_df, _ = predict_batch(pd.DataFrame([]))
                eval_facts = None
        else:
            sample_path = os.path.join(base_dir, "data", "sample_sessions.csv")
            if os.path.exists(sample_path):
                df = pd.read_csv(sample_path)
                preds_df, _ = predict_batch(df)
            else:
                preds_df = pd.DataFrame([])
            eval_facts = None

        copilot = RAGExecutiveCopilot(preds_df, metrics, schema, evaluation_facts=eval_facts)
        result = copilot.query(
            question=request.message,
            chat_history=request.chat_history,
            provider=request.provider or "auto",
            api_key=request.api_key
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Copilot inference error: {str(e)}")


@app.get("/datasets/sample", tags=["Data Engine"])
def get_sample_dataset(mode: str = Query("prediction", description="Either 'prediction' or 'evaluation'")):
    """Returns the preloaded standard sample cohort sessions."""
    base_dir = os.path.dirname(__file__)
    if mode == "evaluation":
        path = os.path.join(base_dir, "data", "sample_evaluation_sessions.csv")
    else:
        path = os.path.join(base_dir, "data", "sample_sessions.csv")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Sample dataset for mode '{mode}' not found.")

    df = pd.read_csv(path)
    records = df.to_dict(orient="records")
    return {
        "mode": mode,
        "filename": os.path.basename(path),
        "total_records": len(df),
        "columns": list(df.columns),
        "sessions": records,
        "data": records
    }

# ── Mount Modern Commercial Web Application ──
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", tags=["Commercial Frontend"])
def serve_dashboard():
    """Serves the primary Commercial SaaS Web Application."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Retail Intelligence Platform API is healthy. Web UI will be mounted shortly."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
