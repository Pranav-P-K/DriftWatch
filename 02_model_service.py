"""
02_model_service.py
-------------------
High-Performance Real-Time Model Serving Microservice (FastAPI + ONNX Runtime).
Provides a modern, secure, and lightning-fast alternative to Python pickle file deserialization:
1. Serves models via ONNX Runtime C++ engine (sub-millisecond inference).
2. Endpoints:
   - GET  /health   -> Check service and active model statuses
   - POST /predict  -> Batch or single real-time scoring
   - POST /reload   -> Zero-downtime hot-reloading when retraining worker promotes a new model
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")

app = FastAPI(
    title="DriftWatch Real-Time Model Inference Service",
    description="Zero-downtime ONNX Runtime Model Serving with Dynamic Hot-Reloading",
    version="2.0.0"
)

# Active in-memory model sessions
ACTIVE_SESSIONS = {}
FALLBACK_MODELS = {}


class InferenceRequest(BaseModel):
    model_name: str = "bank_marketing"
    records: List[Dict[str, Any]]


class ReloadRequest(BaseModel):
    model_name: str
    model_file: Optional[str] = None


def load_model_session(model_name: str, model_filename: Optional[str] = None):
    """Loads an ONNX session, with fallback to joblib if ONNX unavailable."""
    if not model_filename:
        if model_name == "bank_marketing":
            model_filename = "bank_model.onnx"
        elif model_name == "creditcard":
            model_filename = "baseline_model.onnx"
        else:
            model_filename = f"{model_name}.onnx"
            
    onnx_path = os.path.join(MODELS_DIR, model_filename)
    pkl_path = os.path.join(MODELS_DIR, model_filename.replace(".onnx", ".pkl"))
    
    # Try ONNX Runtime first (Primary recommended approach)
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            session = ort.InferenceSession(onnx_path, sess_options)
            ACTIVE_SESSIONS[model_name] = {
                "format": "ONNX",
                "session": session,
                "path": onnx_path,
                "loaded_at": time.time()
            }
            print(f"[+] Loaded ONNX model session: {model_name} from {onnx_path}")
            return True
        except Exception as e:
            print(f"[!] Warning: ONNX load failed ({e}), checking fallback...")
            
    # Fallback to joblib if ONNX file is not yet generated
    if os.path.exists(pkl_path):
        try:
            import joblib
            model = joblib.load(pkl_path)
            FALLBACK_MODELS[model_name] = {
                "format": "JOBLIB",
                "model": model,
                "path": pkl_path,
                "loaded_at": time.time()
            }
            print(f"[+] Loaded fallback Joblib model: {model_name} from {pkl_path}")
            return True
        except Exception as e:
            print(f"[!] Error: Joblib load failed ({e})")
            
    print(f"[-] No model found for {model_name} at {onnx_path} or {pkl_path}")
    return False


@app.on_event("startup")
def startup_event():
    print("[*] Initializing DriftWatch Model Serving Service...")
    load_model_session("bank_marketing")
    load_model_session("creditcard")

# Also initialize immediately on module import
try:
    load_model_session("bank_marketing")
    load_model_session("creditcard")
except Exception as e:
    print(f"[!] Init note: {e}")


@app.get("/health")
def health_check():
    onnx_models = list(ACTIVE_SESSIONS.keys())
    fallback_models = list(FALLBACK_MODELS.keys())
    return {
        "status": "UP",
        "service": "DriftWatch-RealTime-Serving",
        "onnx_models_active": onnx_models,
        "fallback_models_active": fallback_models,
        "total_active_models": len(onnx_models) + len(fallback_models)
    }


@app.post("/reload")
def reload_model(req: ReloadRequest):
    """
    Dynamic zero-downtime hot-reloading endpoint.
    Called by 05_retrain_job.py when a new model is trained and registered.
    """
    success = load_model_session(req.model_name, req.model_file)
    if success:
        return {
            "status": "SUCCESS",
            "message": f"Model '{req.model_name}' successfully hot-reloaded in memory.",
            "format": ACTIVE_SESSIONS.get(req.model_name, {}).get("format") or FALLBACK_MODELS.get(req.model_name, {}).get("format")
        }
    raise HTTPException(status_code=404, detail=f"Failed to reload model '{req.model_name}'. Check file path.")


@app.post("/predict")
def predict(req: InferenceRequest):
    """
    Direct low-latency inference endpoint.
    Accepts records as a list of feature dictionaries.
    """
    start_time = time.perf_counter()
    model_name = req.model_name
    records = req.records
    
    if not records:
        return {"predictions": [], "probabilities": [], "latency_ms": 0.0}
        
    # Check if active ONNX session exists
    if model_name in ACTIVE_SESSIONS:
        sess_info = ACTIVE_SESSIONS[model_name]
        session = sess_info["session"]
        
        # Convert records to DataFrame/Dict format expected by ONNX
        df = pd.DataFrame(records)
        
        input_name = session.get_inputs()[0].name
        
        # If single input tensor
        if len(session.get_inputs()) == 1:
            try:
                # Numerical matrix input
                input_data = df.to_numpy(dtype=np.float32)
                ort_inputs = {input_name: input_data}
            except Exception:
                # Dict input for pipelines with named columns
                ort_inputs = {col: df[[col]].to_numpy() for col in df.columns}
        else:
            ort_inputs = {inp.name: df[[inp.name]].to_numpy() for inp in session.get_inputs()}
            
        ort_outputs = session.run(None, ort_inputs)
        preds = ort_outputs[0].tolist()
        
        # Probabilities if available in output[1]
        probs = []
        if len(ort_outputs) > 1:
            raw_probs = ort_outputs[1]
            if isinstance(raw_probs, list) and len(raw_probs) > 0 and isinstance(raw_probs[0], dict):
                probs = [p.get(1, p.get(preds[i], 0.0)) for i, p in enumerate(raw_probs)]
            elif isinstance(raw_probs, np.ndarray):
                probs = raw_probs[:, 1].tolist() if raw_probs.ndim == 2 else raw_probs.tolist()
                
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "model": model_name,
            "serving_engine": "ONNX Runtime (C++)",
            "predictions": preds,
            "probabilities": probs,
            "count": len(records),
            "latency_ms": round(latency_ms, 3)
        }
        
    # Fallback to in-memory Scikit-learn estimator if ONNX not present
    elif model_name in FALLBACK_MODELS:
        model = FALLBACK_MODELS[model_name]["model"]
        df = pd.DataFrame(records)
        preds = model.predict(df).tolist()
        probs = []
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[:, 1].tolist()
            
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "model": model_name,
            "serving_engine": "In-Memory Estimator",
            "predictions": preds,
            "probabilities": probs,
            "count": len(records),
            "latency_ms": round(latency_ms, 3)
        }
        
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not loaded. Active models: {list(ACTIVE_SESSIONS.keys()) + list(FALLBACK_MODELS.keys())}"
        )


if __name__ == "__main__":
    import uvicorn
    print("[*] Starting DriftWatch Model Serving on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
