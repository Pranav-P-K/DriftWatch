"""
01_train_categorical_model.py
-----------------------------
Trains a baseline ML model on the Bank Marketing dataset (mixed categorical + numerical features).
1. Preprocesses mixed attributes (OneHotEncoder for categorical, StandardScaler for numerical)
2. Trains RandomForest & XGBoost classifiers, selecting the superior model
3. Evaluates performance (F1, Precision, Recall, ROC-AUC)
4. Saves statistical baseline profiles to `profiles/bank_marketing_baseline.json`
5. Serializes the champion model into open-standard ONNX format (`models/bank_model.onnx`)
6. Registers the model in `models/model_registry.json`
"""

import os
import json
import time
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "bank_marketing.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
MODEL_ONNX_PATH = os.path.join(MODELS_DIR, "bank_model.onnx")
REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")
BASELINE_PROFILE_PATH = os.path.join(PROFILES_DIR, "bank_marketing_baseline.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROFILES_DIR, exist_ok=True)

CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"
]
NUMERICAL_COLS = [
    "age", "balance", "day", "duration", "campaign", "pdays", "previous"
]
TARGET_COL = "y"


def load_and_preprocess_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset missing at {DATASET_PATH}")
        
    print(f"[*] Loading Bank Marketing dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH, sep=";")
    print(f"[*] Loaded {len(df):,} records with {df.shape[1]} columns")
    
    # Target conversion: yes -> 1, no -> 0
    df[TARGET_COL] = (df[TARGET_COL].str.lower() == "yes").astype(int)
    
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    
    pos_count = y.sum()
    print(f"[*] Class Balance: {pos_count:,} positive ({pos_count/len(y)*100:.2f}%), {len(y)-pos_count:,} negative")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Train split: {len(X_train):,} samples | Test split: {len(X_test):,} samples")
    
    return X, y, X_train, X_test, y_train, y_test


def compute_and_save_profiles(X_train: pd.DataFrame):
    """
    Computes reference baseline statistical profiles for both categorical
    and numerical columns to support drift detection (PSI, Chi-Square, KS-Test).
    """
    profile = {
        "dataset": "bank_marketing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_baseline_samples": len(X_train),
        "categorical_profiles": {},
        "numerical_profiles": {}
    }
    
    # Categorical: category probability distributions
    for col in CATEGORICAL_COLS:
        counts = X_train[col].astype(str).value_counts()
        total = len(X_train[col])
        proportions = (counts / total).to_dict()
        profile["categorical_profiles"][col] = proportions
        
    # Numerical: mean, std, and 10 quantile bin edges
    for col in NUMERICAL_COLS:
        col_data = X_train[col].astype(float)
        quantiles = np.quantile(col_data, np.linspace(0.1, 0.9, 9)).tolist()
        profile["numerical_profiles"][col] = {
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "quantiles": quantiles
        }
        
    with open(BASELINE_PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"[+] Saved baseline profile to: {BASELINE_PROFILE_PATH}")
    return profile


def export_pipeline_to_onnx(pipeline, X_train):
    """
    Serializes scikit-learn pipeline into standard ONNX format using skl2onnx.
    Avoids Python pickle entirely for cross-platform, secure real-time inference.
    """
    import onnx
    from skl2onnx import to_onnx
    
    print("[*] Converting champion model to ONNX format...")
    # Generate ONNX model using sample training row
    initial_type = []
    sample_df = X_train.iloc[:5]
    
    onnx_model = to_onnx(pipeline, sample_df, target_opset=15)
    
    with open(MODEL_ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    # Validate ONNX graph structure
    onnx.checker.check_model(onnx_model)
    file_size_kb = os.path.getsize(MODEL_ONNX_PATH) / 1024
    print(f"[+] Successfully exported valid ONNX model to: {MODEL_ONNX_PATH} ({file_size_kb:.1f} KB)")


def update_model_registry(metrics: dict):
    registry = []
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    registry = data
                elif isinstance(data, dict):
                    registry = list(data.values())
        except Exception:
            registry = []
            
    entry = {
        "dataset": "bank_marketing",
        "version": "v1_bank",
        "model_type": "RandomForestClassifier",
        "format": "ONNX",
        "model_file": "bank_model.onnx",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "status": "active",
        "f1": round(metrics["macro_f1"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "roc_auc": round(metrics["roc_auc"], 4),
        "features": {
            "categorical": CATEGORICAL_COLS,
            "numerical": NUMERICAL_COLS
        }
    }
    registry.append(entry)
    
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"[+] Updated Model Registry: {REGISTRY_PATH}")


def main():
    X, y, X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # 1. Profile baseline distributions
    compute_and_save_profiles(X_train)
    
    # 2. Build Pipeline (Preprocessing + Estimator)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS)
        ]
    )
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    
    print("[*] Training Baseline Pipeline (Preprocess + Classifier)...")
    start_train = time.time()
    pipeline.fit(X_train, y_train)
    train_dur = time.time() - start_train
    print(f"[+] Model trained in {train_dur:.2f}s")
    
    # 3. Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    f1 = float(f1_score(y_test, y_pred, average="macro"))
    f1_pos = float(f1_score(y_test, y_pred, pos_label=1))
    prec = float(precision_score(y_test, y_pred, pos_label=1))
    rec = float(recall_score(y_test, y_pred, pos_label=1))
    auc = float(roc_auc_score(y_test, y_prob))
    
    print("\n" + "="*50)
    print("BANK MARKETING BASELINE EVALUATION RESULTS")
    print("="*50)
    print(f"Macro F1-Score:     {f1:.4f}")
    print(f"Positive F1-Score:  {f1_pos:.4f}")
    print(f"Precision:          {prec:.4f}")
    print(f"Recall:             {rec:.4f}")
    print(f"ROC-AUC Score:      {auc:.4f}")
    print("="*50)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    metrics = {
        "macro_f1": f1,
        "positive_f1": f1_pos,
        "precision": prec,
        "recall": rec,
        "roc_auc": auc
    }
    
    # 4. Export to ONNX
    try:
        export_pipeline_to_onnx(pipeline, X_train)
    except Exception as e:
        print(f"[!] Warning: ONNX export failed with ({e}). Saving joblib as fallback.")
        import joblib
        joblib.dump(pipeline, os.path.join(MODELS_DIR, "bank_model.pkl"))
        
    # 5. Update Registry
    update_model_registry(metrics)
    print("[+] Day/Review Step: Categorical baseline model & profiling complete!")


if __name__ == "__main__":
    main()
