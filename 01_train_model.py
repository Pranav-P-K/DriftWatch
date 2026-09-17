"""
01_train_model.py
-----------------
Trains baseline models (RandomForest vs XGBoost) on the Credit Card Fraud dataset.
Evaluates both on a held-out test split, selects the superior model, saves it to
`models/baseline_model.pkl`, and initializes `models/model_registry.json`.
"""

import os
import json
import time
from datetime import datetime, timezone
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)
from xgboost import XGBClassifier

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "creditcard.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "baseline_model.pkl")
REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")


def load_and_prep_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at expected location: {DATASET_PATH}")

    print(f"[*] Loading dataset from: {DATASET_PATH}")
    start_time = time.time()
    df = pd.read_csv(DATASET_PATH)
    print(f"[*] Loaded {len(df):,} records with {df.shape[1]} columns in {time.time() - start_time:.2f}s")

    # Target & Features
    # Time is usually dropped in static ML models to prevent temporal overfitting
    X = df.drop(columns=["Class", "Time"])
    y = df["Class"]

    fraud_count = y.sum()
    print(f"[*] Target distribution: {fraud_count:,} frauds ({fraud_count / len(y) * 100:.3f}%), {len(y) - fraud_count:,} normal")

    # Stratified split to maintain rare fraud class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

    return X, y, X_train, X_test, y_train, y_test


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    f1 = f1_score(y_test, y_pred, pos_label=1)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    metrics = {
        "model_name": name,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc)
    }

    print(f"\n--- {name} Results ---")
    print(f"  Fraud F1-Score: {f1:.4f}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  ROC-AUC:        {roc_auc:.4f}")
    print(f"  PR-AUC:         {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    return metrics


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "profiles"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

    X, y, X_train, X_test, y_train, y_test = load_and_prep_data()

    # 1. Train Random Forest
    print("\n" + "=" * 50)
    print("[1/2] Training RandomForestClassifier (100 trees)...")
    print("=" * 50)
    rf_start = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print(f"[*] RandomForest trained in {time.time() - rf_start:.2f}s")
    rf_metrics = evaluate_model("RandomForest", rf, X_test, y_test)

    # 2. Train XGBoost
    print("\n" + "=" * 50)
    print("[2/2] Training XGBClassifier...")
    print("=" * 50)
    xgb_start = time.time()
    scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    print(f"[*] XGBoost trained in {time.time() - xgb_start:.2f}s")
    xgb_metrics = evaluate_model("XGBoost", xgb, X_test, y_test)

    # 3. Model Selection
    print("\n" + "=" * 50)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 50)
    print(f"RandomForest -> F1: {rf_metrics['f1']:.4f} | PR-AUC: {rf_metrics['pr_auc']:.4f}")
    print(f"XGBoost      -> F1: {xgb_metrics['f1']:.4f} | PR-AUC: {xgb_metrics['pr_auc']:.4f}")

    if rf_metrics["f1"] >= xgb_metrics["f1"]:
        winner_name = "RandomForest"
        winner_model = rf
        winner_metrics = rf_metrics
    else:
        winner_name = "XGBoost"
        winner_model = xgb
        winner_metrics = xgb_metrics

    print(f"\n[+] Selected Model: {winner_name} (Highest F1 = {winner_metrics['f1']:.4f})")

    # 4. Save Model Artifact
    joblib.dump(winner_model, MODEL_PATH)
    print(f"[+] Saved baseline model to: {MODEL_PATH}")

    # 5. Initialize Model Registry
    registry_entry = {
        "version": "v1",
        "model_type": winner_name,
        "f1": round(winner_metrics["f1"], 4),
        "precision": round(winner_metrics["precision"], 4),
        "recall": round(winner_metrics["recall"], 4),
        "roc_auc": round(winner_metrics["roc_auc"], 4),
        "pr_auc": round(winner_metrics["pr_auc"], 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": list(X.columns),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "status": "active"
    }

    registry = [registry_entry]
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"[+] Initialized model registry at: {REGISTRY_PATH}")

    # 6. Verification
    print("\n" + "=" * 50)
    print("VERIFYING MODEL ARTIFACT")
    print("=" * 50)
    loaded_model = joblib.load(MODEL_PATH)
    sample_rows = X_test.iloc[:5]
    sample_preds = loaded_model.predict(sample_rows)
    sample_probs = loaded_model.predict_proba(sample_rows)[:, 1]
    print(f"Loaded successfully! Sample 5 predictions: {sample_preds}")
    print(f"Sample 5 fraud probabilities: {[round(p, 4) for p in sample_probs]}")
    print("\n[SUCCESS] Day 1 model training complete!")


if __name__ == "__main__":
    main()
