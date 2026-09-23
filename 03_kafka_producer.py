"""
03_kafka_producer.py
--------------------
Simulates live inference traffic for DriftWatch with real model predictions:
- Supports both Datasets:
    1. 'creditcard' (Numerical features V1-V28, Amount)
    2. 'bank_marketing' (Mixed categorical: job, marital, education, housing, loan + numerical)
- Supports Two Drift Injection Methodologies:
    1. 'natural' (Natural Demographic Subpopulation Partitioning: zero artificial noise)
    2. 'synthetic' (Controlled mathematical distribution perturbation)
- Integrates with Real-Time Model Serving Microservice (HTTP/ONNX) or local ONNX runtime.
- Publishes JSON events to Kafka topic 'ml-inference-raw-events'.
"""

import os
import sys
import csv
import time
import json
import uuid
import argparse
import warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDIT_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "creditcard.csv")
BANK_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "bank_marketing.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")

BANK_CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"
]
BANK_NUMERICAL_COLS = [
    "age", "balance", "day", "duration", "campaign", "pdays", "previous"
]
CREDIT_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def parse_args():
    parser = argparse.ArgumentParser(description="DriftWatch Live Kafka Producer with Multi-Dataset & Drift Simulation")
    parser.add_argument("--dataset", type=str, default="creditcard", choices=["creditcard", "bank_marketing"],
                        help="Target dataset: 'creditcard' (numerical) or 'bank_marketing' (categorical/mixed)")
    parser.add_argument("--drift-mode", type=str, default="natural", choices=["natural", "synthetic"],
                        help="Drift mechanism: 'natural' (demographic subpopulation shift) or 'synthetic' (perturbation)")
    parser.add_argument("--bootstrap-servers", type=str, default="localhost:9092", help="Kafka bootstrap broker")
    parser.add_argument("--topic", type=str, default="ml-inference-raw-events", help="Kafka topic name")
    parser.add_argument("--rate", type=float, default=10.0, help="Target events emitted per second (default: 10)")
    parser.add_argument("--phase-seconds", type=float, default=60.0, help="Duration of each drift phase in seconds (default: 60s)")
    parser.add_argument("--phase", type=str, default=None, choices=["NORMAL", "MILD DRIFT", "SEVERE DRIFT"], help="Force a fixed drift phase")
    parser.add_argument("--max-events", type=int, default=None, help="Stop after N events (default: infinite)")
    parser.add_argument("--use-service", action="store_true", help="Invoke real-time model serving microservice via HTTP")
    parser.add_argument("--service-url", type=str, default="http://127.0.0.1:8000/predict", help="Model serving endpoint")
    return parser.parse_args()


def load_credit_resources(max_records=3000, skip_train=220000):
    if not os.path.exists(CREDIT_DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {CREDIT_DATASET_PATH}.")

    print(f"[*] Loading creditcard dataset (skipping first {skip_train:,} train rows)...")
    normal_records = []
    fraud_records = []
    with open(CREDIT_DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            count += 1
            if count > skip_train:
                if row["Class"] == "1":
                    fraud_records.append(row)
                elif len(normal_records) < max_records:
                    normal_records.append(row)

    print(f"[*] Loaded {len(normal_records):,} normal & {len(fraud_records):,} fraud test records into streaming pool")

    profile_path = os.path.join(PROFILES_DIR, "baseline_profile.json")
    feature_stats = {}
    if os.path.exists(profile_path):
        with open(profile_path, "r") as f:
            profile_data = json.load(f)
            for feat, pdata in profile_data.get("features", {}).items():
                feature_stats[feat] = {
                    "mean": pdata.get("mean", 0.0),
                    "std": pdata.get("std", 1.0)
                }

    return normal_records, fraud_records, feature_stats, CREDIT_COLS


def load_bank_resources():
    if not os.path.exists(BANK_DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {BANK_DATASET_PATH}.")

    print(f"[*] Loading Bank Marketing dataset from {BANK_DATASET_PATH}...")
    df = pd.read_csv(BANK_DATASET_PATH, sep=";")
    df["Class"] = (df["y"].str.lower() == "yes").astype(int)

    # 1. Natural Subpopulations based on demographics
    # Baseline Segment: Salaried / Urban Professionals
    salaried_mask = df["job"].isin(["management", "technician", "admin.", "services"])
    pop_normal = df[salaried_mask].to_dict(orient="records")

    # Mild Drift Segment: Self-Employed / Entrepreneurs / Mixed
    mild_mask = df["job"].isin(["entrepreneur", "self-employed", "housemaid"])
    pop_mild = df[mild_mask].to_dict(orient="records")

    # Severe Drift Segment: Non-Traditional / Retirees / Students / Blue-Collar
    severe_mask = df["job"].isin(["retired", "student", "blue-collar", "unemployed"])
    pop_severe = df[severe_mask].to_dict(orient="records")

    print(f"[*] Bank Marketing Subpopulations (Natural Drift):")
    print(f"    - NORMAL Segment (Salaried/Professional):    {len(pop_normal):,} records")
    print(f"    - MILD DRIFT Segment (Self-Employed/Mixed):   {len(pop_mild):,} records")
    print(f"    - SEVERE DRIFT Segment (Retirees/Students):   {len(pop_severe):,} records")

    all_cols = BANK_NUMERICAL_COLS + BANK_CATEGORICAL_COLS
    return pop_normal, pop_mild, pop_severe, all_cols


def apply_credit_drift(features: dict, phase: str, stats: dict) -> dict:
    modified = dict(features)
    if phase == "NORMAL":
        return modified
    elif phase == "MILD DRIFT":
        modified["Amount"] = round(max(0.0, float(features["Amount"] * 2.5 + np.random.normal(20, 10))), 2)
        std_v14 = stats.get("V14", {}).get("std", 1.0)
        modified["V14"] = float(features["V14"] + 0.8 * std_v14)
        return modified
    elif phase == "SEVERE DRIFT":
        modified["Amount"] = round(float(np.random.lognormal(mean=6.5, sigma=1.2)), 2)
        for feat, shift in [("V14", -3.0), ("V4", 2.8), ("V12", -2.6), ("V10", -2.4), ("V11", 2.5)]:
            std = stats.get(feat, {}).get("std", 1.0)
            modified[feat] = float(features[feat] + shift * std)
        return modified
    return modified


def apply_bank_synthetic_drift(features: dict, phase: str) -> dict:
    modified = dict(features)
    if phase == "NORMAL":
        return modified
    elif phase == "MILD DRIFT":
        modified["duration"] = max(10, int(features["duration"] * 1.8 + np.random.normal(30, 10)))
        modified["age"] = int(np.clip(features["age"] + 8, 18, 95))
        return modified
    elif phase == "SEVERE DRIFT":
        modified["duration"] = max(10, int(features["duration"] * 3.5 + np.random.normal(100, 20)))
        modified["balance"] = float(features["balance"] * 0.2 - 500.0)
        modified["housing"] = "no" if np.random.rand() > 0.3 else "yes"
        modified["contact"] = "cellular"
        return modified
    return modified


def connect_kafka(bootstrap_servers, retries=5, delay=3):
    print(f"[*] Connecting to Kafka broker at {bootstrap_servers}...")
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks=1,
                retries=3,
                linger_ms=10
            )
            print("[+] Successfully connected to Kafka!")
            return producer
        except NoBrokersAvailable:
            print(f"[!] Kafka broker not ready (attempt {attempt}/{retries}). Retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError("Could not connect to Kafka broker. Ensure docker-compose is running.")


def predict_record(dataset_name: str, record_features: dict, use_service: bool, service_url: str, local_session):
    """
    Direct model invocation: either through the real-time model serving microservice (HTTP)
    or via local ONNX Runtime session, eliminating Python pickle deserialization.
    """
    if use_service:
        try:
            resp = requests.post(service_url, json={
                "model_name": dataset_name,
                "records": [record_features]
            }, timeout=1.0)
            if resp.status_code == 200:
                data = resp.json()
                return int(data["predictions"][0]), float(data["probabilities"][0]) if data["probabilities"] else 0.0
        except Exception:
            pass

    # Local ONNX session scoring fallback
    if local_session is not None:
        try:
            df = pd.DataFrame([record_features])
            if len(local_session.get_inputs()) == 1:
                input_data = df.to_numpy(dtype=np.float32)
                ort_inputs = {local_session.get_inputs()[0].name: input_data}
            else:
                ort_inputs = {inp.name: df[[inp.name]].to_numpy() for inp in local_session.get_inputs()}
            outs = local_session.run(None, ort_inputs)
            pred = int(outs[0][0])
            prob = 0.0
            if len(outs) > 1 and isinstance(outs[1], list) and len(outs[1]) > 0:
                prob = float(outs[1][0].get(1, 0.0))
            return pred, prob
        except Exception:
            pass

    return 0, 0.0


def main():
    args = parse_args()
    producer = connect_kafka(args.bootstrap_servers)

    # Initialize ONNX local session
    local_session = None
    try:
        import onnxruntime as ort
        model_filename = "bank_model.onnx" if args.dataset == "bank_marketing" else "baseline_model.onnx"
        model_path = os.path.join(MODELS_DIR, model_filename)
        if os.path.exists(model_path):
            local_session = ort.InferenceSession(model_path)
            print(f"[+] Initialized direct ONNX session from {model_path}")
    except Exception as e:
        print(f"[!] Note: Local ONNX session not initialized ({e})")

    # Dataset Setup
    if args.dataset == "creditcard":
        normal_records, fraud_records, stats, feature_cols = load_credit_resources()
        pop_normal, pop_mild, pop_severe = normal_records, normal_records, normal_records
    else:
        pop_normal, pop_mild, pop_severe, feature_cols = load_bank_resources()
        stats = {}

    interval = 1.0 / args.rate
    p_duration = args.phase_seconds
    start_time = time.time()
    event_count = 0
    idx_normal = 0
    idx_mild = 0
    idx_severe = 0

    print("\n" + "=" * 70)
    print("DriftWatch Live Traffic Generator Initialized")
    print(f"Dataset:          {args.dataset.upper()}")
    print(f"Drift Mode:       {args.drift_mode.upper()} ({'Demographic Partitioning' if args.drift_mode == 'natural' else 'Synthetic Perturbation'})")
    print(f"Inference Mode:   {'HTTP Service (' + args.service_url + ')' if args.use_service else 'Direct ONNX Runtime'}")
    print(f"Target Rate:      {args.rate} events/sec")
    print(f"Phase 1 (NORMAL): 0.0s -> {p_duration:.0f}s")
    print(f"Phase 2 (MILD):   {p_duration:.0f}s -> {2*p_duration:.0f}s")
    print(f"Phase 3 (SEVERE): {2*p_duration:.0f}s+")
    print(f"Target Topic:     {args.topic}")
    print("=" * 70 + "\n")

    try:
        while True:
            elapsed = time.time() - start_time
            if args.phase:
                phase = args.phase
            elif elapsed < p_duration:
                phase = "NORMAL"
            elif elapsed < 2 * p_duration:
                phase = "MILD DRIFT"
            else:
                phase = "SEVERE DRIFT"

            # Select record according to drift mode and phase
            if args.dataset == "creditcard":
                if event_count % 10 == 0 and len(fraud_records) > 0:
                    row = fraud_records[event_count % len(fraud_records)]
                else:
                    row = normal_records[idx_normal % len(normal_records)]
                    idx_normal += 1
                true_label = int(row["Class"])
                raw_features = {c: float(row[c]) for c in feature_cols}
                drifted_features = apply_credit_drift(raw_features, phase, stats)
            else:
                # Bank Marketing dataset
                if args.drift_mode == "natural":
                    # Natural demographic subpopulation partition
                    if phase == "NORMAL":
                        row = pop_normal[idx_normal % len(pop_normal)]
                        idx_normal += 1
                    elif phase == "MILD DRIFT":
                        row = pop_mild[idx_mild % len(pop_mild)]
                        idx_mild += 1
                    else:
                        row = pop_severe[idx_severe % len(pop_severe)]
                        idx_severe += 1
                    true_label = int(row["Class"])
                    drifted_features = {c: row[c] for c in feature_cols}
                else:
                    # Synthetic perturbation mode
                    row = pop_normal[idx_normal % len(pop_normal)]
                    idx_normal += 1
                    true_label = int(row["Class"])
                    raw_features = {c: row[c] for c in feature_cols}
                    drifted_features = apply_bank_synthetic_drift(raw_features, phase)

            # Direct Real-Time Model Inference (ONNX or HTTP Microservice)
            pred_label, pred_prob = predict_record(
                args.dataset, drifted_features, args.use_service, args.service_url, local_session
            )

            event_payload = {
                "event_id": str(uuid.uuid4()),
                "dataset": args.dataset,
                "drift_mode": args.drift_mode,
                "phase": phase,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "features": drifted_features,
                "predicted_label": pred_label,
                "predicted_probability": round(pred_prob, 4),
                "true_label": true_label,
                "model_version": "v1_bank" if args.dataset == "bank_marketing" else "v1"
            }

            producer.send(args.topic, value=event_payload)
            event_count += 1

            if event_count % 50 == 0 or event_count == 1:
                feat_sample = f"job: {drifted_features.get('job', 'N/A')} | age: {drifted_features.get('age', 'N/A')}" if args.dataset == "bank_marketing" else f"Amt: ${drifted_features.get('Amount', 0.0):.2f}"
                print(f"[{phase:<12}] #{event_count:05d} ({elapsed:5.1f}s) | {feat_sample} | Pred: {pred_label} | True: {true_label}")

            if args.max_events and event_count >= args.max_events:
                print(f"\n[*] Reached max events limit ({args.max_events}). Stopping producer.")
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[*] Stopping producer on user interrupt...")
    finally:
        producer.flush()
        producer.close()
        print(f"[+] Producer stopped. Total events emitted: {event_count:,}")


if __name__ == "__main__":
    main()
