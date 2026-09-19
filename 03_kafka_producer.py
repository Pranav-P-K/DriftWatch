"""
03_kafka_producer.py
--------------------
Simulates live inference traffic for DriftWatch with real model predictions:
- Reads unseen transactions from dataset/creditcard.csv using streaming DictReader
- Passes each event through models/baseline_model.pkl to produce real predictions
- Publishes JSON events to Kafka topic 'ml-inference-raw-events'
- Implements 3-Phase Controlled Drift Injection:
    * Phase 1: NORMAL (0 to T) -> Clean test distribution, model performs at baseline
    * Phase 2: MILD DRIFT (T to 2T) -> Moderate shift in Amount and V14
    * Phase 3: SEVERE DRIFT (2T+) -> Multi-feature distribution drift across all Top 5 features
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
import joblib
import numpy as np
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "creditcard.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline_model.pkl")
PROFILES_PATH = os.path.join(BASE_DIR, "profiles", "baseline_profile.json")


def parse_args():
    parser = argparse.ArgumentParser(description="DriftWatch Live Kafka Producer with Drift Simulation")
    parser.add_argument("--bootstrap-servers", type=str, default="localhost:9092", help="Kafka bootstrap broker")
    parser.add_argument("--topic", type=str, default="ml-inference-raw-events", help="Kafka topic name")
    parser.add_argument("--rate", type=float, default=10.0, help="Target events emitted per second (default: 10)")
    parser.add_argument("--phase-seconds", type=float, default=60.0, help="Duration of each drift phase in seconds (default: 60s)")
    parser.add_argument("--max-events", type=int, default=None, help="Stop after N events (default: infinite)")
    return parser.parse_args()


def load_resources(max_records=3000, skip_train=220000):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run 01_train_model.py first.")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}.")

    print(f"[*] Loading baseline model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    print(f"[*] Streaming test records from {DATASET_PATH} (skipping first {skip_train:,} rows)...")
    normal_records = []
    fraud_records = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
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

    # Load feature statistics for drift injection
    feature_stats = {}
    if os.path.exists(PROFILES_PATH):
        with open(PROFILES_PATH, "r") as f:
            profile_data = json.load(f)
            for feat, pdata in profile_data.get("features", {}).items():
                feature_stats[feat] = {
                    "mean": pdata.get("mean", 0.0),
                    "std": pdata.get("std", 1.0)
                }

    # All feature columns in training order
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]

    return model, normal_records, fraud_records, feature_stats, feature_cols


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


def apply_drift(features: dict, phase: str, stats: dict) -> dict:
    """
    Applies synthetic distribution shifts depending on the current phase:
    - NORMAL: no modification
    - MILD DRIFT: shifts Amount and V14
    - SEVERE DRIFT: shifts V14, V4, V12, V10, V11, and Amount
    """
    modified = dict(features)

    if phase == "NORMAL":
        return modified

    elif phase == "MILD DRIFT":
        # Amount: scaled up by 2.5x
        modified["Amount"] = round(max(0.0, float(features["Amount"] * 2.5 + np.random.normal(20, 10))), 2)
        # V14: shift by +0.8 sigma
        std_v14 = stats.get("V14", {}).get("std", 1.0)
        modified["V14"] = float(features["V14"] + 0.8 * std_v14)
        return modified

    elif phase == "SEVERE DRIFT":
        # Amount: explode to luxury/high-ticket range (lognormal distribution)
        modified["Amount"] = round(float(np.random.lognormal(mean=6.5, sigma=1.2)), 2)

        # Severe shifts across all Top 5 features:
        std_v14 = stats.get("V14", {}).get("std", 1.0)
        modified["V14"] = float(features["V14"] - 3.0 * std_v14)

        std_v4 = stats.get("V4", {}).get("std", 1.0)
        modified["V4"] = float(features["V4"] + 2.8 * std_v4)

        std_v12 = stats.get("V12", {}).get("std", 1.0)
        modified["V12"] = float(features["V12"] - 2.6 * std_v12)

        std_v10 = stats.get("V10", {}).get("std", 1.0)
        modified["V10"] = float(features["V10"] - 2.4 * std_v10)

        std_v11 = stats.get("V11", {}).get("std", 1.0)
        modified["V11"] = float(features["V11"] + 2.5 * std_v11)

        return modified

    return modified


def main():
    args = parse_args()
    model, normal_records, fraud_records, stats, feature_cols = load_resources()
    producer = connect_kafka(args.bootstrap_servers)

    interval = 1.0 / args.rate
    p_duration = args.phase_seconds
    start_time = time.time()
    event_count = 0
    record_idx = 0
    fraud_idx = 0
    normal_idx = 0

    print("\n" + "=" * 65)
    print("DriftWatch Live Traffic Generator Initialized")
    print(f"Target Rate:      {args.rate} events/sec (interval: {interval*1000:.0f}ms)")
    print(f"Phase 1 (NORMAL): 0.0s -> {p_duration:.0f}s")
    print(f"Phase 2 (MILD):   {p_duration:.0f}s -> {2*p_duration:.0f}s")
    print(f"Phase 3 (SEVERE): {2*p_duration:.0f}s+")
    print(f"Target Topic:     {args.topic}")
    print("=" * 65 + "\n")

    try:
        while True:
            elapsed = time.time() - start_time

            # Determine drift phase based on elapsed time
            if elapsed < p_duration:
                phase = "NORMAL"
            elif elapsed < 2 * p_duration:
                phase = "MILD DRIFT"
            else:
                phase = "SEVERE DRIFT"

            # Interleave fraud records (approx 10% rate) to enable streaming F1 score tracking
            if record_idx % 10 == 0 and len(fraud_records) > 0:
                row = fraud_records[fraud_idx % len(fraud_records)]
                fraud_idx += 1
            else:
                row = normal_records[normal_idx % len(normal_records)]
                normal_idx += 1
            record_idx += 1

            true_label = int(row["Class"])
            raw_features = {c: float(row[c]) for c in feature_cols}

            # Apply drift transformation
            drifted_features = apply_drift(raw_features, phase, stats)

            # Fast model inference using numpy array
            feature_vector = np.array([[drifted_features[c] for c in feature_cols]], dtype=np.float32)
            pred_label = int(model.predict(feature_vector)[0])

            # Build standardized event payload
            event_payload = {
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "features": {c: round(float(drifted_features[c]), 6) for c in feature_cols},
                "predicted_label": pred_label,
                "true_label": true_label,
                "model_version": "v1"
            }

            producer.send(args.topic, value=event_payload)
            event_count += 1

            # Log periodic status
            if event_count % 50 == 0 or event_count == 1:
                print(f"[{phase:<12}] #{event_count:05d} events sent ({elapsed:5.1f}s) | "
                      f"Amt: ${event_payload['features']['Amount']:>8.2f} | "
                      f"V14: {event_payload['features']['V14']:>6.2f} | "
                      f"Pred: {pred_label} | True: {true_label}")

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
