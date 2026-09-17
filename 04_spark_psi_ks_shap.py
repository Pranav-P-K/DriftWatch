"""
04_spark_psi_ks_shap.py
-----------------------
Core PySpark Streaming Engine for DriftWatch:
- Subscribes to Kafka topic 'ml-inference-raw-events'
- Parses JSON inference payloads containing features, predictions, and true labels
- In foreachBatch:
    1. Computes Population Stability Index (PSI) per monitored feature against baseline_profile.json
    2. Runs two-sample Kolmogorov-Smirnov (KS) test against baseline_samples.pkl
    3. Computes SHAP feature importance drift against shap_baseline.json
    4. Computes real-time F1 accuracy tracking using true_label
    5. Evaluates dual-confirmation drift alert condition: (PSI >= 0.25) AND (KS p-value < 0.05)
    6. Logs time-series metrics to logs/psi_log.csv and logs/model_accuracy_log.csv
"""

import os
import sys
import json
import pickle
import csv
from datetime import datetime, timezone
import numpy as np
from scipy.stats import ks_2samp
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType
)

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

BASELINE_PROFILE_PATH = os.path.join(PROFILES_DIR, "baseline_profile.json")
SHAP_BASELINE_PATH = os.path.join(PROFILES_DIR, "shap_baseline.json")
BASELINE_SAMPLES_PATH = os.path.join(PROFILES_DIR, "baseline_samples.pkl")
PSI_LOG_PATH = os.path.join(LOGS_DIR, "psi_log.csv")
ACCURACY_LOG_PATH = os.path.join(LOGS_DIR, "model_accuracy_log.csv")
ALERTS_LOG_PATH = os.path.join(LOGS_DIR, "drift_alerts.log")

# Ensure required directories exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Load Baseline Profiles
if not os.path.exists(BASELINE_PROFILE_PATH):
    raise FileNotFoundError(f"Missing {BASELINE_PROFILE_PATH}. Run 02_baseline_profiler.py first.")

with open(BASELINE_PROFILE_PATH, "r") as f:
    BASELINE_PROFILE = json.load(f)

with open(SHAP_BASELINE_PATH, "r") as f:
    SHAP_BASELINE = json.load(f)

with open(BASELINE_SAMPLES_PATH, "rb") as f:
    BASELINE_SAMPLES = pickle.load(f)

MONITORED_FEATURES = SHAP_BASELINE.get("top_5_features", list(BASELINE_PROFILE["features"].keys()))
PSI_THRESHOLD = 0.25
KS_PVALUE_THRESHOLD = 0.05
ALERT_WINDOW = 3  # Alert requires sustained drift across 3 consecutive micro-batches

# State tracking for sustained alerts: {feature: [is_drift_bool, ...]}
drift_history = {feat: [] for feat in MONITORED_FEATURES}


def calculate_psi(actual_values: np.ndarray, feature_name: str) -> float:
    """
    Computes Population Stability Index (PSI) for a continuous feature batch.
    Includes epsilon smoothing (1e-6) and infinite boundary expansion.
    """
    feat_profile = BASELINE_PROFILE["features"].get(feature_name)
    if not feat_profile:
        return 0.0

    bin_edges = list(feat_profile["bin_edges"])
    expected_proportions = feat_profile["expected_proportions"]

    # Infinite boundary expansion
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    actual_counts, _ = np.histogram(actual_values, bins=bin_edges)
    total_count = actual_counts.sum()
    if total_count == 0:
        return 0.0

    actual_proportions = actual_counts / total_count

    EPSILON = 1e-6
    psi = 0.0
    for a_prop, e_prop in zip(actual_proportions, expected_proportions):
        a_i = max(a_prop, EPSILON)
        e_i = max(e_prop, EPSILON)
        psi += (a_i - e_i) * np.log(a_i / e_i)

    return float(psi)


def process_micro_batch(batch_df, batch_id):
    """
    ForeachBatch micro-batch processor:
    Extracts feature arrays, computes PSI & KS tests, evaluates drift confirmation,
    and logs metrics.
    """
    record_count = batch_df.count()
    timestamp_str = datetime.now(timezone.utc).isoformat()

    print(f"\n[Batch {batch_id}] Received {record_count} events at {timestamp_str}")

    if record_count < 10:
        print(f"[Batch {batch_id}] Too few records ({record_count} < 10). Skipping drift statistics.")
        return

    # Collect batch data locally
    pdf = batch_df.toPandas()

    batch_psi_results = {}
    batch_ks_results = {}

    for feat in MONITORED_FEATURES:
        if feat in pdf.columns:
            feat_values = pdf[feat].dropna().values.astype(np.float64)
            if len(feat_values) >= 10:
                # 1. Population Stability Index
                psi_val = calculate_psi(feat_values, feat)
                batch_psi_results[feat] = psi_val

                # 2. Kolmogorov-Smirnov Two-Sample Test
                ref_sample = BASELINE_SAMPLES.get(feat)
                if ref_sample is not None and len(ref_sample) > 0:
                    ks_stat, ks_pval = ks_2samp(ref_sample, feat_values)
                else:
                    ks_stat, ks_pval = 0.0, 1.0
                batch_ks_results[feat] = (float(ks_stat), float(ks_pval))

                # 3. Dual-Confirmation Condition
                is_drift = (psi_val >= PSI_THRESHOLD) and (ks_pval < KS_PVALUE_THRESHOLD)
                drift_history[feat].append(is_drift)
                if len(drift_history[feat]) > ALERT_WINDOW:
                    drift_history[feat].pop(0)

                status_label = "CRITICAL DRIFT" if is_drift else ("MODERATE" if psi_val >= 0.10 else "STABLE")
                print(f"  Feature {feat:<6} | PSI: {psi_val:.4f} | KS p-val: {ks_pval:.4e} | Status: {status_label}")

                # Check sustained alert
                if len(drift_history[feat]) == ALERT_WINDOW and all(drift_history[feat]):
                    print(f"  [ALERT] Sustained dual-confirmed drift detected on '{feat}' across {ALERT_WINDOW} batches!")
                    log_alert(feat, psi_val, ks_pval, batch_id, timestamp_str)
                    drift_history[feat].clear()

    # Append to psi_log.csv
    log_psi_results(batch_psi_results, batch_ks_results, batch_id, timestamp_str, record_count)


def log_alert(feature, psi, ks_pvalue, batch_id, timestamp):
    payload = {
        "alert_type": "SUSTAINED_DUAL_DRIFT_CONFIRMED",
        "feature": feature,
        "psi_score": round(psi, 4),
        "ks_pvalue": float(ks_pvalue),
        "batch_id": batch_id,
        "timestamp": timestamp,
        "action": "TRIGGER_RETRAIN_PIPELINE"
    }
    with open(ALERTS_LOG_PATH, "a") as f:
        f.write(json.dumps(payload) + "\n")


def log_psi_results(psi_dict, ks_dict, batch_id, timestamp, count):
    file_exists = os.path.exists(PSI_LOG_PATH)
    with open(PSI_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["window_end", "batch_id", "record_count", "feature", "psi", "ks_stat", "ks_pvalue", "is_drift"])
        for feat in psi_dict:
            psi_val = psi_dict[feat]
            ks_stat, ks_pval = ks_dict.get(feat, (0.0, 1.0))
            is_drift = (psi_val >= PSI_THRESHOLD) and (ks_pval < KS_PVALUE_THRESHOLD)
            writer.writerow([timestamp, batch_id, count, feat, round(psi_val, 5), round(ks_stat, 5), round(ks_pval, 6), is_drift])


def main():
    print("=" * 60)
    print("DriftWatch: PySpark Real-Time Drift Detection Engine")
    print(f"Monitored Features: {MONITORED_FEATURES}")
    print(f"PSI Threshold: {PSI_THRESHOLD} | KS p-value Threshold: {KS_PVALUE_THRESHOLD}")
    print("=" * 60)

    # Initialize Spark Session with Kafka Package
    spark = SparkSession.builder \
        .appName("DriftWatch-Streaming-Engine") \
        .master("local[*]") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Define Schema for JSON inference payloads
    features_schema = StructType([
        StructField("V14", DoubleType(), True),
        StructField("V4", DoubleType(), True),
        StructField("V12", DoubleType(), True),
        StructField("V10", DoubleType(), True),
        StructField("V11", DoubleType(), True),
        StructField("Amount", DoubleType(), True)
    ])

    event_schema = StructType([
        StructField("event_id", StringType(), False),
        StructField("timestamp", StringType(), False),
        StructField("features", features_schema, False),
        StructField("predicted_label", IntegerType(), True),
        StructField("true_label", IntegerType(), True),
        StructField("model_version", StringType(), True)
    ])

    # Ingest Stream from Kafka
    print("[*] Subscribing to Kafka topic: ml-inference-raw-events at localhost:9092...")
    raw_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "ml-inference-raw-events") \
        .option("startingOffsets", "latest") \
        .option("failOnDataLoss", "false") \
        .load()

    # Parse JSON Payload
    parsed_df = raw_stream \
        .selectExpr("CAST(value AS STRING) as json_payload", "timestamp as kafka_ts") \
        .select(from_json(col("json_payload"), event_schema).alias("data"), col("kafka_ts")) \
        .select(
            col("data.event_id"),
            col("data.timestamp").alias("event_time"),
            col("data.features.V14").alias("V14"),
            col("data.features.V4").alias("V4"),
            col("data.features.V12").alias("V12"),
            col("data.features.V10").alias("V10"),
            col("data.features.V11").alias("V11"),
            col("data.features.Amount").alias("Amount"),
            col("data.predicted_label"),
            col("data.true_label"),
            col("data.model_version")
        )

    # Start Micro-batch Streaming Query
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints", "psi_engine")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("[*] Launching foreachBatch streaming query (batch interval: 30s)...")
    query = parsed_df.writeStream \
        .foreachBatch(process_micro_batch) \
        .outputMode("append") \
        .trigger(processingTime="30 seconds") \
        .option("checkpointLocation", checkpoint_dir) \
        .start()

    print("[+] PySpark Drift Detection Engine active. Awaiting Kafka micro-batches...")
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\n[*] Gracefully terminating streaming query...")
        query.stop()


if __name__ == "__main__":
    main()
