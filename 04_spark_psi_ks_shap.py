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
    6. Logs time-series metrics to:
         - logs/psi_log.csv
         - logs/model_accuracy_log.csv
         - logs/shap_drift_log.csv
         - logs/drift_alerts.log
"""

import os
import sys
import json
import pickle
import csv
import warnings
import subprocess
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import joblib
import shap
from sklearn.metrics import f1_score, accuracy_score

warnings.filterwarnings("ignore", category=UserWarning)

# Ensure environment variables are configured for PySpark & Hadoop on Windows
if "HADOOP_HOME" not in os.environ and os.path.exists(r"c:\hadoop"):
    os.environ["HADOOP_HOME"] = r"c:\hadoop"
    os.environ["PATH"] = os.path.join(r"c:\hadoop", "bin") + os.pathsep + os.environ.get("PATH", "")

if "JAVA_HOME" not in os.environ:
    jdk_path = r"C:\Users\krppr\.jdks\openjdk-21"
    if os.path.exists(jdk_path):
        os.environ["JAVA_HOME"] = jdk_path
        os.environ["PATH"] = os.path.join(jdk_path, "bin") + os.pathsep + os.environ.get("PATH", "")

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

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
LAKE_DIR = os.path.join(BASE_DIR, "lake", "raw_inferences")

BASELINE_PROFILE_PATH = os.path.join(PROFILES_DIR, "baseline_profile.json")
SHAP_BASELINE_PATH = os.path.join(PROFILES_DIR, "shap_baseline.json")
BASELINE_SAMPLES_PATH = os.path.join(PROFILES_DIR, "baseline_samples.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "baseline_model.pkl")

PSI_LOG_PATH = os.path.join(LOGS_DIR, "psi_log.csv")
ACCURACY_LOG_PATH = os.path.join(LOGS_DIR, "model_accuracy_log.csv")
SHAP_DRIFT_LOG_PATH = os.path.join(LOGS_DIR, "shap_drift_log.csv")
ALERTS_LOG_PATH = os.path.join(LOGS_DIR, "drift_alerts.log")

# Ensure required directories exist
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(LAKE_DIR, exist_ok=True)

# Load Baseline Profiles & Artifacts
if not os.path.exists(BASELINE_PROFILE_PATH):
    raise FileNotFoundError(f"Missing {BASELINE_PROFILE_PATH}. Run 02_baseline_profiler.py first.")

with open(BASELINE_PROFILE_PATH, "r") as f:
    BASELINE_PROFILE = json.load(f)

with open(SHAP_BASELINE_PATH, "r") as f:
    SHAP_BASELINE = json.load(f)

with open(BASELINE_SAMPLES_PATH, "rb") as f:
    BASELINE_SAMPLES = pickle.load(f)

print(f"[*] Loading baseline model from {MODEL_PATH} for SHAP evaluation...")
BASELINE_MODEL = joblib.load(MODEL_PATH)
print("[*] Initializing cached SHAP TreeExplainer for streaming batches...")
SHAP_EXPLAINER = shap.TreeExplainer(BASELINE_MODEL)

ALL_FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
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


def compute_batch_shap_drift(pdf):
    """
    Computes SHAP feature attribution drift on a sample of <= 50 rows from the micro-batch.
    Returns: dict of {feature: (baseline_shap, batch_shap, drift_ratio)}
    """
    if len(pdf) == 0:
        return {}

    # Sample max 50 rows for sub-second runtime
    sample_size = min(50, len(pdf))
    sample_df = pdf[ALL_FEATURE_COLS].dropna().sample(n=sample_size, random_state=42)

    shap_output = SHAP_EXPLAINER(sample_df)
    if hasattr(shap_output, "values"):
        vals = shap_output.values
        if vals.ndim == 3:
            vals = vals[:, :, 1]
    elif isinstance(shap_output, list):
        vals = shap_output[1]
    else:
        vals = np.array(shap_output)

    mean_abs_shap = np.abs(vals).mean(axis=0)
    batch_shap_map = {feat: float(mean_abs_shap[idx]) for idx, feat in enumerate(ALL_FEATURE_COLS)}

    baseline_shap_map = SHAP_BASELINE.get("features", {})
    shap_drift_results = {}

    for feat in MONITORED_FEATURES:
        b_val = baseline_shap_map.get(feat, 1e-4)
        c_val = batch_shap_map.get(feat, 0.0)
        drift_ratio = abs(c_val - b_val) / (b_val + 1e-6)
        shap_drift_results[feat] = (b_val, c_val, drift_ratio)

    return shap_drift_results


def process_micro_batch(batch_df, batch_id):
    """
    ForeachBatch micro-batch processor:
    1. Computes PSI and KS two-sample statistics
    2. Computes streaming F1-score & accuracy against true_label
    3. Computes SHAP feature importance drift
    4. Evaluates dual-confirmation sustained alert logic
    5. Logs all metrics to time-series CSVs
    """
    record_count = batch_df.count()
    timestamp_str = datetime.now(timezone.utc).isoformat()

    print(f"\n" + "=" * 65)
    print(f"[Micro-Batch {batch_id}] Ingested {record_count} events at {timestamp_str}")
    print("=" * 65)

    if record_count < 10:
        print(f"[Batch {batch_id}] Record count ({record_count} < 10) below threshold. Skipping.")
        return

    pdf = batch_df.toPandas()

    # 1. Real-Time Model Accuracy Tracking (F1 & Accuracy)
    true_labels = pdf["true_label"].astype(int).values
    pred_labels = pdf["predicted_label"].astype(int).values
    model_version = str(pdf["model_version"].iloc[0]) if "model_version" in pdf.columns else "v1"

    fraud_f1 = float(f1_score(true_labels, pred_labels, pos_label=1, zero_division=0))
    macro_f1 = float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))
    accuracy = float(accuracy_score(true_labels, pred_labels))

    print(f"[*] Live Model Performance -> Fraud F1: {fraud_f1:.4f} | Macro F1: {macro_f1:.4f} | Accuracy: {accuracy*100:.2f}%")
    log_model_accuracy(batch_id, timestamp_str, record_count, model_version, fraud_f1, macro_f1, accuracy)

    # 2. PSI & KS-Test Drift Statistics
    batch_psi_results = {}
    batch_ks_results = {}

    print("[*] Statistical Distribution Shifts:")
    for feat in MONITORED_FEATURES:
        if feat in pdf.columns:
            feat_values = pdf[feat].dropna().values.astype(np.float64)
            if len(feat_values) >= 10:
                # A. PSI
                psi_val = calculate_psi(feat_values, feat)
                batch_psi_results[feat] = psi_val

                # B. KS-Test
                ref_sample = BASELINE_SAMPLES.get(feat)
                if ref_sample is not None and len(ref_sample) > 0:
                    ks_stat, ks_pval = ks_2samp(ref_sample, feat_values)
                else:
                    ks_stat, ks_pval = 0.0, 1.0
                batch_ks_results[feat] = (float(ks_stat), float(ks_pval))

                # C. Dual-Confirmation
                is_drift = (psi_val >= PSI_THRESHOLD) and (ks_pval < KS_PVALUE_THRESHOLD)
                drift_history[feat].append(is_drift)
                if len(drift_history[feat]) > ALERT_WINDOW:
                    drift_history[feat].pop(0)

                status_label = "CRITICAL DRIFT" if is_drift else ("MODERATE" if psi_val >= 0.10 else "STABLE")
                print(f"    {feat:<6} | PSI: {psi_val:7.4f} | KS p-val: {ks_pval:.2e} | Status: {status_label}")

                # Sustained Alert Trigger
                if len(drift_history[feat]) == ALERT_WINDOW and all(drift_history[feat]):
                    print(f"    [!] ALERT TRIGGERED: Sustained dual drift on '{feat}' across {ALERT_WINDOW} batches!")
                    log_alert(feat, psi_val, ks_pval, batch_id, timestamp_str)
                    drift_history[feat].clear()

    log_psi_results(batch_psi_results, batch_ks_results, batch_id, timestamp_str, record_count)

    # 3. SHAP Attribution Drift
    shap_drift_results = compute_batch_shap_drift(pdf)
    print("[*] SHAP Feature Attribution Drift (Top Monitored):")
    for feat, (b_shap, c_shap, d_ratio) in shap_drift_results.items():
        print(f"    {feat:<6} | Base SHAP: {b_shap:.5f} | Batch SHAP: {c_shap:.5f} | Drift: {d_ratio*100:6.1f}%")

    log_shap_drift(batch_id, timestamp_str, shap_drift_results)

    # 4. Sink to Parquet Data Lake and Mirror to HDFS
    sink_to_lake_and_hdfs(pdf, batch_id)


def log_model_accuracy(batch_id, timestamp, count, model_version, fraud_f1, macro_f1, acc):
    file_exists = os.path.exists(ACCURACY_LOG_PATH)
    with open(ACCURACY_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["window_end", "batch_id", "record_count", "model_version", "fraud_f1", "macro_f1", "accuracy"])
        writer.writerow([timestamp, batch_id, count, model_version, round(fraud_f1, 4), round(macro_f1, 4), round(acc, 4)])


def log_shap_drift(batch_id, timestamp, shap_drift_map):
    file_exists = os.path.exists(SHAP_DRIFT_LOG_PATH)
    with open(SHAP_DRIFT_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["window_end", "batch_id", "feature", "baseline_shap", "batch_shap", "drift_ratio"])
        for feat, (b_shap, c_shap, d_ratio) in shap_drift_map.items():
            writer.writerow([timestamp, batch_id, feat, round(b_shap, 6), round(c_shap, 6), round(d_ratio, 4)])


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


RETRAIN_TRIGGERED = False


def trigger_retraining(feature, psi, ks_pvalue, batch_id):
    global RETRAIN_TRIGGERED
    if RETRAIN_TRIGGERED:
        print(f"[*] Retraining already dispatched; skipping duplicate trigger for batch {batch_id}.")
        return
    RETRAIN_TRIGGERED = True
    print("\n" + "!" * 70)
    print(f"[!] TRIGGERING CLOSED-LOOP RETRAINING WORKER")
    print(f"    Cause: Sustained drift on {feature} (PSI={psi:.4f}, KS p={ks_pvalue:.2e})")
    print("    Executing: python 05_retrain_job.py")
    print("!" * 70 + "\n")
    try:
        retrain_script = os.path.join(BASE_DIR, "05_retrain_job.py")
        subprocess.Popen([sys.executable, retrain_script])
    except Exception as e:
        print(f"[!] Failed to launch retraining job: {e}")


def sink_to_lake_and_hdfs(pdf, batch_id):
    try:
        ts = int(datetime.now(timezone.utc).timestamp())
        fname = f"batch_{batch_id}_{ts}.parquet"
        local_path = os.path.join(LAKE_DIR, fname)
        pdf.to_parquet(local_path, index=False)
        print(f"[*] Sunk {len(pdf)} records to Data Lake: {fname}")

        # Mirror to HDFS /lake/raw_inferences/
        cmd = f'docker cp "{local_path}" namenode:/tmp/{fname} && docker exec namenode hdfs dfs -put -f /tmp/{fname} /lake/raw_inferences/ && docker exec namenode rm -f /tmp/{fname}'
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[+] Synced {fname} to HDFS /lake/raw_inferences/")
    except Exception as e:
        print(f"[!] Data lake sink error: {e}")


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
    
    # Trigger automated retraining
    trigger_retraining(feature, psi, ks_pvalue, batch_id)


def main():
    print("=" * 65)
    print("DriftWatch: PySpark Real-Time Drift & Accuracy Engine (Day 5)")
    print(f"Monitored Features: {MONITORED_FEATURES}")
    print(f"PSI Threshold: {PSI_THRESHOLD} | KS p-value Threshold: {KS_PVALUE_THRESHOLD}")
    print("=" * 65)

    # Initialize Spark Session with Kafka Package
    spark = SparkSession.builder \
        .appName("DriftWatch-Streaming-Engine") \
        .master("local[*]") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Define Schema matching full 29 features
    features_schema = StructType([
        StructField(col_name, DoubleType(), True) for col_name in ALL_FEATURE_COLS
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
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss", "false") \
        .load()

    # Parse JSON Payload & unpack all columns
    select_fields = [
        col("data.event_id"),
        col("data.timestamp").alias("event_time")
    ] + [col(f"data.features.{c}").alias(c) for c in ALL_FEATURE_COLS] + [
        col("data.predicted_label"),
        col("data.true_label"),
        col("data.model_version")
    ]

    parsed_df = raw_stream \
        .selectExpr("CAST(value AS STRING) as json_payload", "timestamp as kafka_ts") \
        .select(from_json(col("json_payload"), event_schema).alias("data"), col("kafka_ts")) \
        .select(*select_fields)

    # Start Micro-batch Streaming Query
    checkpoint_dir = os.path.abspath(os.path.join(BASE_DIR, "checkpoints", "psi_engine")).replace("\\", "/")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("[*] Launching foreachBatch streaming query (batch interval: 30s)...")
    query = parsed_df.writeStream \
        .foreachBatch(process_micro_batch) \
        .outputMode("append") \
        .trigger(processingTime="10 seconds") \
        .option("checkpointLocation", checkpoint_dir) \
        .start()

    print("[+] PySpark Engine active. Awaiting Kafka micro-batches...")
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\n[*] Gracefully terminating streaming query...")
        query.stop()


if __name__ == "__main__":
    main()
