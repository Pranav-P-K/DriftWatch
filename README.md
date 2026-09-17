# 🚀 DriftWatch: Real-Time Distributed Concept Drift Detection Engine

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Spark](https://img.shields.io/badge/Apache%20Spark-3.5-orange.svg)](https://spark.apache.org/)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-7.4-black.svg)](https://kafka.apache.org/)
[![Hadoop](https://img.shields.io/badge/Hadoop%20HDFS-3.2-yellow.svg)](https://hadoop.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io/)

> **Big Data Analytics (VIT SEM 7)**
> A production-grade distributed streaming ML observability and automated closed-loop retraining pipeline detecting concept and feature drift in real-time.

---

## 👥 Team & Roles

| Member | Reg. No. | Role | Responsibilities |
|---|---|---|---|
| **Muthuselvan B** | 23BLC1145 | Data Pipeline Lead | Kafka Producer, Drift Simulation, Baseline Profiling |
| **Sinthana P** | 23BLC1160 | Observability & Viz Lead | Streamlit Real-Time Dashboard, Grafana & InfluxDB Setup |
| **Pranav P K** | 23BLC1268 | MLOps & Spark Engineer | PySpark Structured Streaming, PSI + KS-Test + SHAP Drift |
| **Hasith Anton M A** | 23BLC1312 | Integration & DevOps Lead | Docker Infrastructure, HDFS Sinks, Closed-Loop Retraining |

---

## 🏗️ Architecture Overview

```
Live Traffic Generator (03_kafka_producer.py)
   │ (Simulates normal traffic -> mild drift -> severe drift)
   ▼ (10 events/sec JSON)
Apache Kafka (Topic: ml-inference-raw-events)
   │
   ▼
Apache Spark Structured Streaming (04_spark_psi_ks_shap.py)
   ├──► Hadoop HDFS Data Lake (/lake/raw_inferences/ in Parquet)
   │
   └──► foreachBatch Statistical Engine:
           ├── Population Stability Index (PSI >= 0.25)
           ├── 2-Sample Kolmogorov-Smirnov Test (p < 0.05)
           ├── SHAP Attribution Drift (Top 5 features)
           └── Real-time F1 Accuracy Tracking
                   │
                   ▼ (Alert condition: sustained PSI >= 0.25 & KS p < 0.05)
         Auto-Retraining Pipeline (05_retrain_job.py)
                   │
                   ▼
         Live Streamlit Observability Dashboard (06_dashboard.py)
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup (Python 3.11)

```bash
# Create and activate virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Distributed Infrastructure

Ensure Docker Desktop is running, then:

```bash
docker-compose up -d
```

Services:
- **Zookeeper**: `localhost:2181`
- **Kafka Broker**: `localhost:9092`
- **HDFS NameNode UI**: `http://localhost:9870`
- **HDFS IPC**: `hdfs://localhost:9000`

### 3. Baseline Model & Registry

```bash
python 01_train_model.py
```
- Trains benchmark **RandomForest** vs **XGBoost** on `dataset/creditcard.csv`.
- Saves serialized model artifact to `models/baseline_model.pkl`.
- Initializes `models/model_registry.json`.

---

## 📊 Current Results & Benchmarks

| Milestone | Status | Details |
|---|---|---|
| **Day 1: Infrastructure & Baseline** |  Completed | Docker cluster healthy, Kafka topic created, HDFS initialized, baseline model trained ($F_1 = 0.8457$, Precision $= 0.9610$). |
| **Day 2: Baseline Profiling & SHAP** | ⏳ Scheduled | Extract top 5 SHAP features, decile bins, and baseline samples. |
| **Day 3: Kafka Producer with Drift** | ⏳ Scheduled | 3-phase live traffic simulator. |
| **Day 4: Spark Dual-Drift Engine** | ⏳ Scheduled | Streaming PSI + KS-test dual confirmation. |
| **Day 5: SHAP Drift & F1 Tracking** | ⏳ Scheduled | Feature attribution drift + live accuracy correlation. |
| **Day 6: Closed-Loop Retraining** | ⏳ Scheduled | HDFS Parquet ingestion + automatic retraining proof. |
| **Day 7: Streamlit & Grafana** | ⏳ Scheduled | 6-panel operational observability dashboard. |
| **Day 8: Final Demo & Polish** | ⏳ Scheduled | Screen recordings, documentation, and viva prep. |
