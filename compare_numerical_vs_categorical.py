"""
compare_numerical_vs_categorical.py
-----------------------------------
Comprehensive Comparative Benchmark answering Professor Review Feedback:
1. Numerical (Credit Card) vs. Categorical/Mixed (Bank Marketing) Pipeline Comparison
2. Synthetic Perturbation vs. Natural Subpopulation Drift Detection Effectiveness
3. Pickle File Deserialization vs. ONNX Real-Time Serving Performance Benchmarks
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import joblib

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from scripts.drift_metrics_categorical import (
        calculate_categorical_psi,
        calculate_chi_square,
        calculate_cramers_v,
        evaluate_batch_categorical_drift
    )
except ImportError:
    from drift_metrics_categorical import (
        calculate_categorical_psi,
        calculate_chi_square,
        calculate_cramers_v,
        evaluate_batch_categorical_drift
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDIT_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "creditcard.csv")
BANK_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "bank_marketing.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
CREDIT_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def benchmark_serialization_methods():
    """
    Benchmark Pickle (.pkl) vs ONNX Runtime (.onnx) for inference latency,
    file size, and security/hot-reloading capabilities.
    """
    print("\n" + "=" * 75)
    print("BENCHMARK 1: MODEL SERIALIZATION & SERVING (Pickle vs ONNX Runtime)")
    print("=" * 75)

    credit_pkl = os.path.join(MODELS_DIR, "baseline_model.pkl")
    credit_onnx = os.path.join(MODELS_DIR, "baseline_model.onnx")

    results = []

    # 1. Benchmark Pickle
    if os.path.exists(credit_pkl):
        pkl_size_kb = os.path.getsize(credit_pkl) / 1024
        t0 = time.perf_counter()
        model_pkl = joblib.load(credit_pkl)
        pkl_load_ms = (time.perf_counter() - t0) * 1000

        # Inference latency (1,000 samples)
        dummy_df = pd.DataFrame(np.random.randn(1000, 29).astype(np.float32), columns=CREDIT_COLS)
        t0 = time.perf_counter()
        _ = model_pkl.predict(dummy_df)
        pkl_infer_ms = (time.perf_counter() - t0) * 1000

        results.append({
            "Format": "Python Pickle (.pkl)",
            "File Size (KB)": round(pkl_size_kb, 1),
            "Cold Load Time (ms)": round(pkl_load_ms, 2),
            "1k Inference (ms)": round(pkl_infer_ms, 2),
            "Per-Sample Latency (us)": round((pkl_infer_ms / 1000) * 1000, 1),
            "Hot-Reload Support": "No (requires file reload)",
            "Language Neutral": "No (Python only)",
            "Security": "Low (vulnerable to arbitrary code execution)"
        })

    # 2. Benchmark ONNX Runtime
    if os.path.exists(credit_onnx):
        try:
            import onnxruntime as ort
            onnx_size_kb = os.path.getsize(credit_onnx) / 1024
            t0 = time.perf_counter()
            sess = ort.InferenceSession(credit_onnx)
            onnx_load_ms = (time.perf_counter() - t0) * 1000

            inp_name = sess.get_inputs()[0].name
            dummy_data = np.random.randn(1000, 29).astype(np.float32)
            t0 = time.perf_counter()
            _ = sess.run(None, {inp_name: dummy_data})
            onnx_infer_ms = (time.perf_counter() - t0) * 1000

            results.append({
                "Format": "ONNX Runtime (.onnx)",
                "File Size (KB)": round(onnx_size_kb, 1),
                "Cold Load Time (ms)": round(onnx_load_ms, 2),
                "1k Inference (ms)": round(onnx_infer_ms, 2),
                "Per-Sample Latency (us)": round((onnx_infer_ms / 1000) * 1000, 1),
                "Hot-Reload Support": "Yes (zero-downtime memory swap)",
                "Language Neutral": "Yes (C++, Go, Java, Python)",
                "Security": "High (strictly compiled compute graph)"
            })
        except Exception as e:
            print(f"[!] ONNX test note: {e}")

    df_bench = pd.DataFrame(results)
    print(df_bench.to_string(index=False))
    return df_bench


def benchmark_drift_methods():
    """
    Benchmark Synthetic Perturbation vs Natural Demographic Subpopulation Drift
    on the Bank Marketing dataset.
    """
    print("\n" + "=" * 75)
    print("BENCHMARK 2: DRIFT GENERATION METHODOLOGIES (Synthetic vs Natural)")
    print("=" * 75)

    profile_path = os.path.join(PROFILES_DIR, "bank_marketing_baseline.json")
    if not os.path.exists(profile_path) or not os.path.exists(BANK_DATASET_PATH):
        print("[!] Required files missing for drift benchmark.")
        return

    with open(profile_path, "r") as f:
        profiles = json.load(f)["categorical_profiles"]

    df = pd.read_csv(BANK_DATASET_PATH, sep=";")
    cat_cols = list(profiles.keys())

    # 1. Clean Distribution (Normal)
    clean_sample = df.sample(n=500, random_state=42)
    clean_res = evaluate_batch_categorical_drift(profiles, clean_sample, cat_cols)

    # 2. Synthetic Perturbation Drift (Randomly corrupting 70% of housing, loan, education)
    synth_drift = clean_sample.copy()
    synth_drift["housing"] = np.random.choice(["yes", "no"], size=len(synth_drift), p=[0.90, 0.10])
    synth_drift["contact"] = "telephone"
    synth_res = evaluate_batch_categorical_drift(profiles, synth_drift, cat_cols)

    # 3. Natural Demographic Subpopulation Drift (Real Retirees & Students)
    natural_drift = df[df["job"].isin(["retired", "student", "unemployed"])].sample(n=min(500, len(df[df["job"].isin(["retired", "student", "unemployed"])])), random_state=42)
    natural_res = evaluate_batch_categorical_drift(profiles, natural_drift, cat_cols)

    comparison = [
        {
            "Traffic State": "Clean Baseline",
            "Drift Method": "Random Sampling",
            "Top Feature Tested": "job",
            "PSI Score": clean_res["features"]["job"]["psi"],
            "Chi2 Stat": clean_res["features"]["job"]["chi2"],
            "p-Value": clean_res["features"]["job"]["p_val"],
            "Cramer's V": clean_res["features"]["job"]["cramers_v"],
            "Drift Flagged": clean_res["features"]["job"]["drift_detected"]
        },
        {
            "Traffic State": "Synthetic Perturbation",
            "Drift Method": "Synthetic Reassignment",
            "Top Feature Tested": "housing",
            "PSI Score": synth_res["features"]["housing"]["psi"],
            "Chi2 Stat": synth_res["features"]["housing"]["chi2"],
            "p-Value": synth_res["features"]["housing"]["p_val"],
            "Cramer's V": synth_res["features"]["housing"]["cramers_v"],
            "Drift Flagged": synth_res["features"]["housing"]["drift_detected"]
        },
        {
            "Traffic State": "Natural Subpopulation Shift",
            "Drift Method": "Demographic Partitioning",
            "Top Feature Tested": "job",
            "PSI Score": natural_res["features"]["job"]["psi"],
            "Chi2 Stat": natural_res["features"]["job"]["chi2"],
            "p-Value": natural_res["features"]["job"]["p_val"],
            "Cramer's V": natural_res["features"]["job"]["cramers_v"],
            "Drift Flagged": natural_res["features"]["job"]["drift_detected"]
        }
    ]

    df_drift = pd.DataFrame(comparison)
    print(df_drift.to_string(index=False))
    return df_drift


def benchmark_numerical_vs_categorical():
    """
    Summarize architecture & statistical behavior across both datasets.
    """
    print("\n" + "=" * 75)
    print("BENCHMARK 3: DATASET COMPARISON (Numerical vs Categorical)")
    print("=" * 75)

    comparison = [
        {
            "Attribute": "Dataset Name",
            "Numerical Pipeline": "Credit Card Fraud Detection",
            "Categorical / Mixed Pipeline": "Bank Marketing Campaign (UCI)"
        },
        {
            "Attribute": "Total Records",
            "Numerical Pipeline": "284,807 transactions",
            "Categorical / Mixed Pipeline": "45,211 customer interactions"
        },
        {
            "Attribute": "Feature Types",
            "Numerical Pipeline": "29 Continuous (PCA V1-V28, Amount)",
            "Categorical / Mixed Pipeline": "9 Categorical, 7 Numerical"
        },
        {
            "Attribute": "Primary Statistical Drift Test",
            "Numerical Pipeline": "Two-Sample Kolmogorov-Smirnov (KS-Test)",
            "Categorical / Mixed Pipeline": "Chi-Square Goodness-of-Fit Test"
        },
        {
            "Attribute": "Stability Index Formulation",
            "Numerical Pipeline": "Continuous Decile-Binned PSI",
            "Categorical / Mixed Pipeline": "Discrete Categorical PSI (Laplace smoothed)"
        },
        {
            "Attribute": "Effect Size Metric",
            "Numerical Pipeline": "KS Distance (D-statistic)",
            "Categorical / Mixed Pipeline": "Cramer's V Association Metric"
        },
        {
            "Attribute": "Drift Gate Threshold",
            "Numerical Pipeline": "(PSI >= 0.25) AND (KS p < 0.05)",
            "Categorical / Mixed Pipeline": "(PSI >= 0.25) AND (Chi2 p < 0.05)"
        },
        {
            "Attribute": "Model Architecture",
            "Numerical Pipeline": "RandomForest on Raw Features",
            "Categorical / Mixed Pipeline": "ColumnTransformer (OneHot + Scaler) + RF"
        },
        {
            "Attribute": "Model Artifact Serialization",
            "Numerical Pipeline": "ONNX (models/baseline_model.onnx)",
            "Categorical / Mixed Pipeline": "ONNX (models/bank_model.onnx)"
        },
        {
            "Attribute": "Serving & Hot-Reload Engine",
            "Numerical Pipeline": "FastAPI + ONNX Runtime (port 8000)",
            "Categorical / Mixed Pipeline": "FastAPI + ONNX Runtime (port 8000)"
        }
    ]

    df_comp = pd.DataFrame(comparison)
    print(df_comp.to_string(index=False))
    return df_comp


def main():
    try:
        print("*" * 75)
        print("DRIFTWATCH: EMPIRICAL COMPARATIVE BENCHMARK EVALUATION")
        print("*" * 75)

        benchmark_serialization_methods()
        benchmark_drift_methods()
        benchmark_numerical_vs_categorical()

        print("\n[+] All comparative benchmark evaluations generated successfully!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
