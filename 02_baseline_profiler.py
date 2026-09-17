"""
02_baseline_profiler.py
-----------------------
Computes the baseline statistical profile for drift detection:
1. Uses SHAP TreeExplainer on baseline_model.pkl to identify the Top 5 most influential features.
2. Computes 10 equal-frequency quantile bin edges (deciles) and expected proportions for each top feature (for PSI).
3. Extracts reference sample distributions (1,000 samples/feature) for two-sample Kolmogorov-Smirnov (KS) testing.
4. Serializes:
   - profiles/baseline_profile.json
   - profiles/shap_baseline.json
   - profiles/baseline_samples.pkl
"""

import os
import gc
import json
import time
import pickle
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "creditcard.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline_model.pkl")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
BASELINE_PROFILE_JSON = os.path.join(PROFILES_DIR, "baseline_profile.json")
SHAP_BASELINE_JSON = os.path.join(PROFILES_DIR, "shap_baseline.json")
BASELINE_SAMPLES_PKL = os.path.join(PROFILES_DIR, "baseline_samples.pkl")


def load_assets():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Baseline model not found at {MODEL_PATH}. Run 01_train_model.py first.")

    print(f"[*] Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    print(f"[*] Loading dataset with optimized dtypes from {DATASET_PATH}...")
    # Explicit dtypes to minimize memory allocation on low-RAM environments
    dtypes = {f"V{i}": np.float32 for i in range(1, 29)}
    dtypes["Amount"] = np.float32
    dtypes["Class"] = np.int8

    start_time = time.time()
    df = pd.read_csv(DATASET_PATH, dtype=dtypes, usecols=list(dtypes.keys()))
    print(f"[*] Loaded {len(df):,} records ({df.memory_usage().sum() / 1e6:.1f} MB) in {time.time() - start_time:.2f}s")

    return model, df


def compute_shap_rankings(model, df, sample_size=1000):
    print(f"\n[*] Extracting stratified sample of {sample_size:,} records for SHAP TreeExplainer...")
    pos_df = df[df["Class"] == 1]
    neg_df = df[df["Class"] == 0]

    pos_n = min(50, len(pos_df))
    neg_n = sample_size - pos_n
    sample_df = pd.concat([
        pos_df.sample(n=pos_n, random_state=42),
        neg_df.sample(n=neg_n, random_state=42)
    ]).sample(frac=1.0, random_state=42)

    X_sample = sample_df.drop(columns=["Class"])
    feature_names = list(X_sample.columns)

    print(f"[*] Initializing SHAP TreeExplainer on {len(X_sample)} sample records...")
    start_time = time.time()
    explainer = shap.TreeExplainer(model)
    shap_output = explainer(X_sample)

    # Extract SHAP values for class 1 (fraud)
    if hasattr(shap_output, "values"):
        vals = shap_output.values
        if vals.ndim == 3:
            vals = vals[:, :, 1]
    elif isinstance(shap_output, list):
        vals = shap_output[1]
    else:
        vals = np.array(shap_output)

    print(f"[*] SHAP values computed in {time.time() - start_time:.2f}s (matrix shape: {vals.shape})")

    # Mean absolute SHAP value per feature
    mean_abs_shap = np.abs(vals).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 50)
    print("SHAP FEATURE IMPORTANCE RANKING (Top 10)")
    print("=" * 50)
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {idx+1:02d}. {row['feature']:<10} : {row['mean_abs_shap']:.5f}")

    top_5_features = feature_importance["feature"].head(5).tolist()
    print("\n" + "=" * 50)
    print(f"[+] Selected Top 5 Monitored Features: {top_5_features}")
    print("=" * 50)

    # Free memory
    del X_sample, sample_df, explainer, shap_output, vals
    gc.collect()

    return feature_importance, top_5_features


def compute_baseline_profiles(df, top_5_features):
    print("\n[*] Computing 10 equal-frequency quantile bins for Top 5 features...")
    baseline_profile = {
        "features": {},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_bins": 10
    }

    for feat in top_5_features:
        values = df[feat].values

        # 10 deciles -> 11 edges (0%, 10%, ..., 100%)
        percentiles = np.linspace(0, 100, 11)
        raw_edges = np.percentile(values, percentiles)

        # Enforce strict monotonicity to handle clustered distributions
        bin_edges = [float(raw_edges[0])]
        for val in raw_edges[1:]:
            val = float(val)
            if val <= bin_edges[-1]:
                val = bin_edges[-1] + 1e-5
            bin_edges.append(val)

        # Count proportions using infinite boundary margins
        extended_edges = list(bin_edges)
        extended_edges[0] = -np.inf
        extended_edges[-1] = np.inf

        counts, _ = np.histogram(values, bins=extended_edges)
        proportions = counts / counts.sum()

        baseline_profile["features"][feat] = {
            "bin_edges": [round(e, 6) for e in bin_edges],
            "expected_proportions": [round(float(p), 6) for p in proportions],
            "min_val": round(float(values.min()), 6),
            "max_val": round(float(values.max()), 6),
            "mean": round(float(values.mean()), 6),
            "std": round(float(values.std()), 6)
        }

        print(f"  Feature '{feat:<8}': 10 bins created | sum(proportions) = {proportions.sum():.4f}")

    return baseline_profile


def save_profiles(baseline_profile, feature_importance, top_5_features, df):
    os.makedirs(PROFILES_DIR, exist_ok=True)

    # 1. baseline_profile.json
    with open(BASELINE_PROFILE_JSON, "w") as f:
        json.dump(baseline_profile, f, indent=2)
    print(f"\n[+] Saved baseline bin profile to: {BASELINE_PROFILE_JSON}")

    # 2. shap_baseline.json
    shap_dict = {
        "top_5_features": top_5_features,
        "features": {
            row["feature"]: round(float(row["mean_abs_shap"]), 6)
            for _, row in feature_importance.iterrows()
        }
    }
    with open(SHAP_BASELINE_JSON, "w") as f:
        json.dump(shap_dict, f, indent=2)
    print(f"[+] Saved SHAP baseline rankings to: {SHAP_BASELINE_JSON}")

    # 3. baseline_samples.pkl (1,000 reference samples per top feature for KS-test)
    sample_data = {}
    sample_df = df.sample(n=min(1000, len(df)), random_state=42)
    for feat in top_5_features:
        sample_data[feat] = sample_df[feat].values.astype(np.float64)

    with open(BASELINE_SAMPLES_PKL, "wb") as f:
        pickle.dump(sample_data, f)
    print(f"[+] Saved 1,000 reference samples/feature to: {BASELINE_SAMPLES_PKL}")


def main():
    model, df = load_assets()
    feature_importance, top_5_features = compute_shap_rankings(model, df, sample_size=1000)
    baseline_profile = compute_baseline_profiles(df, top_5_features)
    save_profiles(baseline_profile, feature_importance, top_5_features, df)
    print("\n[SUCCESS] Day 2 Baseline Profiling complete!")


if __name__ == "__main__":
    main()
