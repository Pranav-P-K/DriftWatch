import os
import joblib
import numpy as np
import onnx
from skl2onnx import to_onnx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL_PATH = os.path.join(BASE_DIR, "models", "baseline_model.pkl")
ONNX_PATH = os.path.join(BASE_DIR, "models", "baseline_model.onnx")

if os.path.exists(PKL_PATH):
    model = joblib.load(PKL_PATH)
    dummy = np.zeros((1, 29), dtype=np.float32)
    onnx_model = to_onnx(model, dummy, target_opset=15)
    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
    onnx.checker.check_model(onnx_model)
    size_kb = os.path.getsize(ONNX_PATH) / 1024
    print(f"[+] Successfully exported {ONNX_PATH} ({size_kb:.1f} KB)")
else:
    print(f"[-] {PKL_PATH} not found.")
