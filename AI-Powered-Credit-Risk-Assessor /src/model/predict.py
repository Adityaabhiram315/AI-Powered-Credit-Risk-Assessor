import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.model.model_utils import load_model, load_scaler
from src.config import MODEL_PATH

model = load_model(MODEL_PATH)
scaler = load_scaler("model/scaler.pkl")

def predict_single(input_dict):
    x = np.array([list(input_dict.values())])
    x_scaled = scaler.transform(x)
    pred_proba = model.predict_proba(x_scaled)[0][1]
    pred_class = int(pred_proba >= 0.5)
    return {"risk_score": float(pred_proba), "creditworthy": bool(not pred_class)}