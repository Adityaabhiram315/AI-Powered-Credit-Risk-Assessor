from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.model.predict import predict_single

app = FastAPI(title="AI-Powered Credit Risk Assessor")

class CreditFeatures(BaseModel):
    # Define your features and types here, example:
    age: float
    income: float
    loan_amount: float
    # ... add other features according to your dataset

@app.post("/predict")
def predict(features: CreditFeatures):
    input_dict = features.dict()
    result = predict_single(input_dict)
    return result