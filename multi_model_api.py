from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle

# Load your models
with open("logreg_model.pkl", "rb") as f:
    logreg_model = pickle.load(f)

with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Create FastAPI app
app = FastAPI(title="ML Prediction API", version="1.0")

# Define request schema
class PredictRequest(BaseModel):
    data: List[List[float]]  # Each row = list of features


@app.post("/predict/logreg")
async def predict_logreg(request: PredictRequest):
    input_data = request.data
    preds = logreg_model.predict_proba(input_data)[:, 1].tolist()
    return {"model": "Logistic Regression", "predictions": preds}


@app.post("/predict/rf")
async def predict_rf(request: PredictRequest):
    input_data = request.data
    preds = rf_model.predict_proba(input_data)[:, 1].tolist()
    return {"model": "Random Forest", "predictions": preds}
