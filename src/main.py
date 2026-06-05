
 from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="Credit scoring API for Buy Now Pay Later service",
    version="1.0.0"
)

# Load model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/best_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../../models/scaler.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Model and scaler loaded!")
except Exception as e:
    print(f"⚠️ Could not load model: {e}")
    model = None
    scaler = None

class CustomerFeatures(BaseModel):
    total_transactions: float
    total_amount: float
    avg_amount: float
    std_amount: float
    max_amount: float
    min_amount: float
    total_value: float
    unique_products: float
    unique_channels: float
    Recency: float
    Frequency: float
    Monetary: float
    RFM_Score: float

class PredictionResponse(BaseModel):
    customer_id: str
    risk_label: int
    risk_probability: float
    risk_category: str
    recommendation: str

@app.get("/")
def root():
    return {
        "message": "Bati Bank Credit Risk API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(customer_id: str, features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        feature_array = np.array([[
            features.total_transactions,
            features.total_amount,
            features.avg_amount,
            features.std_amount,
            features.max_amount,
            features.min_amount,
            features.total_value,
            features.unique_products,
            features.unique_channels,
            features.Recency,
            features.Frequency,
            features.Monetary,
            features.RFM_Score
        ]])
        
        # Predict
        risk_label = int(model.predict(feature_array)[0])
        risk_prob = float(model.predict_proba(feature_array)[0][1])
        
        # Risk category
        if risk_prob >= 0.7:
            risk_category = "HIGH RISK"
            recommendation = "Reject loan application"
        elif risk_prob >= 0.4:
            risk_category = "MEDIUM RISK"
            recommendation = "Manual review required"
        else:
            risk_category = "LOW RISK"
            recommendation = "Approve loan application"
        
        return PredictionResponse(
            customer_id=customer_id,
            risk_label=risk_label,
            risk_probability=round(risk_prob, 4),
            risk_category=risk_category,
            recommendation=recommendation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
