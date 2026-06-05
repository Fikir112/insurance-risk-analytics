from pydantic import BaseModel

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
