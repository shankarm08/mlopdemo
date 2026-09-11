import time
import logging
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from src.predict import predict_single, DEFAULT_MODEL_PATH

# -------------------------------------------------------------
# Setup Logging & Simple Monitoring
# -------------------------------------------------------------
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
MONITORING_LOG_FILE = os.path.join(LOGS_DIR, "prediction_monitoring.log")

# Configure logger
logger = logging.getLogger("churn_api_monitoring")
logger.setLevel(logging.INFO)

# File handler for monitoring records
file_handler = logging.FileHandler(MONITORING_LOG_FILE)
file_formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# -------------------------------------------------------------
# FastAPI Application
# -------------------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="FastAPI service for real-time customer churn prediction with basic MLOps monitoring.",
    version="1.0.0",
)


# -------------------------------------------------------------
# Pydantic Schemas
# -------------------------------------------------------------
class CustomerData(BaseModel):
    age: int = Field(..., ge=18, le=120, examples=[35])
    monthly_bill: float = Field(..., ge=0.0, examples=[75.5])
    tenure_months: int = Field(..., ge=0, examples=[12])
    support_calls: int = Field(..., ge=0, examples=[3])
    usage_hours: float = Field(..., ge=0.0, examples=[25.0])
    contract_type: str = Field(..., examples=["monthly"])


class PredictionResponse(BaseModel):
    churn_prediction: int
    message: str


class HealthResponse(BaseModel):
    status: str


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint to verify API and model status."""
    return {"status": "healthy"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
def predict(customer: CustomerData):
    """
    Predict whether a customer is likely to churn.
    Monitors input features, prediction outcome, and response time.
    """
    start_time = time.time()
    input_data = customer.model_dump()

    try:
        result = predict_single(input_data, model_path=DEFAULT_MODEL_PATH)
    except FileNotFoundError as fnf_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(fnf_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction error: {str(exc)}",
        )

    # Calculate response time
    response_time_ms = round((time.time() - start_time) * 1000, 2)

    # Structured Monitoring Log Entry
    monitoring_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input_features": input_data,
        "prediction": result["churn_prediction"],
        "message": result["message"],
        "response_time_ms": response_time_ms,
    }

    # Log monitoring payload
    logger.info(json.dumps(monitoring_record))

    return result
