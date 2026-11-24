"""FastAPI prediction service with validation rules and structured JSON logging."""

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field

from src.mp255.clean import clean_department

# --- Structured JSON Logger (Handbook Section 9) ---
class JsonFormatter(logging.Formatter):
    """Format log record as a single JSON line to stdout."""
    def format(self, record: logging.LogRecord) -> str:
        log_payload = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_payload.update(record.extra_data)
        return json.dumps(log_payload)


json_handler = logging.StreamHandler(sys.stdout)
json_handler.setFormatter(JsonFormatter())

logger = logging.getLogger("mp255_api")
logger.setLevel(logging.INFO)
logger.addHandler(json_handler)
logger.propagate = False


# --- FastAPI Application ---
app = FastAPI(
    title="MP255 OPD Waiting Time Prediction Service",
    description="Surat District Hospital OPD waiting time estimator",
    version="1.0.0",
)

MODEL_PATH = Path("models/model.joblib")
_model: Any = None


def get_model() -> Any:
    """Lazy load champion model from models/model.joblib."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Champion model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


# --- Request and Response Schemas (PROBLEM.md Section 5) ---
class PredictRequest(BaseModel):
    # Rule 1: patients_ahead above 150 is rejected
    patients_ahead: int = Field(
        ...,
        ge=0,
        le=150,
        description="Tokens ahead in queue (max 150)",
    )
    doctors_on_duty: int = Field(
        ...,
        ge=1,
        description="Doctors currently on duty in that OPD",
    )
    arrival_hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour token was taken (0-23)",
    )
    # Rule 2: avg_consult_min below 2 is rejected
    avg_consult_min: float = Field(
        ...,
        ge=2.0,
        description="Average consultation length in minutes (min 2)",
    )
    lab_tests_pending: int = Field(
        ...,
        ge=0,
        description="Lab reports awaited",
    )
    distance_km: float = Field(
        ...,
        ge=0.0,
        description="Patient travel distance in km",
    )
    department: str = Field(
        ...,
        description="OPD Department (General, Paediatrics, Orthopaedics)",
    )
    wing: str = Field(
        ...,
        description="Hospital wing (Old Wing, New Wing, Annexe)",
    )


class PredictResponse(BaseModel):
    prediction: float
    wait_min: float


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint for Kubernetes liveness/readiness probes."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict OPD waiting time in minutes."""
    try:
        model = get_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    dept_canonical = clean_department(request.department)

    input_df = pd.DataFrame([
        {
            "patients_ahead": request.patients_ahead,
            "doctors_on_duty": request.doctors_on_duty,
            "arrival_hour": request.arrival_hour,
            "avg_consult_min": request.avg_consult_min,
            "lab_tests_pending": request.lab_tests_pending,
            "distance_km": request.distance_km,
            "department": dept_canonical,
            "wing": request.wing,
        }
    ])

    raw_pred = float(model.predict(input_df)[0])
    pred_val = round(max(0.0, raw_pred), 2)

    # Output structured JSON log to stdout (never logging PII)
    log_record = logging.LogRecord(
        name="mp255_api",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="prediction_served",
        args=(),
        exc_info=None,
    )
    log_record.extra_data = {
        "event": "predict",
        "department": dept_canonical,
        "wing": request.wing,
        "patients_ahead": request.patients_ahead,
        "doctors_on_duty": request.doctors_on_duty,
        "predicted_wait_min": pred_val,
    }
    logger.handle(log_record)

    return PredictResponse(prediction=pred_val, wait_min=pred_val)
