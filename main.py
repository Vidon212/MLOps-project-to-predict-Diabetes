from typing import Optional
import os
import logging

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Diabetes Prediction API")


def _model_path() -> str:
    return os.getenv("MODEL_PATH", "diabetes_model.pkl")


def _load_model_safely() -> Optional[object]:
    try:
        return joblib.load(_model_path())
    except Exception as exc:
        logging.exception("Failed to load model: %s", exc)
        return None


def _get_model() -> Optional[object]:
    # Small helper to access the model from app state
    return getattr(app.state, "model", None)


@app.on_event("startup")
def on_startup() -> None:
    # Load once on startup to avoid import-time side effects and allow
    # better error messages/health checks if loading fails
    app.state.model = _load_model_safely()


class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., ge=0, description="Number of pregnancies (non-negative integer)")
    Glucose: float = Field(..., gt=0, description="Plasma glucose concentration")
    BloodPressure: float = Field(..., gt=0, description="Diastolic blood pressure")
    BMI: float = Field(..., gt=0, description="Body mass index")
    Age: int = Field(..., ge=0, description="Age in years")

    class Config:
        extra = "forbid"


@app.get("/")
def read_root() -> dict:
    return {"message": "Diabetes Prediction API is live"}


@app.get("/health")
def health() -> dict:
    model_loaded = _get_model() is not None
    return {"status": "ok" if model_loaded else "degraded", "model_loaded": model_loaded}


@app.post("/predict")
def predict(data: DiabetesInput) -> dict:
    model = _get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = np.asarray(
        [[data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]],
        dtype=np.float32,
    )

    pred = model.predict(features)[0]
    response = {"diabetic": bool(pred)}

    # Optionally return probability if the model supports it
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        if len(proba) == 2:
            response["probability_diabetic"] = float(proba[1])

    return response
