import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from loguru import logger

from preprocess import NUMERIC_COLS, CATEGORICAL_COLS

# Chargement du pipeline au démarrage
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
DB_PATH = os.getenv("DB_PATH", "data/richness.db")

pipeline = joblib.load(MODEL_PATH)
logger.info(f"Modèle chargé depuis {MODEL_PATH}")

# Métriques Prometheus
prediction_counter = Counter("predictions_total", "Nombre total de prédictions", ["predicted_class"])
prediction_latency = Histogram("prediction_latency_seconds", "Latence des prédictions")
error_counter = Counter("prediction_errors_total", "Nombre d'erreurs de prédiction")

app = FastAPI(title="Richness Predictor API", version="2.0")


# Schéma d'entrée — sans données sensibles
class PredictionInput(BaseModel):
    age: int
    workclass: str
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    fnlwgt: float = 0.0


class PredictionOutput(BaseModel):
    prediction: int
    label: str
    confidence: float


def input_to_dataframe(data: PredictionInput) -> pd.DataFrame:
    # Convertit le schéma Pydantic en DataFrame compatible avec le pipeline
    return pd.DataFrame([{
        "age": data.age,
        "fnlwgt": data.fnlwgt,
        "education.num": data.education_num,
        "capital.gain": data.capital_gain,
        "capital.loss": data.capital_loss,
        "hours.per.week": data.hours_per_week,
        "workclass": data.workclass,
        "education": data.education,
        "marital.status": data.marital_status,
        "occupation": data.occupation,
        "relationship": data.relationship,
    }])


def save_to_db(data: PredictionInput, prediction: int, confidence: float):
    # Sauvegarde la prédiction dans SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (age, workclass, education, occupation, prediction, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        data.age, data.workclass, data.education,
        data.occupation, prediction, confidence,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


@app.get("/", summary="Racine", description="Vérifie que l'API est en ligne.")
async def root():
    return {"message": "Richness Predictor API v2.0"}


@app.get("/health", summary="Santé", description="Retourne le statut de l'API.")
async def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.get("/metrics", summary="Métriques Prometheus", description="Expose les métriques au format Prometheus.")
async def metrics():
    # endpoint scrappé par Prometheus
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionOutput, summary="Prédiction", description="Prédit si une personne gagne plus ou moins de 50k$/an.")
async def predict(data: PredictionInput):
    with prediction_latency.time():
        try:
            df = input_to_dataframe(data)
            prediction = int(pipeline.predict(df)[0])
            confidence = round(float(pipeline.predict_proba(df)[0][prediction]), 4)

            label = ">50K" if prediction == 1 else "<=50K"
            prediction_counter.labels(predicted_class=label).inc()

            logger.info(f"[PREDICT] prediction={label} confidence={confidence}")

            save_to_db(data, prediction, confidence)

            return PredictionOutput(prediction=prediction, label=label, confidence=confidence)

        except Exception as e:
            error_counter.inc()
            logger.error(f"[PREDICT] Erreur : {e}")
            raise HTTPException(status_code=500, detail=str(e))