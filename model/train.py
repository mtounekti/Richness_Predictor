import os
import time
import sqlite3
import psutil
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from loguru import logger

from preprocess import (
    load_data, clean_data, prepare_features,
    build_pipeline, save_pipeline, PIPELINE_PATH
)

DATA_PATH = os.path.join("data", "adult.csv")
MODEL_PATH = os.path.join("artifacts", "model.joblib")
DB_PATH = os.path.join("data", "richness.db")
MLFLOW_EXPERIMENT = "Richness-Predictor"

mlflow.set_experiment(MLFLOW_EXPERIMENT)


def init_db():
    # crée la table predictions si elle n'existe pas
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            workclass TEXT,
            education TEXT,
            occupation TEXT,
            prediction INTEGER,
            confidence REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Base de données initialisée : {DB_PATH}")


def measure_resources(func, *args, **kwargs):
    # mesure temps et RAM avant/après exécution
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024
    start = time.perf_counter()

    result = func(*args, **kwargs)

    duration = time.perf_counter() - start
    mem_after = process.memory_info().rss / 1024 / 1024

    return result, duration, mem_after - mem_before


def train_and_evaluate(name, model, preprocessor, X_train, X_test, y_train, y_test):
    # construit le pipeline complet preprocessor + modèle
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    logger.info(f"[{name}] Entraînement...")
    (pipeline, train_duration, train_ram) = measure_resources(
        pipeline.fit, X_train, y_train
    )

    logger.info(f"[{name}] Inférence...")
    (y_pred, infer_duration, _) = measure_resources(
        pipeline.predict, X_test
    )

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info(f"[{name}] Accuracy : {acc:.4f} | Train : {train_duration:.2f}s | RAM : {train_ram:.1f}MB")

    # log MLflow
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("train_duration_s", round(train_duration, 3))
        mlflow.log_metric("train_ram_mb", round(train_ram, 2))
        mlflow.log_metric("infer_duration_s", round(infer_duration, 4))
        mlflow.log_metric("precision_macro", report["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report["macro avg"]["recall"])
        mlflow.log_metric("f1_macro", report["macro avg"]["f1-score"])
        mlflow.sklearn.log_model(pipeline, "pipeline")

    return pipeline, acc, train_duration, train_ram


def main():
    init_db()

    logger.info("Chargement des données...")
    df = load_data(DATA_PATH)

    logger.info("Nettoyage...")
    df = clean_data(df)

    logger.info("Préparation des features...")
    X, y = prepare_features(df)

    logger.info("Split train/test (80/20 stratifié)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_pipeline()

    # comparaison de 3 architectures
    candidates = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)),
        ("RandomForest-100", RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)),
        ("LightGBM", lgb.LGBMClassifier(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42)),
    ]

    results = []
    for name, model in candidates:
        pipeline, acc, duration, ram = train_and_evaluate(
            name, model, preprocessor, X_train, X_test, y_train, y_test
        )
        results.append((name, pipeline, acc, duration, ram))

    # meilleur modèle (accuracy / ressources)
    best = max(results, key=lambda r: r[2])
    logger.info(f"Meilleur modèle : {best[0]} — Accuracy : {best[2]:.4f}")

    # save du meilleur pipeline
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best[1], MODEL_PATH)
    logger.info(f"Pipeline sauvegardé : {MODEL_PATH}")

    print("\n=== Comparaison des architectures ===")
    print(f"{'Modèle':<25} {'Accuracy':>10} {'Temps train':>12} {'RAM (MB)':>10}")
    print("-" * 60)
    for name, _, acc, duration, ram in sorted(results, key=lambda r: r[2], reverse=True):
        print(f"{name:<25} {acc:>10.4f} {duration:>11.2f}s {ram:>9.1f}MB")


if __name__ == "__main__":
    main()