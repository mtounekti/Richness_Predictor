import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from loguru import logger

from preprocess import load_data, clean_data, prepare_features, load_pipeline

MODEL_PATH = os.path.join("artifacts", "model.joblib")
DB_PATH = os.path.join("data", "richness.db")
DATA_PATH = os.path.join("data", "adult.csv")

# Colonnes sensibles supprimées du modèle mais gardées pour l'analyse d'équité
SENSITIVE_COLS = ["sex", "race", "native.country"]


def evaluate_performance(pipeline, X_test, y_test):
    # Métriques globales du modèle
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"Accuracy  : {acc:.4f}")
    logger.info(f"AUC-ROC   : {auc:.4f}")

    print("\n=== Rapport de classification ===")
    print(report)
    print("Matrice de confusion :")
    print(cm)

    return {"accuracy": acc, "auc": auc, "confusion_matrix": cm}


def evaluate_fairness(pipeline, df_test, y_test):
    # Analyse d'équité sur les groupes sensibles
    # Les colonnes sensibles ne sont PAS dans le modèle
    # mais on les utilise ici pour mesurer les biais résiduels
    print("\n=== Analyse d'équité (fairness) ===")

    X_test = df_test.drop(columns=SENSITIVE_COLS + ["income"], errors="ignore")
    y_pred = pipeline.predict(X_test)

    results = []

    for col in SENSITIVE_COLS:
        if col not in df_test.columns:
            continue

        print(f"\n--- Groupe : {col} ---")
        groups = df_test[col].unique()

        for group in sorted(groups):
            mask = df_test[col] == group
            if mask.sum() < 10:
                continue

            acc_group = accuracy_score(y_test[mask], y_pred[mask])
            positive_rate = y_pred[mask].mean()

            print(f"  {group:<30} accuracy={acc_group:.4f}  taux_positif={positive_rate:.4f}")
            results.append({
                "variable": col,
                "groupe": group,
                "accuracy": acc_group,
                "taux_positif": positive_rate,
                "n": int(mask.sum())
            })

    df_fairness = pd.DataFrame(results)

    # Détection des écarts importants
    for col in SENSITIVE_COLS:
        subset = df_fairness[df_fairness["variable"] == col]
        if subset.empty:
            continue
        ecart = subset["taux_positif"].max() - subset["taux_positif"].min()
        if ecart > 0.1:
            logger.warning(f"[FAIRNESS] Écart important sur {col} : {ecart:.4f} — vérifier les biais")
        else:
            logger.info(f"[FAIRNESS] {col} : écart acceptable ({ecart:.4f})")

    return df_fairness


def save_predictions_to_db(pipeline, X_test, y_test, df_test):
    # Sauvegarde les prédictions dans SQLite pour historique
    conn = sqlite3.connect(DB_PATH)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    from datetime import datetime
    timestamp = datetime.now().isoformat()

    records = []
    for i in range(len(X_test)):
        records.append({
            "age": int(df_test.iloc[i]["age"]) if "age" in df_test.columns else None,
            "workclass": str(df_test.iloc[i]["workclass"]) if "workclass" in df_test.columns else None,
            "education": str(df_test.iloc[i]["education"]) if "education" in df_test.columns else None,
            "occupation": str(df_test.iloc[i]["occupation"]) if "occupation" in df_test.columns else None,
            "prediction": int(y_pred[i]),
            "confidence": round(float(y_proba[i]), 4),
            "timestamp": timestamp
        })

    df_records = pd.DataFrame(records)
    df_records.to_sql("predictions", conn, if_exists="append", index=False)
    conn.close()
    logger.info(f"{len(records)} prédictions sauvegardées dans SQLite")


def main():
    logger.info("Chargement du modèle...")
    pipeline = joblib.load(MODEL_PATH)

    logger.info("Chargement des données...")
    df = load_data(DATA_PATH)
    df = clean_data(df)

    from sklearn.model_selection import train_test_split
    from preprocess import prepare_features

    X, y = prepare_features(df)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Index du jeu de test pour récupérer les colonnes sensibles
    df_test = df.iloc[y_test.index] if hasattr(y_test, "index") else df.iloc[-len(y_test):]

    # évaluation des performances
    evaluate_performance(pipeline, X_test, y_test)

    # analyse d'équité
    evaluate_fairness(pipeline, df_test, y_test)

    # save dans SQLite
    save_predictions_to_db(pipeline, X_test, y_test, df_test)


if __name__ == "__main__":
    main()