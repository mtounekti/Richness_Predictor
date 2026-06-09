import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import os

# colones numériques: pas de données sensibles
NUMERIC_COLS = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week"
]

# colonnes catégorielles: race, sex et native.country supprimés
CATEGORICAL_COLS = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship"
]

TARGET_COL = "income"
PIPELINE_PATH = os.path.join("artifacts", "pipeline.joblib")


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # remplace les valeurs manquantes encodées " ?"
    df.replace(" ?", np.nan, inplace=True)
    df.dropna(inplace=True)
    # clean des espaces dans les valeurs string
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def encode_target(df: pd.DataFrame) -> pd.Series:
    # >50K → 1, <=50K → 0
    return df[TARGET_COL].apply(lambda x: 1 if x.strip() == ">50K" else 0)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_COLS),
        # handle_unknown="ignore" règle le problème des catégories inconnues à l'inférence
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS)
    ])
    return preprocessor


def prepare_features(df: pd.DataFrame):
    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = encode_target(df)
    return X, y


def save_pipeline(pipeline, path: str = PIPELINE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Pipeline sauvegardé dans : {path}")


def load_pipeline(path: str = PIPELINE_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline introuvable : {path}")
    return joblib.load(path)