from __future__ import annotations

import os
import json
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
TARGET = "Outcome"
MODEL_PATH = os.getenv("MODEL_PATH", "diabetes_model.pkl")
METADATA_PATH = os.getenv("MODEL_METADATA_PATH", "diabetes_model_meta.json")
DATA_URL = os.getenv("DATA_URL", "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")


def load_dataset(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    missing_cols = {TARGET, *FEATURES} - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {sorted(missing_cols)}")
    return df


def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Select required columns in correct order
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    # Replace known sentinel zeros with NaN for specific medical measures then impute
    zero_as_missing = ["Glucose", "BloodPressure", "BMI"]
    for col in zero_as_missing:
        X.loc[X[col] == 0, col] = np.nan

    return X, y


def build_model() -> Pipeline:
    # RF is robust to scaling; we just impute missing values
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", clf),
    ])
    return pipeline


def main() -> None:
    df = load_dataset(DATA_URL)
    X, y = preprocess_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    try:
        y_proba = getattr(model, "predict_proba")(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None

    # Save model and lightweight metadata (feature order)
    joblib.dump(model, MODEL_PATH)
    meta = {"features": FEATURES, "accuracy": acc, "roc_auc": auc}
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f)

    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"ℹ️  Metrics: accuracy={acc:.4f}" + (f", roc_auc={auc:.4f}" if auc is not None else ""))
    print(f"ℹ️  Metadata saved to {METADATA_PATH}")


if __name__ == "__main__":
    main()
