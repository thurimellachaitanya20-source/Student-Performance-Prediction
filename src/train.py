from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "student_performance.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "student_performance_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
FEATURES = ["Study_Hours", "Attendance", "Previous_Marks"]
TARGET = "Final_Marks"


def train_model() -> dict[str, float]:
    """Train and save a linear regression model from the included sample data."""
    data = pd.read_csv(DATA_PATH)
    missing_columns = set(FEATURES + [TARGET]) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    features = data[FEATURES]
    target = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "mse": round(float(mean_squared_error(y_test, predictions)), 2),
        "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 2),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "test_samples": int(len(y_test)),
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    results = train_model()
    print(f"Saved model to: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print("Evaluation metrics:")
    for metric, value in results.items():
        print(f"  {metric}: {value}")
