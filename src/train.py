"""Machine Learning Training Pipeline for Student Performance Prediction.
Trains multiple regression models, evaluates performance metrics,
and serializes artifacts for Streamlit and Desktop GUI apps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "student_performance.csv"
DATASET_PATH = PROJECT_ROOT / "data" / "student_performance_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "student_performance_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

CORE_FEATURES = ["Study_Hours", "Attendance", "Previous_Marks"]
EXTENDED_FEATURES = ["Study_Hours", "Attendance", "Previous_Marks", "Sleep_Hours", "Practice_Tests"]
TARGET = "Final_Marks"


def load_dataset() -> pd.DataFrame:
    """Load dataset with fallback paths."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    raise FileNotFoundError(f"Neither {DATA_PATH} nor {DATASET_PATH} found.")


def evaluate_predictions(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate key regression evaluation metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
    }


def train_model(features_list: list[str] | None = None) -> dict[str, Any]:
    """Train multiple regression models, select best, and save artifacts."""
    data = load_dataset()
    features = features_list or CORE_FEATURES

    # Verify features and target exist
    missing = set(features + [TARGET]) - set(data.columns)
    if missing:
        # Fallback to core features if extended columns are absent
        features = [col for col in CORE_FEATURES if col in data.columns]
        missing = set(features + [TARGET]) - set(data.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    x = data[features]
    y = data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3),
    }

    model_metrics = {}
    trained_models = {}
    test_predictions = {}

    for name, model in candidate_models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        preds_clipped = np.clip(preds, 0, 100)

        # Cross validation R2
        cv_scores = cross_val_score(model, x, y, cv=5, scoring="r2")

        metrics = evaluate_predictions(y_test, preds_clipped)
        metrics["cv_r2_mean"] = round(float(np.mean(cv_scores)), 4)
        metrics["cv_r2_std"] = round(float(np.std(cv_scores)), 4)
        metrics["test_samples"] = int(len(y_test))

        model_metrics[name] = metrics
        trained_models[name] = model
        test_predictions[name] = [round(float(p), 2) for p in preds_clipped]

    # Pick best model based on R2 score
    best_model_name = max(model_metrics, key=lambda k: model_metrics[k]["r2"])
    best_model = trained_models[best_model_name]

    # Calculate feature importances
    feature_importance: dict[str, float] = {}
    if hasattr(best_model, "feature_importances_"):
        raw_imp = best_model.feature_importances_
        feature_importance = {feat: round(float(imp * 100), 2) for feat, imp in zip(features, raw_imp)}
    elif hasattr(best_model, "coef_"):
        abs_coefs = np.abs(best_model.coef_)
        total = np.sum(abs_coefs) or 1.0
        feature_importance = {feat: round(float((c / total) * 100), 2) for feat, c in zip(features, abs_coefs)}

    # Also train extended 5-feature model if available in data
    model_5f = None
    metrics_5f = None
    has_extended = all(col in data.columns for col in EXTENDED_FEATURES)
    if has_extended:
        x_ext = data[EXTENDED_FEATURES]
        x_tr_e, x_te_e, y_tr_e, y_te_e = train_test_split(x_ext, y, test_size=0.2, random_state=42)
        rf_ext = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3)
        rf_ext.fit(x_tr_e, y_tr_e)
        preds_ext = np.clip(rf_ext.predict(x_te_e), 0, 100)
        metrics_5f = evaluate_predictions(y_te_e, preds_ext)
        model_5f = {
            "model": rf_ext,
            "features": EXTENDED_FEATURES,
            "metrics": metrics_5f,
            "feature_importance": {
                feat: round(float(imp * 100), 2)
                for feat, imp in zip(EXTENDED_FEATURES, rf_ext.feature_importances_)
            },
        }

    # Dataset correlation summary
    corr_matrix = data[features + [TARGET]].corr().round(3).to_dict()

    artifact = {
        "model": best_model,
        "best_model_name": best_model_name,
        "features": features,
        "candidate_models": trained_models,
        "metrics": model_metrics,
        "feature_importance": feature_importance,
        "correlations": corr_matrix,
        "model_5f": model_5f,
        "extended_features": EXTENDED_FEATURES if has_extended else None,
        "y_test_sample": [round(float(v), 2) for v in y_test.iloc[:50]],
        "y_pred_sample": test_predictions[best_model_name][:50],
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    # Save human-readable metrics JSON
    summary_metrics = {
        "best_model": best_model_name,
        "models": model_metrics,
        "feature_importance": feature_importance,
        "extended_5f_metrics": metrics_5f,
    }
    METRICS_PATH.write_text(json.dumps(summary_metrics, indent=2), encoding="utf-8")

    return artifact


if __name__ == "__main__":
    print("Training Student Performance ML Models...")
    artifact = train_model()
    print(f"Artifacts successfully saved to: {MODEL_PATH}")
    print(f"Best Model Selected: {artifact['best_model_name']}")
    print("\nModel Comparison Table:")
    for model_name, m in artifact["metrics"].items():
        print(f"  • {model_name:20s} | R²: {m['r2']:.4f} | RMSE: {m['rmse']:5.2f} | MAE: {m['mae']:5.2f} | 5-Fold CV: {m['cv_r2_mean']:.4f}")
    print("\nFeature Importance:")
    for feat, imp in artifact["feature_importance"].items():
        print(f"  • {feat:16s}: {imp:.1f}%")
