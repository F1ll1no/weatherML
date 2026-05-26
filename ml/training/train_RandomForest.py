from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ml.data.open_meteo_client import OpenMeteoClient
from ml.features.dataset_builder import build_supervised_dataset, TARGET_COLUMNS
from ml.models.random_forest import create_random_forest_model

MODEL_PATH = Path("artifacts/models/weather_rf.joblib")
METRICS_PATH = Path("artifacts/metrics/weather_rf_metrics.json")

def train_test_split_by_time(
        x: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_idx = int(len(x) * (1 - test_size))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]

    return x_train, x_test, y_train, y_test

def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizon: int,
) -> dict:
    metrics = { # общие метрики + отдельный на каждый таргет на каждый день горизонта
        "overall_rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "overall_mae": mean_absolute_error(y_true, y_pred),
        "by_target": {},
    }

    for day_idx in range(horizon):
        for target_idx, target_name in enumerate(TARGET_COLUMNS):
            column_idx = day_idx * len(TARGET_COLUMNS) + target_idx

            mae = mean_absolute_error(y_true[:, column_idx], y_pred[:, column_idx])
            rmse = mean_squared_error(y_true[:, column_idx], y_pred[:, column_idx]) ** 0.5

            key = f"day_{day_idx+1}_{target_name}"
            metrics["by_target"][key] = {
                "rmse": rmse, 
                "mae": mae,
                }
            
    return metrics

def make_presistence_baseline( # baseline для понимания полезности модели.
        x_test: np.ndarray,
        window_size: int,
        horizon: int,
        feature_count: int,
) -> np.ndarray:
    preds = []

    temp_max_idx = 0
    temp_min_idx = 1
    precipitation_idx = 3

    for sample in x_test:
        window = sample.reshape(window_size, feature_count)
        last_day = window[-1]

        one_day_prediction = [
            last_day[temp_max_idx],
            last_day[temp_min_idx],
            last_day[precipitation_idx],
        ]

        sample_prediction = one_day_prediction * horizon
        preds.append(sample_prediction)

    return np.asarray(preds, dtype=np.float32)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Random Forest model for weather forecasting")
    parser.add_argument("--latitude", type=float, required=True, help="Latitude for weather data")
    parser.add_argument("--longitude", type=float, required=True, help="Longitude for weather data")
    parser.add_argument("--start_date", type=str, dest="start_date", required=True, help="Start date for weather data (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, dest="end_date", required=True, help="End date for weather data (YYYY-MM-DD)")
    parser.add_argument("--window_size", type=int, dest="window_size", default=30, help="Window size for supervised dataset")
    parser.add_argument("--horizon", type=int, dest="horizon", default=3, help="Horizon for supervised dataset")

    args = parser.parse_args()

    client = OpenMeteoClient()
    df = client.get_daily_weather(
        latitude=args.latitude,
        longitude=args.longitude,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    x, y, feature_names  = build_supervised_dataset(df=df, 
                                       window_size=args.window_size, 
                                       horizon=args.horizon)
    
    x_train, x_test, y_train, y_test = train_test_split_by_time(x, y)

    model = create_random_forest_model()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    feature_count = len(feature_names) // args.window_size

    baseline_pred = make_presistence_baseline(
        x_test=x_test,
        window_size=args.window_size,
        horizon=args.horizon,
        feature_count=feature_count,
    )

    model_metrics = calculate_metrics(
        y_true=y_test,
        y_pred=y_pred,
        horizon=args.horizon,
    )

    baseline_metrics = calculate_metrics(
        y_true=y_test,
        y_pred=baseline_pred,
        horizon=args.horizon,
    )

    metrics = {
    "model": "RandomForestRegressor",
    "window_size": args.window_size,
    "horizon": args.horizon,
    "random_forest": model_metrics,
    "persistence_baseline": baseline_metrics,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": model,
        "window_size": args.window_size,
        "horizon": args.horizon,
        "target_columns": TARGET_COLUMNS,
    }, 
    MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding = "utf-8",)
    
    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()