from __future__ import annotations 

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.data.open_meteo_client import OpenMeteoClient
from ml.features.dataset_builder import add_calendar_features, FEATURE_COLUMNS, TARGET_COLUMNS

MODEL_PATH = Path("artifacts/models/weather_rf.joblib")

def load_model_artifact(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Train the model before inference.")
    artifact = joblib.load(model_path)

    required_keys = {"model", "window_size", "horizon"}
    missing_keys = required_keys - artifact.keys()
    if missing_keys:
        raise KeyError(f"Model artifact is missing required keys: {missing_keys}")
    
    return artifact

def get_recent_weather_data(
        latitude: float,
        longitude: float,
        window_size: int,
) -> pd.DataFrame:
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=window_size + 30) # берем с запасом, чтобы точно покрыть окно

    client = OpenMeteoClient()
    df = client.get_daily_weather(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    df = df.sort_values("time").reset_index(drop=True)
    df = df.dropna(subset=FEATURE_COLUMNS) # удаляем строки с пропусками в важных столбцах

    recent_df = df.tail(window_size).copy()

    if len(recent_df) < window_size:
        raise ValueError(f"Not enough recent data to fill the window. Required: {window_size}, "
                         f"available: {len(recent_df)}")
    
    return recent_df

def build_latest_features_vector(
        df: pd.DataFrame,
        window_size: int,
) -> np.ndarray:
    if len(df) < window_size:
        raise ValueError(f"Dataframe has fewer rows than the window size. Required: {window_size}, "
                         f"available: {len(df)}")
    
    feature_df = df.tail(window_size).copy()
    feature_df = add_calendar_features(feature_df)

    feature_columns = FEATURE_COLUMNS + ["day_sin", "day_cos", "month"]

    x = feature_df[feature_columns].to_numpy(dtype=np.float32)
    x = x.reshape(1, -1) # преобразуем в вектор для модели

    return x

def postprocess_day_predictions(
        day_prediction: dict[str,float],
) -> dict[str, float]:
    temp_max = day_prediction["temperature_2m_max"]
    temp_min = day_prediction["temperature_2m_min"]
    precipitation = day_prediction["precipitation_sum"]

    if temp_max < temp_min:
        temp_max, temp_min = temp_min, temp_max

    precipitation = max(0.0, precipitation)

    return {
        "temperature_2m_max": temp_max,
        "temperature_2m_min": temp_min,
        "precipitation_sum": precipitation,
    }

def format_predictions(
        prediction: np.ndarray,
        horizon: int,
        last_observed_date: pd.Timestamp,
        target_columns: list[str] = TARGET_COLUMNS,
) -> list[dict[str, float | str]]:
    result = []
    target_count = len(target_columns)

    for day_idx in range(horizon):
        offset = day_idx * target_count
        forecast_date = last_observed_date + pd.Timedelta(days=day_idx + 1)

        raw_day_prediction = {
            target_columns[target_idx]: float(
                prediction[ offset + target_idx]
            )
            for target_idx in range(target_count)
        }

        processed_day_prediction = postprocess_day_predictions(raw_day_prediction)

        result.append({
            "date": forecast_date.date().isoformat(),
            **processed_day_prediction
        })

    return result

def predict_weather(
        latitude: float,
        longitude: float,
        model_path: Path = MODEL_PATH,
) -> list[dict[str, float | str]]:
    
    artifact = load_model_artifact(model_path)
    model = artifact["model"]
    window_size = artifact["window_size"]
    horizon = artifact["horizon"]
    target_columns = artifact.get("target_columns", TARGET_COLUMNS)

    recent_df = get_recent_weather_data(
        latitude=latitude,
        longitude=longitude,
        window_size=window_size,
    )

    x = build_latest_features_vector(df= recent_df, window_size=window_size)
    prediction = model.predict(x)[0]

    return format_predictions(
        prediction=prediction,
        horizon=horizon,
        last_observed_date=recent_df["time"].max(),
        target_columns=target_columns,
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict weather using a trained model.")
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)

    args = parser.parse_args()

    forecast = predict_weather(
        latitude=args.latitude,
        longitude=args.longitude,
    )

    for day in forecast:
        print(f"Date: {day['date']}, Max Temp: {day['temperature_2m_max']:.1f}°C, "
              f"Min Temp: {day['temperature_2m_min']:.1f}°C, Precipitation: {day['precipitation_sum']:.1f}mm")
        
if __name__ == "__main__":
    main()
