from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
]

TARGET_COLUMNS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame: # добавляем календарные признаки в датасет
    result = df.copy()

    day_of_year = result["time"].dt.dayofyear
    result["day_sin"] = np.sin(2*np.pi* day_of_year/365.25)
    result["day_cos"] = np.cos(2*np.pi* day_of_year/365.25)
    result["month"] = result["time"].dt.month

    return result


def build_supervised_dataset(
        df: pd.DataFrame,
        window_size: int = 30, # берем последние 30 дней и предсказываем следующие 3
        horizon: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[str]]: 
    
    df = add_calendar_features(df)

    feature_columns = FEATURE_COLUMNS + ["day_sin", "day_cos", "month"]

    values = df[feature_columns].to_numpy(dtype = np.float32)
    targets = df[TARGET_COLUMNS].to_numpy(dtype = np.float32)

    x_samples = []
    y_samples = []

    max_start = len(df) - window_size - horizon + 1

    for start_idx in range(max_start):
        input_end_idx = start_idx + window_size
        target_end_idx = input_end_idx + horizon

        x_window = values[start_idx:input_end_idx]
        y_window = targets[input_end_idx:target_end_idx]

        x_samples.append(x_window.reshape(-1))
        y_samples.append(y_window.reshape(-1))

    x = np.asarray(x_samples, dtype= np.float32)
    y = np.asarray(y_samples, dtype= np.float32)

    flattened_feature_names = []
    for day_idx in range(window_size):
        for column in feature_columns:
            flattened_feature_names.append(f"day_{day_idx - window_size + 1}_{column}")

    return x, y, flattened_feature_names