from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import httpx

@dataclass(frozen=True)
class OpenMeteoConfig:
    base_url: str = "https://archive-api.open-meteo.com/v1/archive"


class OpenMeteoClient:
    def __init__(self, config: OpenMeteoConfig | None = None) -> None:
        self.config = config or OpenMeteoConfig()

    def get_daily_weather(
            self,
            latitude: float,
            longitude: float,
            start_date: str,
            end_date: str,
    ) -> pd.DataFrame:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "precipitation_sum",
                "rain_sum",
                "snowfall_sum",
                "wind_speed_10m_max",
                "wind_gusts_10m_max",
            ],
            "timezone": "auto",
        }

        response = httpx.get(
            self.config.base_url,
            params=params,
            timeout=30.0,
        )

        response.raise_for_status()

        data = response.json()
        daily_data = data["daily"]

        df = pd.DataFrame(daily_data)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
        
        return df
