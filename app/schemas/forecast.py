from __future__ import annotations

from pydantic import BaseModel, Field

class ForecastRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class ForecastDay(BaseModel):
    date: str
    temperature_2m_max: float
    temperature_2m_min: float
    precipitation_sum: float


class ForecastResponse(BaseModel):
    model: str
    horizon: int
    forecast: list[ForecastDay]