from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.routes_forecast import get_forecast_service
from app.main import app


class FakeForecastService:
    def get_forecast(self, request):
        return {
            "model": "RandomForestRegressor",
            "horizon": 3,
            "forecast": [
                {
                    "date": "2026-06-02",
                    "temperature_2m_max": 17.5,
                    "temperature_2m_min": 8.1,
                    "precipitation_sum": 0.4,
                },
                {
                    "date": "2026-06-03",
                    "temperature_2m_max": 18.2,
                    "temperature_2m_min": 9.0,
                    "precipitation_sum": 1.1,
                },
                {
                    "date": "2026-06-04",
                    "temperature_2m_max": 16.8,
                    "temperature_2m_min": 7.7,
                    "precipitation_sum": 0.0,
                },
            ],
        }


def test_forecast_route() -> None:
    app.dependency_overrides[get_forecast_service] = lambda: FakeForecastService()

    client = TestClient(app)

    response = client.post(
        "/forecast",
        json={
            "latitude": 60.1695,
            "longitude": 24.9354,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "RandomForestRegressor"
    assert data["horizon"] == 3
    assert len(data["forecast"]) == 3
    assert data["forecast"][0]["temperature_2m_max"] == 17.5