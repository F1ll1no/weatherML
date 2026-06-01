from __future__ import annotations

from fastapi import FastAPI

from app.api.routes_forecast import router as forecast_router
from app.api.routes_health import router as health_router

app = FastAPI(title="Weather Forecast API",
                description="API for providing weather forecasts based on machine learning models.",
                version="0.1.0")

app.include_router(health_router)
app.include_router(forecast_router)

@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Weather Forecast API", "version": "0.1.0",
            "status": "running",
            }