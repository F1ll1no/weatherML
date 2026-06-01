from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends

from app.schemas.forecast import ForecastRequest, ForecastResponse
from app.services.forecast_service import ForecastService

router = APIRouter(tags=["forecast"])

def get_forecast_service() -> ForecastService:
    return ForecastService()

@router.post("/forecast", response_model=ForecastResponse)
def create_forecast(
    request: ForecastRequest,
    service: ForecastService = Depends(get_forecast_service),
) -> ForecastResponse:
    try:
        return service.get_forecast(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc