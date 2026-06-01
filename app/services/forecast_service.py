from __future__ import annotations

from pathlib import Path
from typing import Any

from app.schemas.forecast import ForecastRequest, ForecastResponse
from ml.inference.predict import MODEL_PATH, load_model_artifact, predict_weather


class ForecastService:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path

    def get_forecast(self, request: ForecastRequest) -> ForecastResponse:
        artifact = load_model_artifact(self.model_path)

        forecast = predict_weather(
            latitude=request.latitude,
            longitude=request.longitude,
            model_path=self.model_path,
        )

        return ForecastResponse(
            model=self._get_model_name(artifact),
            horizon=artifact["horizon"],
            forecast=forecast,
        )

    @staticmethod
    def _get_model_name(artifact: dict[str, Any]) -> str:
        model = artifact["model"]

        if hasattr(model, "named_steps") and "model" in model.named_steps:
            return model.named_steps["model"].__class__.__name__

        return model.__class__.__name__