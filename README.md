# WeatherAI / Weather ML Service

**WeatherAI** is a machine learning powered weather forecasting service.  
The project is built as a portfolio-grade ML application with a clear data pipeline, baseline comparison, model training, inference logic and future API/UI integration.

At the current MVP stage, the service uses historical weather data from Open-Meteo and trains a lightweight `scikit-learn` model to predict weather for the next 3 days.

---

# WeatherAI / Сервис прогноза погоды на ML

**WeatherAI** — это погодный сервис с использованием машинного обучения.  
Проект создается как полноценное портфолио-приложение: с нормальной архитектурой, пайплайном данных, обучением модели, сравнением с baseline, инференсом и дальнейшей интеграцией через API/UI.

На текущем MVP-этапе сервис использует исторические погодные данные Open-Meteo и обучает легкую модель из `scikit-learn` для прогноза погоды на следующие 3 дня.

---

## Project Goal / Цель проекта

The goal is not just to train a model, but to build a real application around it:

- collect historical weather data;
- transform time series into supervised ML features;
- train a lightweight baseline model;
- compare it with a naive persistence baseline;
- save the trained model;
- build inference logic;
- later expose predictions through FastAPI and a user interface.

Цель проекта — не просто обучить модель, а собрать вокруг нее полноценное приложение:

- получать исторические погодные данные;
- преобразовывать временной ряд в признаки для ML;
- обучать легкую baseline-модель;
- сравнивать ее с наивным прогнозом;
- сохранять обученную модель;
- реализовать инференс;
- позже подключить FastAPI и пользовательский интерфейс.

---

## Current MVP Scope / Текущий MVP

Current model:

- source: Open-Meteo Historical Weather API;
- location: one selected location by latitude and longitude;
- model: `RandomForestRegressor`;
- forecast horizon: 3 days;
- input window: previous 30 days;
- task type: multi-output regression.

Текущая модель:

- источник данных: Open-Meteo Historical Weather API;
- локация: одна выбранная точка по широте и долготе;
- модель: `RandomForestRegressor`;
- горизонт прогноза: 3 дня;
- входное окно: предыдущие 30 дней;
- тип задачи: multi-output regression.

The model predicts:

- daily maximum temperature;
- daily minimum temperature;
- daily precipitation sum.

Модель прогнозирует:

- максимальную дневную температуру;
- минимальную дневную температуру;
- сумму осадков за день.

---

## Tech Stack / Стек технологий

- Python 3.11+
- uv
- pandas
- numpy
- scikit-learn
- httpx
- joblib
- FastAPI planned
- Streamlit planned
- Docker planned

---

## Project Structure / Структура проекта

```text
weatherAI/
│
├── app/
│   ├── main.py
│   ├── schemas/
│   └── services/
│
├── ml/
│   ├── data/
│   │   └── open_meteo_client.py
│   ├── features/
│   │   └── dataset_builder.py
│   ├── models/
│   │   └── random_forest.py
│   ├── training/
│   │   └── train_RandomForest.py
│   └── inference/
│       └── predict.py
│
├── artifacts/
│   ├── models/
│   │   └── weather_rf.joblib
│   └── metrics/
│       └── weather_rf_metrics.json
│
├── tests/
│
├── pyproject.toml
├── uv.lock
└── README.md