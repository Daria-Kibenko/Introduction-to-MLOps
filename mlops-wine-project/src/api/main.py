"""
FastAPI-сервис для предсказания качества вина.

Эндпоинты:
  POST /predict      - предсказать оценку качества вина
  GET  /healthcheck  - проверка работоспособности сервиса
  GET  /model-info   - информация о загруженной модели
"""

import json
import os
import pickle
import subprocess
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Конфигурация
MODEL_PATH = os.getenv("MODEL_PATH", "models/wine_quality_model.pkl")
METADATA_PATH = "models/metadata.json"

app = FastAPI(
    title="Wine Quality Prediction API",
    description="Предсказывает качество красного вина (0-10) по физико-химическим параметрам.",
    version="1.0.0",
)

# Глобальный объект модели
_model = None


def _pull_model_from_dvc(path: str) -> None:
    """Скачивает файл модели из удалённого хранилища DVC, если он отсутствует."""
    result = subprocess.run(
        ["dvc", "pull", path + ".dvc"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dvc pull failed: {result.stderr}")


def get_model():
    """Возвращает загруженную модель; при первом вызове загружает её с диска."""
    global _model
    if _model is None:
        # Если файла нет локально - тянем из DVC
        if not Path(MODEL_PATH).exists():
            _pull_model_from_dvc(MODEL_PATH)
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


# Схемы данных (Pydantic)
class WineFeatures(BaseModel):
    """Физико-химические характеристики вина — входные данные для предсказания."""
    fixed_acidity: float       = Field(..., gt=0, example=7.4)
    volatile_acidity: float    = Field(..., gt=0, example=0.70)
    citric_acid: float         = Field(..., ge=0, example=0.00)
    residual_sugar: float      = Field(..., gt=0, example=1.9)
    chlorides: float           = Field(..., gt=0, example=0.076)
    free_sulfur_dioxide: float = Field(..., gt=0, example=11.0)
    total_sulfur_dioxide: float= Field(..., gt=0, example=34.0)
    density: float             = Field(..., gt=0, example=0.9978)
    pH: float                  = Field(..., gt=0, example=3.51)
    sulphates: float           = Field(..., gt=0, example=0.56)
    alcohol: float             = Field(..., gt=0, example=9.4)

    class Config:
        json_schema_extra = {
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.70,
                "citric_acid": 0.00,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
            }
        }


class PredictionResponse(BaseModel):
    """Ответ сервиса: числовая оценка и текстовая метка качества."""
    quality_score: float  # предсказанный балл (0-10)
    quality_label: str    # словесная оценка: poor / good / excellent


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: WineFeatures) -> PredictionResponse:
    """
    Предсказывает качество вина (0-10) по физико-химическим параметрам.
    Возвращает числовой балл и текстовую метку качества.
    """
    model = get_model()

    # Формируем датафрейм из одной строки для совместимости со sklearn
    data = pd.DataFrame([features.model_dump()])
    score = float(model.predict(data)[0])
    score = round(score, 2)

    # Определяем словесную оценку по пороговым значениям
    if score >= 7:
        label = "excellent"
    elif score >= 5:
        label = "good"
    else:
        label = "poor"

    return PredictionResponse(quality_score=score, quality_label=label)


@app.get("/healthcheck", tags=["Monitoring"])
def healthcheck() -> dict:
    """Проверка работоспособности сервиса - возвращает ok если сервис запущен."""
    return {"status": "ok"}


@app.get("/model-info", tags=["Monitoring"])
def model_info() -> dict:
    """Возвращает метаданные о текущей загруженной модели."""
    try:
        model = get_model()

        # Читаем метаданные, если файл существует
        meta: dict = {}
        if Path(METADATA_PATH).exists():
            with open(METADATA_PATH) as f:
                meta = json.load(f)

        return {
            "model_type": type(model).__name__,  # тип модели (класс sklearn)
            "model_path": MODEL_PATH,            # путь к файлу модели
            "metadata": meta,                    # метаданные из DVC-пайплайна
            "features": list(WineFeatures.model_fields.keys()),  # список признаков
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# Событие запуска приложения
@app.on_event("startup")
def startup_event():
    """Предзагружает модель при старте, чтобы первый запрос не тормозил."""
    try:
        get_model()
        print("Модель успешно загружена при запуске сервиса.")
    except Exception as exc:
        print(f"Предупреждение: не удалось предзагрузить модель - {exc}")
