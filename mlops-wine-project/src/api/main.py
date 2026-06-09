"""
FastAPI-сервис для предсказания качества вина.

Эндпоинты:
  POST /predict      – предсказать оценку качества вина
  GET  /healthcheck  – проверка работоспособности сервиса
  GET  /model-info   – информация о загруженной модели
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
MODEL_PATH    = os.getenv("MODEL_PATH",    "models/wine_quality_model.pkl")
METADATA_PATH = os.getenv("METADATA_PATH", "models/metadata.json")

# DVC-указатель модели - именно этот файл есть в репозитории
MODEL_DVC_FILE = MODEL_PATH + ".dvc"

app = FastAPI(
    title="Wine Quality Prediction API",
    description="Предсказывает качество красного вина (0-10) по физико-химическим параметрам.",
    version="1.0.0",
)

# Глобальный объект модели (загружается один раз при старте)
_model = None


def _pull_from_dvc() -> None:
    """
    Скачивает модель из удалённого хранилища через DVC.
    Вызывается только если pkl-файл отсутствует локально.
    Использует .dvc-файл, который реально есть в репозитории.
    """
    if not Path(MODEL_DVC_FILE).exists():
        raise FileNotFoundError(
            f"DVC-указатель {MODEL_DVC_FILE} не найден в репозитории. "
            "Убедитесь, что файл добавлен через `dvc add` и закоммичен."
        )

    print(f"Скачиваем модель из DVC: {MODEL_DVC_FILE}")
    result = subprocess.run(
        ["dvc", "pull", MODEL_DVC_FILE],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dvc pull завершился с ошибкой:\n{result.stderr}")
    print(f"dvc pull выполнен: {result.stdout.strip()}")


def get_model():
    """Возвращает загруженную модель; при первом вызове загружает её с диска."""
    global _model
    if _model is None:
        # Если pkl нет локально - тянем через DVC
        if not Path(MODEL_PATH).exists():
            _pull_from_dvc()
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        print(f"Модель загружена из {MODEL_PATH}")
    return _model


# Схемы данных (Pydantic)
class WineFeatures(BaseModel):
    """Физико-химические характеристики вина - входные данные для предсказания."""
    fixed_acidity: float        = Field(..., gt=0, example=7.4)    # нелетучая кислотность
    volatile_acidity: float     = Field(..., gt=0, example=0.70)   # летучая кислотность
    citric_acid: float          = Field(..., ge=0, example=0.00)   # лимонная кислота
    residual_sugar: float       = Field(..., gt=0, example=1.9)    # остаточный сахар
    chlorides: float            = Field(..., gt=0, example=0.076)  # хлориды
    free_sulfur_dioxide: float  = Field(..., gt=0, example=11.0)   # свободный SO₂
    total_sulfur_dioxide: float = Field(..., gt=0, example=34.0)   # общий SO₂
    density: float              = Field(..., gt=0, example=0.9978) # плотность
    pH: float                   = Field(..., gt=0, example=3.51)   # кислотность pH
    sulphates: float            = Field(..., gt=0, example=0.56)   # сульфаты
    alcohol: float              = Field(..., gt=0, example=9.4)    # содержание алкоголя

    class Config:
        json_schema_extra = {
            "example": {
                "fixed_acidity": 7.4, "volatile_acidity": 0.70,
                "citric_acid": 0.00, "residual_sugar": 1.9,
                "chlorides": 0.076, "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0, "density": 0.9978,
                "pH": 3.51, "sulphates": 0.56, "alcohol": 9.4,
            }
        }


class PredictionResponse(BaseModel):
    """Ответ сервиса: числовая оценка и текстовая метка качества."""
    quality_score: float   # предсказанный балл (0-10)
    quality_label: str     # словесная оценка: poor / good / excellent


# Эндпоинты
@app.post("/predict", response_model=PredictionResponse, tags=["Предсказание"])
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
        label = "excellent"   # отличное
    elif score >= 5:
        label = "good"        # хорошее
    else:
        label = "poor"        # плохое

    return PredictionResponse(quality_score=score, quality_label=label)


@app.get("/healthcheck", tags=["Мониторинг"])
def healthcheck() -> dict:
    """Проверка работоспособности сервиса."""
    return {"status": "ok"}


@app.get("/model-info", tags=["Мониторинг"])
def model_info() -> dict:
    """Возвращает метаданные о текущей загруженной модели."""
    try:
        model = get_model()

        # Читаем метаданные из файла, если он существует
        meta: dict = {}
        if Path(METADATA_PATH).exists():
            with open(METADATA_PATH) as f:
                meta = json.load(f)

        return {
            "model_type": type(model).__name__,
            "model_path": MODEL_PATH,
            "dvc_pointer": MODEL_DVC_FILE,
            "metadata": meta,
            "features": list(WineFeatures.model_fields.keys()),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# Событие запуска приложения
@app.on_event("startup")
def startup_event():
    """Предзагружает модель при старте (включая dvc pull если нужно)."""
    try:
        get_model()
        print("Модель успешно загружена при запуске сервиса.")
    except Exception as exc:
        # Логируем предупреждение, но не падаем - healthcheck должен работать
        print(f"Предупреждение: не удалось предзагрузить модель - {exc}")
