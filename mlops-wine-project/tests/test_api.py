"""
Тесты для API предсказания качества вина.

Запуск:
    pytest tests/ -v
"""

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Тестовые данные

# Валидный образец вина - все 11 обязательных признаков
VALID_WINE = {
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


# Фикстуры
@pytest.fixture(scope="module")
def mock_model():
    """Лёгкий мок, имитирующий интерфейс sklearn-модели."""
    model = MagicMock()
    model.predict.return_value = np.array([6.5])
    return model


@pytest.fixture(scope="module")
def client(mock_model, tmp_path_factory):
    """Создаёт тестовый клиент с подменённым загрузчиком модели."""
    tmp = tmp_path_factory.mktemp("models")
    # Сериализуем мок-модель во временный файл
    model_file = tmp / "wine_quality_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(mock_model, f)

    # Создаём тестовые метаданные
    metadata = {"version": "test", "trained_at": "2024-01-01T00:00:00"}
    meta_file = tmp / "metadata.json"
    meta_file.write_text(json.dumps(metadata))
    # Патчим пути и модель, чтобы не зависеть от реальных файлов
    with patch("src.api.main.MODEL_PATH", str(model_file)), \
         patch("src.api.main.METADATA_PATH", str(meta_file)), \
         patch("src.api.main._model", mock_model):

        from src.api.main import app
        yield TestClient(app)


# Тесты /healthcheck
class TestHealthcheck:
    def test_returns_200(self, client):
        resp = client.get("/healthcheck")
        assert resp.status_code == 200

    def test_body_ok(self, client):
        resp = client.get("/healthcheck")
        assert resp.json() == {"status": "ok"}


# Тесты /predict
class TestPredict:
    def test_valid_input_returns_200(self, client):
        resp = client.post("/predict", json=VALID_WINE)
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client):
        resp = client.post("/predict", json=VALID_WINE)
        data = resp.json()
        assert "quality_score" in data
        assert "quality_label" in data

    def test_score_is_numeric(self, client):
        resp = client.post("/predict", json=VALID_WINE)
        assert isinstance(resp.json()["quality_score"], float)

    def test_label_is_valid(self, client):
        resp = client.post("/predict", json=VALID_WINE)
        assert resp.json()["quality_label"] in ("excellent", "good", "poor")

    def test_missing_field_returns_422(self, client):
        bad = {k: v for k, v in VALID_WINE.items() if k != "alcohol"}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_negative_acidity_returns_422(self, client):
        bad = {**VALID_WINE, "fixed_acidity": -1.0}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_empty_body_returns_422(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_high_quality_label(self, client, mock_model):
        mock_model.predict.return_value = np.array([8.0])
        resp = client.post("/predict", json=VALID_WINE)
        assert resp.json()["quality_label"] == "excellent"

    def test_low_quality_label(self, client, mock_model):
        mock_model.predict.return_value = np.array([3.0])
        resp = client.post("/predict", json=VALID_WINE)
        assert resp.json()["quality_label"] == "poor"


# Тесты /model-info
class TestModelInfo:
    def test_returns_200(self, client):
        resp = client.get("/model-info")
        assert resp.status_code == 200

    def test_has_model_type(self, client):
        resp = client.get("/model-info")
        assert "model_type" in resp.json()

    def test_has_features_list(self, client):
        resp = client.get("/model-info")
        data = resp.json()
        assert "features" in data
        assert isinstance(data["features"], list)
        assert len(data["features"]) == 11


# Тест независимости тестов
class TestIndependence:
    """Проверяем, что тесты не влияют на состояние друг друга."""

    def test_predict_is_idempotent(self, client, mock_model):
        mock_model.predict.return_value = np.array([6.5])
        r1 = client.post("/predict", json=VALID_WINE).json()
        r2 = client.post("/predict", json=VALID_WINE).json()
        assert r1 == r2
