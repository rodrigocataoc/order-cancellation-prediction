"""Testes da API FastAPI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "cancellation_rf_pipeline.joblib"


def test_health_ok(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert "status" in body
    assert "model_loaded" in body
    assert body["model_loaded"] is True
    assert body["status"] == "ok"


def test_root_returns_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "Cancelamento" in res.text or "cancelamento" in res.text.lower()


def test_model_info(client):
    res = client.get("/model-info")
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == "RandomForestClassifier"
    assert "item_count" in body["features"]
    assert body["threshold"] == pytest.approx(0.5996514109885884, rel=1e-6)
    assert body["validation_metrics"] is not None
    assert "limitations" in body
    assert len(body["limitations"]) >= 1


def test_predict_success(client, valid_order):
    res = client.post("/predict", json=valid_order)
    assert res.status_code == 200
    body = res.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["risk_label"] in ("alto", "baixo")
    assert "threshold" in body
    assert "explanation" in body
    assert len(body["explanation"]) > 10


def test_predict_invalid_hour(client, valid_order):
    payload = {**valid_order, "order_hour": 25}
    res = client.post("/predict", json=payload)
    assert res.status_code == 422


def test_predict_invalid_weekday(client, valid_order):
    payload = {**valid_order, "order_weekday": 9}
    res = client.post("/predict", json=payload)
    assert res.status_code == 422


def test_predict_missing_field(client, valid_order):
    payload = {k: v for k, v in valid_order.items() if k != "subtotal"}
    res = client.post("/predict", json=payload)
    assert res.status_code == 422


def test_predict_negative_subtotal(client, valid_order):
    payload = {**valid_order, "subtotal": -10}
    res = client.post("/predict", json=payload)
    assert res.status_code == 422


def test_predict_without_model_file(client, valid_order):
    with patch("app.main.model_exists", return_value=False):
        res = client.post("/predict", json=valid_order)
        assert res.status_code == 503
        assert "Modelo" in res.json()["detail"] or "modelo" in res.json()["detail"].lower()


def test_model_info_without_model_file(client):
    with patch("app.main.model_exists", return_value=False):
        res = client.get("/model-info")
        assert res.status_code == 503


def test_health_degraded_without_model(client):
    with patch("app.main.model_exists", return_value=False):
        res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["model_loaded"] is False
        assert body["status"] == "degraded"


def test_model_file_present():
    assert MODEL_PATH.exists(), (
        f"Arquivo do modelo ausente: {MODEL_PATH}. "
        "Execute python scripts/train_and_export.py"
    )
