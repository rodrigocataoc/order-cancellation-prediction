"""Fixtures compartilhadas dos testes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402
from src.predict import reset_model_cache  # noqa: E402


@pytest.fixture
def client():
    reset_model_cache()
    with TestClient(app) as c:
        yield c
    reset_model_cache()


@pytest.fixture
def valid_order():
    return {
        "item_count": 3,
        "subtotal": 62.5,
        "delivery_fee": 4.99,
        "estimated_delivery_min": 40,
        "order_hour": 20,
        "order_weekday": 5,
    }
