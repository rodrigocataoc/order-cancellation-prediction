"""
Inferência do modelo de cancelamento de pedidos.

Uso:
    from src.predict import predict_cancellation
    result = predict_cancellation({
        "item_count": 2,
        "subtotal": 45.0,
        "delivery_fee": 4.99,
        "estimated_delivery_min": 35,
        "order_hour": 20,
        "order_weekday": 5,
    })
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "cancellation_rf_pipeline.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

_FEATURES = [
    "item_count",
    "subtotal",
    "delivery_fee",
    "estimated_delivery_min",
    "order_hour",
    "order_weekday",
]

_pipeline = None
_threshold = 0.5
_metadata: dict[str, Any] | None = None


def reset_model_cache() -> None:
    """Limpa o cache em memória (útil em testes)."""
    global _pipeline, _threshold, _metadata
    _pipeline = None
    _threshold = 0.5
    _metadata = None


def get_model_path() -> Path:
    return MODEL_PATH


def model_exists() -> bool:
    return MODEL_PATH.exists()


def _load() -> None:
    global _pipeline, _threshold, _metadata
    if _pipeline is not None:
        return
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_PATH}. "
            "Execute: python scripts/train_and_export.py"
        )
    _pipeline = joblib.load(MODEL_PATH)
    if META_PATH.exists():
        _metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
        _threshold = float(_metadata.get("threshold", 0.5))
    else:
        _metadata = {}


def get_metadata() -> dict[str, Any]:
    """Retorna metadados do modelo (carrega se necessário)."""
    _load()
    return dict(_metadata or {})


def get_threshold() -> float:
    _load()
    return float(_threshold)


def predict_cancellation(order: dict[str, Any]) -> dict[str, Any]:
    """
    Prevê risco de cancelamento a partir das features do pedido.

    Returns
    -------
    dict com probability, prediction (0/1), threshold, risk_label.
    """
    _load()
    missing = [f for f in _FEATURES if f not in order]
    if missing:
        raise ValueError(f"Features ausentes: {missing}")

    row = pd.DataFrame([{f: order[f] for f in _FEATURES}])
    proba = float(_pipeline.predict_proba(row)[0, 1])
    pred = int(proba >= _threshold)
    return {
        "probability": round(proba, 4),
        "prediction": pred,
        "threshold": round(float(_threshold), 6),
        "risk_label": "alto" if pred == 1 else "baixo",
    }


if __name__ == "__main__":
    example = {
        "item_count": 3,
        "subtotal": 62.5,
        "delivery_fee": 4.99,
        "estimated_delivery_min": 40,
        "order_hour": 20,
        "order_weekday": 5,
    }
    print(predict_cancellation(example))
