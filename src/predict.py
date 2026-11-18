"""
Inferência do modelo de cancelamento de pedidos.

Uso:
    from predict import predict_cancellation
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


def _load() -> None:
    global _pipeline, _threshold
    if _pipeline is not None:
        return
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_PATH}. "
            "Treine o modelo executando o notebook."
        )
    _pipeline = joblib.load(MODEL_PATH)
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        _threshold = float(meta.get("threshold", 0.5))


def predict_cancellation(order: dict[str, Any]) -> dict[str, Any]:
    """Prevê risco de cancelamento a partir das features do pedido."""
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
        "threshold": _threshold,
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
