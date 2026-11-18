"""
API FastAPI — Order Cancellation Prediction

Endpoints:
  GET  /           → página HTML da interface
  GET  /health     → health check
  GET  /model-info → metadados e métricas do modelo
  POST /predict    → previsão de risco de cancelamento
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Garante import de src/ a partir da raiz do projeto
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas import (  # noqa: E402
    HealthResponse,
    ModelInfoResponse,
    OrderFeatures,
    PredictionResponse,
)
from src.predict import (  # noqa: E402
    get_metadata,
    model_exists,
    predict_cancellation,
)

app = FastAPI(
    title="Order Cancellation Prediction API",
    description=(
        "API de portfólio para prever risco de cancelamento de pedidos de delivery. "
        "Modelo: RandomForestClassifier com threshold escolhido na validação. "
        "Dados sintéticos (CC BY-NC 4.0) — uso educacional."
    ),
    version="1.0.0",
    contact={"name": "Rodrigo Catão"},
)

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _build_explanation(probability: float, risk_label: str, threshold: float) -> str:
    pct = probability * 100
    th_pct = threshold * 100
    if risk_label == "alto":
        return (
            f"A probabilidade estimada de cancelamento é {pct:.1f}%, "
            f"acima do limiar de decisão ({th_pct:.1f}%). "
            "O modelo classifica este pedido como risco alto. "
            "Atenção: o desempenho do modelo é modesto e os dados são sintéticos."
        )
    return (
        f"A probabilidade estimada de cancelamento é {pct:.1f}%, "
        f"abaixo do limiar de decisão ({th_pct:.1f}%). "
        "O modelo classifica este pedido como risco baixo. "
        "Atenção: o desempenho do modelo é modesto e os dados são sintéticos."
    )


@app.get("/", response_class=HTMLResponse, tags=["Interface"])
async def home(request: Request):
    """Interface web para testar previsões."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "Order Cancellation Prediction"},
    )


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health():
    """Verifica se a API está no ar e se o modelo está disponível."""
    loaded = model_exists()
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        message=(
            "API operacional e modelo disponível."
            if loaded
            else "API no ar, mas o arquivo do modelo não foi encontrado."
        ),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Modelo"])
async def model_info():
    """Retorna metadados, hiperparâmetros e métricas de validação do modelo."""
    if not model_exists():
        raise HTTPException(
            status_code=503,
            detail="Modelo não encontrado. Execute scripts/train_and_export.py.",
        )
    try:
        meta = get_metadata()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Erro ao carregar metadados: {exc}") from exc

    return ModelInfoResponse(
        model=meta.get("model", "RandomForestClassifier"),
        features=meta.get("features", []),
        threshold=float(meta.get("threshold", 0.5)),
        threshold_selection=meta.get("threshold_selection"),
        params=meta.get("params"),
        validation_metrics=meta.get("validation_metrics_at_threshold"),
        note=meta.get("note"),
        limitations=[
            "Dados sintéticos — não representam operação real de delivery.",
            "Amostra pequena (2.000 pedidos; ~8,6% cancelados).",
            "Desempenho preditivo modesto (F1 ≈ 0,26 na validação).",
            "Threshold escolhido apenas na validação para maximizar F1.",
            "Uso educacional / portfólio — licença dos dados: CC BY-NC 4.0.",
        ],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Previsão"])
async def predict(order: OrderFeatures):
    """
    Prevê o risco de cancelamento de um pedido.

    Utiliza apenas features disponíveis no momento do pedido
    (sem data leakage de variáveis pós-evento).
    """
    if not model_exists():
        raise HTTPException(
            status_code=503,
            detail="Modelo não encontrado. Execute scripts/train_and_export.py.",
        )
    try:
        result = predict_cancellation(order.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Erro na previsão: {exc}") from exc

    explanation = _build_explanation(
        result["probability"], result["risk_label"], result["threshold"]
    )
    return PredictionResponse(
        probability=result["probability"],
        prediction=result["prediction"],
        threshold=result["threshold"],
        risk_label=result["risk_label"],
        explanation=explanation,
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):  # noqa: ARG001
    return JSONResponse(status_code=404, content={"detail": "Endpoint não encontrado."})
