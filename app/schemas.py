"""Schemas Pydantic para validação de entrada/saída da API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OrderFeatures(BaseModel):
    """Features disponíveis no momento do pedido (sem data leakage)."""

    item_count: int = Field(
        ...,
        ge=1,
        le=50,
        description="Quantidade de itens no pedido",
        examples=[3],
    )
    subtotal: float = Field(
        ...,
        ge=0,
        le=10_000,
        description="Valor dos produtos (antes de taxa e gorjeta)",
        examples=[62.5],
    )
    delivery_fee: float = Field(
        ...,
        ge=0,
        le=100,
        description="Taxa de entrega",
        examples=[4.99],
    )
    estimated_delivery_min: float = Field(
        ...,
        ge=1,
        le=180,
        description="Tempo estimado de entrega em minutos",
        examples=[40],
    )
    order_hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hora do pedido (0–23)",
        examples=[20],
    )
    order_weekday: int = Field(
        ...,
        ge=0,
        le=6,
        description="Dia da semana (0=segunda … 6=domingo)",
        examples=[5],
    )


class PredictionResponse(BaseModel):
    """Resposta da previsão de cancelamento."""

    probability: float = Field(..., description="Probabilidade estimada de cancelamento (0–1)")
    prediction: int = Field(..., description="0 = entregue, 1 = cancelado")
    threshold: float = Field(..., description="Threshold usado na decisão")
    risk_label: Literal["alto", "baixo"] = Field(
        ..., description="Rótulo de risco derivado do threshold"
    )
    explanation: str = Field(..., description="Explicação simples do resultado")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str


class ModelInfoResponse(BaseModel):
    model: str
    features: list[str]
    threshold: float
    threshold_selection: str | None = None
    params: dict | None = None
    validation_metrics: dict | None = None
    note: str | None = None
    limitations: list[str]
