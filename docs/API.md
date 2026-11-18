# Documentação da API

Base URL local: `http://127.0.0.1:8000`  
Documentação interativa: `http://127.0.0.1:8000/docs`

## Endpoints

### `GET /`
Interface web (HTML) para testar previsões.

### `GET /health`
Verifica se a API está no ar e se o modelo está carregável.

**Resposta 200**
```json
{
  "status": "ok",
  "model_loaded": true,
  "message": "API operacional e modelo disponível."
}
```

Se o arquivo do modelo estiver ausente: `status = "degraded"`, `model_loaded = false`.

### `GET /model-info`
Metadados do modelo, hiperparâmetros, métricas de validação e limitações.

**Resposta 200** — inclui `model`, `features`, `threshold`, `params`, `validation_metrics`, `limitations`.  
**Resposta 503** — modelo não encontrado.

### `POST /predict`
Prevê risco de cancelamento.

**Body (JSON)**
| Campo | Tipo | Intervalo | Descrição |
|-------|------|-----------|-----------|
| `item_count` | int | 1–50 | Quantidade de itens |
| `subtotal` | float | ≥ 0 | Valor dos produtos |
| `delivery_fee` | float | ≥ 0 | Taxa de entrega |
| `estimated_delivery_min` | float | 1–180 | Tempo estimado (min) |
| `order_hour` | int | 0–23 | Hora do pedido |
| `order_weekday` | int | 0–6 | Dia da semana (0=seg) |

**Exemplo**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "item_count": 3,
    "subtotal": 62.5,
    "delivery_fee": 4.99,
    "estimated_delivery_min": 40,
    "order_hour": 20,
    "order_weekday": 5
  }'
```

**Resposta 200**
```json
{
  "probability": 0.42,
  "prediction": 0,
  "threshold": 0.599651,
  "risk_label": "baixo",
  "explanation": "A probabilidade estimada de cancelamento é 42.0%..."
}
```

**Erros**
- `422` — validação Pydantic (campo inválido ou ausente)
- `503` — modelo não encontrado no disco

## Modelo
- Algoritmo: `RandomForestClassifier` (`class_weight='balanced'`)
- Threshold: ≈ 0,5997 (maximiza F1 na **validação**)
- Artefato: `models/cancellation_rf_pipeline.joblib`
