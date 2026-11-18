# Order Cancellation Prediction

> Portfólio de Machine Learning + API: prever se um pedido de delivery será **entregue** ou **cancelado**.

**Autor:** Rodrigo Catão  
**Objetivo:** estágio / primeiro emprego em IA, ML e Ciência de Dados  
**Stack:** scikit-learn · FastAPI · Pydantic · pytest

---

## Visão geral

Classificação binária com features disponíveis **no momento do pedido** (sem data leakage).

| Classe | Significado |
|--------|-------------|
| `0` | Entregue |
| `1` | Cancelado |

Com ~91% de entregas, **acurácia isolada é enganosa**. O baseline (`DummyClassifier`) atinge 91,5% sem identificar nenhum cancelamento.

**Dataset:** [mindweave/food-delivery-orders](https://huggingface.co/datasets/mindweave/food-delivery-orders) (sintético, **CC BY-NC 4.0** — atribuição; uso comercial não permitido).

---

## Demonstração rápida

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Confirme: ls models/cancellation_rf_pipeline.joblib
uvicorn app.main:app --reload --port 8000
```

- Interface: http://127.0.0.1:8000  
- Docs OpenAPI: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/health  

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"item_count":3,"subtotal":62.5,"delivery_fee":4.99,"estimated_delivery_min":40,"order_hour":20,"order_weekday":5}'
```

---

## Modelo

| Item | Valor |
|------|--------|
| Algoritmo | RandomForestClassifier |
| Pipeline | scikit-learn Pipeline |
| `n_estimators` | 300 |
| `max_depth` | 5 |
| `class_weight` | balanced |
| `random_state` | 42 |
| Threshold (F1 máx. na validação) | ≈ **0,5997** |
| Artefato | `models/cancellation_rf_pipeline.joblib` |

### Features (sem leakage)

`item_count`, `subtotal`, `delivery_fee`, `estimated_delivery_min`, `order_hour`, `order_weekday`

Excluídas: `actual_delivery_min`, `cancellation_reason`, `rating`, IDs, `tip`, `total`.

### Métricas na validação (threshold ótimo F1)

| Modelo | Threshold | Precision | Recall | F1 |
|--------|----------:|----------:|-------:|---:|
| Regressão Logística | ≈ 0,533 | 0,113 | 0,486 | 0,184 |
| **Random Forest** | ≈ 0,600 | **0,308** | 0,229 | **0,262** |

Matriz RF (validação): `[[347, 18], [27, 8]]`

**Interpretação:** desempenho modesto — adequado a portfólio e estudo, não a produção com estes dados. O teste já foi inspecionado em experimentos anteriores e não deve ser tratado como hold-out intocado.

---

## API

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/` | Interface web |
| `GET` | `/health` | Saúde da API e do modelo |
| `GET` | `/model-info` | Metadados e métricas |
| `POST` | `/predict` | Previsão de risco |

Detalhes: [docs/API.md](docs/API.md)

---

## Estrutura

```text
order-cancellation-prediction/
├── app/
│   ├── main.py              # FastAPI
│   ├── schemas.py           # Pydantic
│   └── templates/index.html # Interface
├── src/predict.py           # Inferência
├── models/
│   ├── cancellation_rf_pipeline.joblib
│   └── model_metadata.json
├── scripts/train_and_export.py
├── notebooks/01_data_exploration.ipynb
├── tests/                   # pytest
├── docs/API.md · docs/DEPLOY.md
├── Dockerfile · render.yaml
├── requirements.txt
└── README.md
```

---

## Desenvolvimento

```bash
pip install -r requirements.txt

# Testes
pytest -v

# Treinar / exportar modelo e gráficos (precisa de data/orders.csv)
python scripts/train_and_export.py
```

Instruções de deploy (local, Docker, Render): [docs/DEPLOY.md](docs/DEPLOY.md)

---

## Limitações (transparência)

1. Dados **sintéticos** — não representam operação real.  
2. Poucos eventos da classe positiva (172 cancelamentos).  
3. Poder preditivo limitado (F1 ≈ 0,26 na validação).  
4. Conjunto de teste já visto em avaliações anteriores da Regressão Logística.  
5. Licença dos dados: **CC BY-NC 4.0** (não comercial).

---

## Licença dos dados

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — atribuição obrigatória; uso comercial **não** permitido.  
Fonte: [mindweave/food-delivery-orders](https://huggingface.co/datasets/mindweave/food-delivery-orders)

---

**Repositório:** https://github.com/rodrigocataoc/order-cancellation-prediction  
**Autor:** Rodrigo Catão
