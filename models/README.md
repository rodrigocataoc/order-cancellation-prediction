# Modelos

| Arquivo | Descrição |
|---------|-----------|
| `cancellation_rf_pipeline.joblib` | Pipeline Random Forest (treino+validação) |
| `model_metadata.json` | Features, threshold ≈ 0.5997, métricas de validação |

Gerar de novo (com `data/orders.csv`):

```bash
python scripts/train_and_export.py
```
