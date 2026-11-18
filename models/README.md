# Modelos

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `cancellation_rf_pipeline.joblib` | Pipeline Random Forest treinado (treino+validação) |
| `model_metadata.json` | Features, threshold e métricas de validação |

## Como gerar

Execute o notebook `notebooks/01_data_exploration.ipynb` até o final. As células finais salvam o `.joblib` e atualizam o metadata.

O threshold de decisão (~0,60) foi escolhido **apenas na validação** para maximizar o F1.

## Inferência

Veja `src/predict.py` ou o exemplo no README principal.
