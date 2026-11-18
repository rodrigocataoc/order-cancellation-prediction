# Order Cancellation Prediction

> Portfólio de Machine Learning: prever se um pedido de delivery será **entregue** ou **cancelado**.

**Autor:** Rodrigo Catão  
**Foco:** estágio / primeiro emprego em IA, ML e Ciência de Dados  
**Status:** validado, notebook reproduzível, modelo exportado para inferência

---

## Problema de negócio

Cancelamentos geram custo e ineficiência. O modelo usa apenas informações disponíveis **no momento do pedido** para sinalizar risco de cancelamento.

| Classe | Significado |
|--------|-------------|
| `0` | Entregue |
| `1` | Cancelado |

Com ~91% de entregas, **acurácia isolada é enganosa**: um classificador que sempre prevê “entregue” já acerta 91,5%.

---

## Dataset

- **Fonte:** [mindweave/food-delivery-orders](https://huggingface.co/datasets/mindweave/food-delivery-orders)
- **Tipo:** sintético (não representa operação real)
- **Amostra:** 2.000 pedidos · 1.828 entregues (91,4%) · 172 cancelados (8,6%)
- **Licença:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (atribuição; uso comercial não permitido)
- O CSV **não** está no repositório — veja `data/README.md`

![Distribuição das classes](images/class_distribution.png)

---

## Features (sem data leakage)

| Feature | Descrição |
|---------|-----------|
| `item_count` | Qtd. de itens |
| `subtotal` | Valor dos produtos |
| `delivery_fee` | Taxa de entrega |
| `estimated_delivery_min` | Tempo estimado |
| `order_hour` | Hora do pedido |
| `order_weekday` | Dia da semana (0–6) |

**Excluídas de propósito:** `actual_delivery_min`, `cancellation_reason`, `rating` (pós-evento), IDs, `tip`, `total`.

---

## Modelos e validação

- Split estratificado: ~1.200 treino / 400 validação / 400 teste (`random_state=42`)
- Threshold que maximiza **F1** escolhido **somente na validação**
- Pipelines scikit-learn (scaler só no treino, quando aplicável)

### Baseline (`DummyClassifier`, most_frequent)

- Accuracy no teste: **0,915** (não identifica cancelamentos)

### Validação — threshold 0,50

| Métrica | Regressão Logística | Random Forest |
|---------|--------------------:|--------------:|
| Accuracy | 0,4950 | 0,6325 |
| Precision | 0,0927 | 0,1111 |
| Recall | 0,5429 | 0,4571 |
| F1-score | 0,1583 | 0,1788 |
| PR-AUC | 0,1240 | 0,1668 |
| ROC-AUC | 0,5354 | 0,5220 |

### Validação — threshold que maximiza F1

| Modelo | Threshold | Precision | Recall | F1 |
|--------|----------:|----------:|-------:|---:|
| Regressão Logística | ≈ 0,533 | 0,113 | 0,486 | 0,184 |
| **Random Forest** | ≈ 0,600 | **0,308** | 0,229 | **0,262** |

**Matriz LR (validação, th≈0,533):** `[[232, 133], [18, 17]]`  
**Matriz RF (validação, th≈0,600):** `[[347, 18], [27, 8]]`

![Matriz LR](images/confusion_matrix_logistic_regression.png)

![Matriz RF](images/confusion_matrix_random_forest.png)

![Curva PR](images/precision_recall_curve.png)

### Importância das features (RF)

| Feature | Importância |
|---------|------------:|
| subtotal | ≈ 0,296 |
| estimated_delivery_min | ≈ 0,237 |
| order_weekday | ≈ 0,226 |
| order_hour | ≈ 0,187 |
| item_count | ≈ 0,054 |
| delivery_fee | 0,000 |

![Feature importance](images/feature_importance.png)

### Interpretação

- **LR:** maior recall → mais cancelamentos capturados, muitos falsos positivos.
- **RF:** maior precision e F1 na validação → menos alertas incorretos, perde mais cancelamentos.
- Desempenho **modesto**; adequado para aprendizado e portfólio, não para produção com estes dados.

### Nota sobre o conjunto de teste

O teste **já foi usado** em avaliações anteriores da Regressão Logística. Não tratá-lo como hold-out intocado. Métricas de teste no notebook existem só para referência, com essa limitação explícita.

---

## Como reproduzir

```bash
git clone https://github.com/rodrigocataoc/order-cancellation-prediction.git
cd order-cancellation-prediction

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Baixe data/orders.csv (ver data/README.md)

jupyter notebook notebooks/01_data_exploration.ipynb
# Execute todas as células — gráficos em images/ e modelo em models/
```

---

## Inferência com o modelo salvo

Após executar o notebook:

```python
import joblib, json
import pandas as pd

pipe = joblib.load("models/cancellation_rf_pipeline.joblib")
meta = json.loads(open("models/model_metadata.json").read())
th = meta["threshold"]

row = pd.DataFrame([{
    "item_count": 3,
    "subtotal": 62.5,
    "delivery_fee": 4.99,
    "estimated_delivery_min": 40,
    "order_hour": 20,
    "order_weekday": 5,
}])
proba = pipe.predict_proba(row)[0, 1]
pred = int(proba >= th)
print(proba, pred)  # probabilidade e 0/1
```

Helper: `src/predict.py` (`predict_cancellation`).

Modelo exportado: **Random Forest** (melhor F1 na validação), treinado em treino+validação, threshold da validação.

---

## Estrutura

```text
order-cancellation-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── 01_data_exploration.ipynb
├── models/
│   ├── cancellation_rf_pipeline.joblib   # gerado ao rodar o notebook
│   └── model_metadata.json
├── src/
│   └── predict.py
└── images/
    ├── class_distribution.png
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix_random_forest.png
    ├── precision_recall_curve.png
    └── feature_importance.png
```

---

## Limitações

1. Dados sintéticos  
2. Poucos eventos da classe positiva (172)  
3. Poder preditivo limitado  
4. Teste já inspecionado historicamente  
5. Features restritas ao instante do pedido  

---

## Tecnologias

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · joblib · Jupyter

---

## Licença dos dados

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — atribuição obrigatória; uso comercial não permitido.

---

**Repositório:** https://github.com/rodrigocataoc/order-cancellation-prediction  
**Autor:** Rodrigo Catão
