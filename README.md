# Order Cancellation Prediction

> Projeto de portfólio em Machine Learning: previsão de cancelamento de pedidos de delivery com dados sintéticos.

**Autor:** Rodrigo Catão  
**Objetivo de carreira:** estágio / primeiro emprego em Inteligência Artificial, Machine Learning e Ciência de Dados  
**Status:** experimentos documentados, notebook reproduzível e repositório organizado

---

## 1. Título e descrição

Este projeto desenvolve e avalia modelos de **classificação binária** para prever se um pedido de delivery será **entregue** ou **cancelado**, utilizando apenas informações que estariam disponíveis no momento do pedido.

O foco está em:

- Prevenção de **data leakage**
- Tratamento consciente do **desbalanceamento de classes**
- Escolha e interpretação de **métricas adequadas** (Precision, Recall, F1, PR-AUC, matriz de confusão)
- Documentação técnica clara e reproduzível

---

## 2. Problema de negócio

Cancelamentos geram custo operacional, frustração do cliente e ineficiência na alocação de entregadores e restaurantes. Identificar antecipadamente pedidos com maior risco de cancelamento pode apoiar ações preventivas (confirmação, priorização, comunicação).

O modelo deve sinalizar o risco **antes** do desfecho do pedido.

---

## 3. Objetivo do modelo

Classificação binária:

| Classe | Significado   |
|--------|---------------|
| `0`    | Entregue      |
| `1`    | Cancelado     |

O sucesso não é medido só por acurácia. Em um cenário com ~91% de entregas, um modelo que sempre prevê “entregue” já atinge alta acurácia sem identificar nenhum cancelamento.

---

## 4. Origem e características do dataset

**Fonte:** [mindweave/food-delivery-orders](https://huggingface.co/datasets/mindweave/food-delivery-orders) (Hugging Face)

| Característica              | Valor                          |
|----------------------------|--------------------------------|
| Tipo                       | Sintético                      |
| Registros utilizados       | 2.000                          |
| Colunas originais          | 16                             |
| Pedidos entregues          | 1.828 (91,4%)                  |
| Pedidos cancelados         | 172 (8,6%)                     |
| Licença                    | CC BY-NC 4.0 (atribuição + não comercial) |

**Importante:** os dados **não** representam operação real de nenhuma plataforma (incluindo iFood).

O arquivo de dados **não** está no repositório. Veja `data/README.md` para instruções de download.

---

## 5. Análise exploratória (resumo)

- Desbalanceamento acentuado da variável-alvo.
- Valores ausentes concentrados em colunas pós-evento (`actual_delivery_min`, `cancellation_reason`, `rating`), coerentes com o desfecho do pedido.
- Sem registros duplicados na amostra utilizada.
- Análise temporal exploratória: horários de pico (ex.: final de tarde / noite) apresentam taxas de cancelamento relativamente mais altas; isso é descritivo, não causal.

Gráfico da distribuição das classes:

![Distribuição das classes](images/class_distribution.png)

---

## 6. Tratamento e preparação dos dados

1. Criação do alvo: `cancelled = 1` se `status == 'cancelled'`, senão `0`.
2. Extração de `order_hour` e `order_weekday` a partir de `order_timestamp`.
3. Seleção de features disponíveis no momento do pedido.
4. Divisão estratificada:
   - 80% treino+validação / 20% teste
   - Do treino+validação: 75% treino / 25% validação  
   → aproximadamente 1.200 treino / 400 validação / 400 teste
5. Pré-processamento dentro de **pipelines** do scikit-learn (ex.: `StandardScaler` apenas no treino, aplicado depois na validação/teste) para evitar vazamento.

---

## 7. Features utilizadas

| Feature                   | Descrição                              |
|---------------------------|----------------------------------------|
| `item_count`              | Quantidade de itens                    |
| `subtotal`                | Valor dos produtos                     |
| `delivery_fee`            | Taxa de entrega                        |
| `estimated_delivery_min`  | Tempo estimado de entrega              |
| `order_hour`              | Hora do pedido                         |
| `order_weekday`           | Dia da semana (0=segunda … 6=domingo)  |

---

## 8. Justificativa para exclusão de variáveis

| Variável                 | Motivo da exclusão                                      |
|--------------------------|---------------------------------------------------------|
| `actual_delivery_min`    | Só conhecida após a entrega                             |
| `cancellation_reason`    | Só conhecida após o cancelamento                        |
| `rating`                 | Avaliação posterior ao pedido                           |
| `id`, `customer_id`, `restaurant_id`, `driver_id` | Identificadores; disponibilidade no momento da previsão é questionável |
| `tip`, `total`           | Podem não estar fechados ou ser redundantes no instante da previsão |

Usar qualquer uma dessas colunas como feature introduziria **data leakage** ou informação indisponível na prática.

---

## 9. Modelos testados

### Baseline
- `DummyClassifier(strategy='most_frequent')`
- Sempre prevê a classe majoritária (entregue)
- Acurácia no teste ≈ **0,915** — referência, não solução

### Regressão Logística
- Pipeline: `StandardScaler` + `LogisticRegression`
- `class_weight='balanced'`, `random_state=42`, `max_iter=1000`

### Random Forest
- `n_estimators=300`, `max_depth=5`
- `min_samples_split=10`, `min_samples_leaf=5`
- `class_weight='balanced'`, `random_state=42`, `n_jobs=-1`

---

## 10. Métricas e resultados (reexecutados)

Os números abaixo foram **reexecutados** no notebook deste repositório, com a mesma divisão estratificada (`random_state=42`) e os mesmos hiperparâmetros.  
Thresholds ótimos para F1 foram escolhidos **apenas na validação**.

### Validação — threshold 0,50

| Métrica    | Regressão Logística | Random Forest |
|------------|--------------------:|--------------:|
| Accuracy   | 0,4950              | 0,6325        |
| Precision  | 0,0927              | 0,1111        |
| Recall     | 0,5429              | 0,4571        |
| F1-score   | 0,1583              | 0,1788        |
| PR-AUC     | 0,1240              | 0,1668        |
| ROC-AUC    | 0,5354              | 0,5220        |

### Validação — threshold que maximiza F1

| Modelo               | Threshold | Precision | Recall | F1-score |
|----------------------|----------:|----------:|-------:|---------:|
| Regressão Logística  | ≈ 0,5332  | 0,1133    | 0,4857 | 0,1838   |
| Random Forest        | ≈ 0,5997  | 0,3077    | 0,2286 | 0,2623   |

**Matriz de confusão — Regressão Logística (th ≈ 0,533)**

```text
[[232, 133],
 [ 18,  17]]
```

**Matriz de confusão — Random Forest (th ≈ 0,600)**

```text
[[347,  18],
 [ 27,   8]]
```

![Matriz de confusão — Regressão Logística](images/confusion_matrix_logistic_regression.png)

![Matriz de confusão — Random Forest](images/confusion_matrix_random_forest.png)

### Importância das features (Random Forest)

| Feature                  | Importância |
|--------------------------|------------:|
| `subtotal`               | ≈ 0,296     |
| `estimated_delivery_min` | ≈ 0,237     |
| `order_weekday`          | ≈ 0,226     |
| `order_hour`             | ≈ 0,187     |
| `item_count`             | ≈ 0,054     |
| `delivery_fee`           | 0,000       |

![Importância das features](images/feature_importance.png)

![Curva Precision-Recall](images/precision_recall_curve.png)

---

## 11. Comparação entre modelos

- **Regressão Logística** (com threshold ajustado): maior **recall** → identifica mais cancelamentos, porém com muitos falsos positivos.
- **Random Forest** (com threshold ajustado): maior **precision** e **F1** → menos alertas incorretos, porém deixa passar mais cancelamentos.
- Nenhum modelo apresenta desempenho forte o suficiente para uso operacional com as features e o volume atuais.
- A escolha depende do **custo de negócio** de falso positivo vs. falso negativo.

---

## 12. Discussão sobre desbalanceamento

Com 8,6% de cancelamentos:

- Acurácia alta é fácil de obter e pouco informativa.
- Precision tende a ser baixa quando o modelo tenta aumentar o recall.
- PR-AUC é mais adequada que ROC-AUC em cenários muito desbalanceados (embora ambas sejam reportadas).
- `class_weight='balanced'` ajuda o otimizador a não ignorar a classe minoritária, mas não resolve sozinho a dificuldade do problema.

---

## 13. Limitações do projeto

1. Dados sintéticos — generalização para o mundo real não é garantida.
2. Amostra pequena e poucos eventos da classe positiva.
3. Capacidade preditiva ainda modesta.
4. Features restritas ao que está disponível no momento do pedido.
5. O conjunto de teste já foi utilizado em avaliações anteriores da Regressão Logística; não deve ser tratado como “intocado” em relatos de performance final. Para uma avaliação independente seria necessário novo protocolo ou dados externos.
6. Resultados de threshold e matrizes de confusão de execuções antigas continham pequenas divergências; os números deste README e do notebook foram **reconfirmados** na execução atual.

---

## 14. Como reproduzir o experimento

```bash
# 1. Clone o repositório
git clone https://github.com/rodrigocataoc/order-cancellation-prediction.git
cd order-cancellation-prediction

# 2. Ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Dependências
pip install -r requirements.txt

# 4. Baixe o dataset
#    - Acesse https://huggingface.co/datasets/mindweave/food-delivery-orders
#    - Salve data/orders.csv (ver data/README.md)

# 5. Abra e execute o notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

Execute as células em ordem. Os gráficos e métricas serão gerados localmente.

---

## 15. Tecnologias utilizadas

- Python 3
- pandas, NumPy
- scikit-learn (Pipeline, LogisticRegression, RandomForestClassifier, métricas)
- matplotlib, seaborn
- Jupyter Notebook

---

## 16. Estrutura do repositório

```text
order-cancellation-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md          # instruções de download (CSV não versionado)
├── notebooks/
│   └── 01_data_exploration.ipynb
└── images/
    ├── class_distribution.png
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix_random_forest.png
    ├── precision_recall_curve.png
    └── feature_importance.png
```

---

## 17. Fonte e licença dos dados

- **Fonte:** [mindweave/food-delivery-orders](https://huggingface.co/datasets/mindweave/food-delivery-orders)
- **Licença dos dados:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — atribuição obrigatória; uso comercial **não** permitido.
- O código e a documentação deste repositório são de autoria do projeto de portfólio; reutilize com atribuição quando fizer sentido.

---

## Aprendizados e honestidade técnica

Este projeto prioriza:

- Clareza sobre o que o modelo **consegue e não consegue** fazer
- Documentação de limitações e de decisões (features, leakage, métricas)
- Reprodução e verificação de resultados em vez de números “bonitos”

Resultados modestos, bem explicados e tecnicamente corretos, são preferíveis a performance inflada ou documentação incompleta.

---

**Repositório:** https://github.com/rodrigocataoc/order-cancellation-prediction  
**Autor:** Rodrigo Catão
