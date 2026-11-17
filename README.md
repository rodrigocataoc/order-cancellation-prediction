# order-cancellation-prediction

# Order Cancellation Prediction

Projeto de Machine Learning para análise e previsão de cancelamentos de pedidos em uma plataforma de delivery.

O objetivo é investigar se informações disponíveis no momento do pedido podem ajudar a identificar pedidos com maior risco de cancelamento, utilizando técnicas de classificação supervisionada.

> **Status:** em desenvolvimento — exploração dos dados, preparação, treinamento inicial, comparação de modelos e análise de thresholds realizados. Avaliação final no conjunto de teste ainda pendente.

## 1. Objetivo do projeto

O cancelamento de pedidos é um desafio operacional para plataformas de delivery. Identificar antecipadamente pedidos com maior risco de cancelamento pode ajudar a direcionar ações preventivas e melhorar a eficiência da operação.

Neste projeto, desenvolvo modelos de classificação binária para prever se um pedido será:

- `0`: Entregue.
- `1`: Cancelado.

O foco não é apenas obter uma boa acurácia, mas investigar a capacidade dos modelos de identificar cancelamentos, considerando o desbalanceamento das classes e os custos de falsos positivos e falsos negativos.

## 2. Dataset

**Fonte:** [Food Delivery Orders — Hugging Face](https://huggingface.co/datasets/mindweave/food-delivery-orders)

O dataset disponibiliza informações de pedidos de uma operação fictícia de delivery.

Características da amostra utilizada:

- 2.000 pedidos.
- 500 clientes.
- 10 entregadores.
- 20 restaurantes.
- 16 colunas originais.

**Importante:** os dados são sintéticos. Não representam dados reais do iFood ou de qualquer outra plataforma de delivery.

**Licença da fonte:** CC BY-NC 4.0. Consulte os termos da licença antes de reutilizar ou redistribuir os dados, especialmente em contextos comerciais. Atribuição à fonte original deve ser mantida.

## 3. Análise exploratória dos dados (EDA)

A primeira etapa foi compreender a estrutura, a distribuição das classes, os valores ausentes e as características dos pedidos.

### Distribuição da variável-alvo

| Status | Quantidade | Proporção |
|---|---:|---:|
| Entregue | 1.828 | 91,4% |
| Cancelado | 172 | 8,6% |
| **Total** | **2.000** | **100%** |

Foi identificado um desbalanceamento significativo entre as classes. Apenas 8,6% dos pedidos da amostra foram cancelados.

Esse cenário exige cuidado na escolha das métricas, pois um modelo pode apresentar alta acurácia simplesmente prevendo a classe majoritária.

### Valores ausentes

Foram identificados valores ausentes nas seguintes colunas:

| Coluna | Valores ausentes |
|---|---:|
| `actual_delivery_min` | 172 |
| `cancellation_reason` | 1.828 |
| `rating` | 172 |

Esses valores estão relacionados ao resultado do pedido: pedidos cancelados não possuem tempo real de entrega ou avaliação, enquanto pedidos entregues não possuem motivo de cancelamento.

Não foram encontrados registros duplicados.

### Motivos de cancelamento

| Motivo | Quantidade |
|---|---:|
| Customer changed mind | 31 |
| Customer unreachable at delivery | 24 |
| Weather conditions | 24 |
| Item unavailable | 22 |
| Excessive wait time | 22 |
| Restaurant too busy | 21 |
| No drivers available | 18 |
| Restaurant closed unexpectedly | 10 |

### Análise temporal

Na análise por horário, 20h apresentou a maior taxa de cancelamento, aproximadamente 12,44%, com 28 cancelamentos em 225 pedidos.

O horário das 18h apresentou o maior número absoluto de cancelamentos: 30.

Essas observações são exploratórias e não demonstram, por si só, uma relação causal.

## 4. Preparação dos dados e prevenção de Data Leakage

A variável-alvo foi construída a partir da coluna `status`:

- `cancelled` → 1
- `delivered` → 0

Foram selecionadas inicialmente seis variáveis preditoras:

| Variável | Descrição |
|---|---|
| `item_count` | Quantidade de itens do pedido |
| `subtotal` | Valor dos produtos |
| `delivery_fee` | Taxa de entrega |
| `estimated_delivery_min` | Tempo estimado de entrega |
| `order_hour` | Hora em que o pedido foi realizado |
| `order_weekday` | Dia da semana do pedido |

As variáveis temporais foram extraídas de `order_timestamp`.

### Variáveis excluídas

Algumas colunas foram removidas para evitar vazamento de informação (*data leakage*) ou por não serem adequadas ao cenário inicial de previsão:

- `actual_delivery_min`: informação conhecida após o evento de entrega.
- `cancellation_reason`: motivo conhecido após o cancelamento.
- `rating`: avaliação posterior ao pedido.
- `id`, `customer_id`, `restaurant_id` e `driver_id`: identificadores inicialmente excluídos.
- `tip` e `total`: excluídos inicialmente por questões de disponibilidade no momento da previsão e possível redundância com outras informações.

A exclusão de `driver_id` também considera que essa informação pode não estar disponível no momento em que o pedido é realizado.

A prevenção de data leakage é importante para que o modelo utilize somente informações que poderiam estar disponíveis no momento real da previsão.

## 5. Estratégia de validação

Foi utilizado `train_test_split` com divisão estratificada para preservar a proporção das classes.

Divisão inicial:

- Treinamento: 80% — 1.600 registros.
- Teste: 20% — 400 registros.

A divisão inicial utilizou `random_state=42` e `stratify=y`.

Posteriormente, o conjunto de treinamento foi dividido novamente:

- Subconjunto de treinamento: 1.200 registros.
- Validação: 400 registros.

A validação também foi estratificada, com `random_state=42`.

Distribuição da validação:

| Classe | Quantidade |
|---|---:|
| Entregue | 365 |
| Cancelado | 35 |
| **Total** | **400** |

O conjunto de validação foi utilizado para comparar modelos e selecionar thresholds. O conjunto de teste foi reservado para a avaliação final independente.

## 6. Modelos desenvolvidos

### 6.1 Baseline

Foi utilizado um `DummyClassifier` com estratégia `most_frequent`.

Esse modelo sempre prevê a classe majoritária, servindo como referência simples para avaliar a utilidade dos modelos treinados.

Na avaliação preliminar realizada no conjunto de teste, apresentou acurácia de 91,5%.

Esse resultado demonstra por que a acurácia isolada não é uma métrica adequada para avaliar o problema: um classificador pode acertar a maioria dos pedidos entregues sem identificar os cancelamentos.

### 6.2 Regressão Logística

Foi implementado um pipeline contendo:

- `StandardScaler` para padronização das variáveis.
- `LogisticRegression` para classificação.
- `class_weight="balanced"` para lidar com o desbalanceamento das classes.
- `max_iter=1000`.
- `random_state=42`.

O modelo foi treinado inicialmente no subconjunto de treinamento da validação.

### 6.3 Random Forest

Também foi desenvolvido um modelo Random Forest com os seguintes parâmetros:

- `n_estimators=300`
- `max_depth=5`
- `min_samples_split=10`
- `min_samples_leaf=5`
- `class_weight="balanced"`
- `random_state=42`

O objetivo foi testar um algoritmo baseado em árvores de decisão e comparar seu comportamento com a Regressão Logística.

## 7. Métricas de avaliação

Devido ao desbalanceamento das classes, foram consideradas as seguintes métricas:

- **Accuracy:** proporção geral de previsões corretas.
- **Precision:** proporção de pedidos sinalizados como cancelados que realmente foram cancelados.
- **Recall:** proporção dos cancelamentos reais identificados pelo modelo.
- **F1-score:** equilíbrio entre precision e recall.
- **PR-AUC:** área sob a curva Precision-Recall, especialmente útil em problemas com classes desbalanceadas.
- **ROC-AUC:** capacidade de discriminação entre as classes.

A análise também considera a matriz de confusão para identificar falsos positivos e falsos negativos.

## 8. Resultados experimentais

### Comparação na validação — threshold padrão

Os resultados abaixo foram obtidos na mesma validação de 400 registros, utilizando o threshold padrão de 0,50.

| Métrica | Regressão Logística | Random Forest |
|---|---:|---:|
| Accuracy | 0,4950 | 0,7625 |
| Precision | 0,0927 | 0,1250 |
| Recall | 0,5429 | 0,2857 |
| F1-score | 0,1583 | 0,1739 |
| PR-AUC | 0,1240 | 0,1700 |
| ROC-AUC | 0,5354 | 0,5209 |

O Random Forest apresentou maior acurácia, precision, F1-score e PR-AUC.

A Regressão Logística apresentou maior recall e ROC-AUC.

Os resultados mostram que os modelos possuem comportamentos diferentes e que nenhum deles apresenta, até o momento, desempenho suficientemente forte para uma aplicação real.

### 8.1 Ajuste de threshold

Foi realizada uma análise da curva Precision-Recall e uma busca pelo threshold que maximiza o F1-score na validação.

Resultados registrados durante essa etapa:

| Modelo | Threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Regressão Logística | 0,5332 | 0,1133 | 0,4857 | 0,1838 |
| Random Forest | 0,5536 | 0,3077 | 0,2286 | 0,2623 |

O ajuste do threshold permite modificar o equilíbrio entre precision e recall sem alterar as probabilidades produzidas pelo modelo.

A Regressão Logística apresentou maior recall, enquanto o Random Forest apresentou maior precision e F1-score nessa tabela experimental.

**Nota de verificação:** a matriz de confusão posteriormente registrada para o Random Forest foi `[[347, 18], [28, 7]]`. Ela corresponde a precision de 0,28, recall de 0,20 e F1-score de aproximadamente 0,2333, diferentes dos valores da tabela de thresholds. Essa divergência precisa ser conferida no notebook antes de consolidar os resultados finais.

### 8.2 Matrizes de confusão após ajuste de threshold

**Regressão Logística**

```text
[[232 133]
 [ 18  17]]
```

- Verdadeiros negativos: 232.
- Falsos positivos: 133.
- Falsos negativos: 18.
- Verdadeiros positivos: 17.

**Random Forest**

```text
[[347  18]
 [ 28   7]]
```

- Verdadeiros negativos: 347.
- Falsos positivos: 18.
- Falsos negativos: 28.
- Verdadeiros positivos: 7.

Esses resultados evidenciam uma diferença operacional importante:

- A Regressão Logística identifica mais cancelamentos, mas gera mais alertas incorretos.
- O Random Forest gera menos alertas incorretos, mas deixa passar mais cancelamentos.

A escolha entre os modelos depende do custo operacional associado a cada tipo de erro.

## 9. Importância das variáveis — Random Forest

A análise inicial de importância das features apresentou os seguintes resultados:

| Variável | Importância |
|---|---:|
| `subtotal` | 0,298413 |
| `estimated_delivery_min` | 0,238492 |
| `order_weekday` | 0,206237 |
| `order_hour` | 0,192754 |
| `item_count` | 0,064104 |
| `delivery_fee` | 0,000000 |

As variáveis `subtotal`, `estimated_delivery_min`, `order_weekday` e `order_hour` concentraram a maior parte da importância calculada pelo modelo.

Essa importância representa a contribuição das variáveis nas divisões das árvores e não deve ser interpretada como causalidade.

## 10. Limitações atuais

O projeto possui limitações que precisam ser consideradas:

1. **Dados sintéticos:** o dataset não representa o comportamento real de uma plataforma de delivery.
2. **Tamanho da amostra:** foram utilizados 2.000 registros, com apenas 172 cancelamentos.
3. **Desbalanceamento:** a classe de cancelamentos representa somente 8,6% dos dados.
4. **Capacidade preditiva:** os resultados atuais ainda são modestos.
5. **Generalização:** os resultados precisam ser avaliados em dados independentes e, idealmente, em dados reais.
6. **Variáveis disponíveis:** algumas informações podem não estar disponíveis no momento em que a previsão precisa ser realizada.

Os resultados apresentados são experimentais e não demonstram capacidade de previsão em uma operação real.

## 11. Próximas etapas

- [ ] Conferir a divergência entre os resultados de threshold e a matriz de confusão do Random Forest.
- [ ] Realizar a avaliação final no conjunto de teste reservado.
- [ ] Comparar os resultados finais dos modelos.
- [ ] Investigar outras estratégias para lidar com o desbalanceamento.
- [ ] Avaliar a calibração das probabilidades.
- [ ] Investigar outras variáveis e possibilidades de engenharia de atributos.
- [ ] Documentar os resultados finais e as limitações.
- [ ] Organizar o notebook e os arquivos para reprodução do experimento.

## 12. Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab
- Hugging Face Datasets

## 13. Estrutura do projeto

Estrutura sugerida para o repositório:

```text
order-cancellation-prediction/
│
├── README.md
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── data/
│   └── README.md
│
├── images/
│   ├── class_distribution.png
│   ├── precision_recall_curve.png
│   └── feature_importance.png
│
└── requirements.txt
```

Os dados não precisam ser incluídos no repositório. O arquivo `data/README.md` pode explicar como obter a amostra diretamente da fonte original.

## 14. Objetivo de aprendizado

Este projeto faz parte do meu desenvolvimento em Machine Learning e Engenharia de Inteligência Artificial.

O trabalho busca desenvolver habilidades práticas em:

- Análise exploratória de dados.
- Preparação de dados para Machine Learning.
- Prevenção de data leakage.
- Classificação supervisionada.
- Tratamento de classes desbalanceadas.
- Avaliação e comparação de modelos.
- Ajuste de threshold.
- Interpretação de métricas e resultados.
- Análise de problemas sob a perspectiva de negócio.

---

**Autor:** Rodrigo Catão

**Repositório:** `order-cancellation-prediction`

**Fonte dos dados:** [mindweave/food-delivery-orders — Hugging Face](https://huggingface.co/datasets/mindweave/food-delivery-orders)
