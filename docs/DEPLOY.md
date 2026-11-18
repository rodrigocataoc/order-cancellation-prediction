# Deploy

## Local

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Confirme que o modelo existe
ls models/cancellation_rf_pipeline.joblib

# Se não existir:
# 1) baixe data/orders.csv (ver data/README.md)
# 2) python scripts/train_and_export.py

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Abra: http://127.0.0.1:8000

## Docker

```bash
docker build -t order-cancellation-api .
docker run --rm -p 8000:8000 order-cancellation-api
```

## Render (gratuito)

1. Crie conta em [render.com](https://render.com)
2. **New → Web Service** → conecte este repositório
3. Configurações sugeridas:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Alternativa: use o arquivo `render.yaml` (Blueprint)
5. Deploy

O modelo `.joblib` precisa estar no repositório (já previsto no `.gitignore` com exceção explícita).

## Variáveis de ambiente
Nenhuma obrigatória além de `PORT` (fornecido automaticamente pelo Render).

## Observações
- Plano free do Render pode “dormir” após inatividade.
- Não publique o dataset (licença CC BY-NC 4.0 + política do projeto).
- Esta API é demonstração educacional; o desempenho preditivo é modesto.
