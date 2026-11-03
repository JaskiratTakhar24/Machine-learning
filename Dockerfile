FROM python:3.14-slim

WORKDIR /app

COPY requirements-api.txt .
RUN python -m pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
RUN python src/ml_learning/train.py \
    && useradd --create-home appuser

ENV PYTHONPATH=/app/src
EXPOSE 8000

USER appuser

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

CMD ["uvicorn", "ml_learning.api:app", "--host", "0.0.0.0", "--port", "8000"]
