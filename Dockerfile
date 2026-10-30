FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY models/hydraulic/comparisons/hydraulic-compare-20261021-2104d/random_forest.joblib \
     ./models/hydraulic/comparisons/hydraulic-compare-20261021-2104d/random_forest.joblib

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "src.hydraulic.api:app", "--host", "0.0.0.0", "--port", "8000"]
