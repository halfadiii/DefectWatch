# DefectWatch API: FastAPI + ONNX Runtime on CPU. No PyTorch in the image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Packages first, so a code-only change reuses this layer.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY serving/__init__.py serving/app.py serving/inference.py serving/preprocess.py serving/
COPY models/ models/

# Do not run as root.
RUN useradd --create-home --uid 10001 appuser
USER appuser

ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION} \
    MODELS_DIR=/app/models \
    PORT=8080

EXPOSE 8080

# One worker: Cloud Run scales by adding containers, and each worker would
# hold its own copy of the models in memory.
CMD ["sh", "-c", "exec uvicorn serving.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
