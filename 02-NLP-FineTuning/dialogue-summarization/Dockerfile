# syntax=docker/dockerfile:1.7
# Inference-only Docker image for the Gradio demo. Training is intended to
# happen in Colab/Kaggle (free GPU) — see notebooks/01_finetune_samsum_lora.ipynb.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
