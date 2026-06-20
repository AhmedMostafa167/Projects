# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf_cache \
    TRANSFORMERS_CACHE=/app/.hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.hf_cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch FIRST, with --index-url (not --extra-index-url) so
# there is no ambiguity about which wheel gets picked. --extra-index-url lets
# pip/uv still fall back to PyPI's default (CUDA) wheel for transitive deps,
# which is what pulled in the ~6-8GB of nvidia/cu13 .so files previously.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.4.1"

# constraints.txt pins torch to the exact CPU build just installed, so the
# next step can never "upgrade" it back to a CUDA wheel even if a package's
# floor requirement tightens later.
COPY constraints.txt .
COPY requirements.txt .
RUN pip install -c constraints.txt -r requirements.txt

COPY . .

# HF Spaces requires the app to listen on 7860 by default.
EXPOSE 7860

CMD ["python", "app.py"]