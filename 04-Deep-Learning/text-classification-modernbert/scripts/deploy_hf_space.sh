#!/usr/bin/env bash
# Deploy the Gradio demo to a Hugging Face Space (Docker SDK).
#
# Usage: bash scripts/deploy_hf_space.sh <hf-username> <space-name>
# Prereqs:
#   1. pip install -U huggingface_hub
#   2. huggingface-cli login
#   3. .env has HF_MODEL_REPO=<username>/modernbert-banking77

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <hf-username> <space-name>" >&2
    exit 1
fi

HF_USER="$1"
SPACE_NAME="$2"
REPO_ID="${HF_USER}/${SPACE_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

cd "${PROJECT_DIR}"

echo "==> Creating Space ${REPO_ID}..."
huggingface-cli repo create "${SPACE_NAME}" --type space --space_sdk docker -y || true

echo "==> Swapping in Space-formatted README..."
cp README.md README.github.md
cp .huggingface/SPACE_README.md README.md

echo "==> Uploading..."
huggingface-cli upload "${REPO_ID}" . . --repo-type space --commit-message "Deploy"

mv README.github.md README.md

if [[ -f .env ]]; then
    echo "==> Uploading secrets from .env..."
    while IFS='=' read -r key value; do
        [[ -z "${key}" || "${key}" =~ ^# ]] && continue
        value="${value#\"}"; value="${value%\"}"
        value="${value#\'}"; value="${value%\'}"
        if [[ -n "${value}" ]]; then
            echo "    - ${key}"
            huggingface-cli space secrets put "${REPO_ID}" "${key}" "${value}" || true
        fi
    done < .env
fi

echo
echo "Done. https://huggingface.co/spaces/${REPO_ID}"
