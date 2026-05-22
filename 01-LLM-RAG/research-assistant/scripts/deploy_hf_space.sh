#!/usr/bin/env bash
# Deploy this app to a Hugging Face Space (Docker SDK).
#
# Usage: bash scripts/deploy_hf_space.sh <hf-username> <space-name>
# Prereqs:
#   1. Install the HF CLI:  pip install -U huggingface_hub
#   2. Log in:              huggingface-cli login   (paste a write token)
#   3. Ensure your .env has the API keys you need at runtime
#
# What this script does:
#   1. Creates the Space (Docker SDK) if it doesn't exist
#   2. Swaps in the SDK metadata header on README.md
#   3. Pushes the current directory to the Space's git remote
#   4. Uploads each secret from .env to the Space

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

echo "==> Creating Space ${REPO_ID} (no-op if it exists)..."
huggingface-cli repo create "${SPACE_NAME}" \
    --type space \
    --space_sdk docker \
    -y || true

echo "==> Swapping in Space-formatted README..."
cp README.md README.github.md
cp .huggingface/SPACE_README.md README.md

echo "==> Uploading the project to the Space..."
huggingface-cli upload "${REPO_ID}" . . \
    --repo-type space \
    --commit-message "Deploy from local"

echo "==> Restoring the GitHub README..."
mv README.github.md README.md

if [[ -f .env ]]; then
    echo "==> Uploading secrets from .env..."
    while IFS='=' read -r key value; do
        [[ -z "${key}" || "${key}" =~ ^# ]] && continue
        # Strip surrounding quotes/whitespace from value.
        value="${value#\"}"
        value="${value%\"}"
        value="${value#\'}"
        value="${value%\'}"
        if [[ -n "${value}" ]]; then
            echo "    - ${key}"
            huggingface-cli space secrets put "${REPO_ID}" "${key}" "${value}" || true
        fi
    done < .env
fi

echo
echo "Done. Your Space: https://huggingface.co/spaces/${REPO_ID}"
