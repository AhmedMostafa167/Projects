#!/usr/bin/env bash
# Deploy this app to a Hugging Face Space using the `hf` CLI and git.
#
# Usage: bash scripts/deploy_hf_space.sh <hf-username> <space-name>
# Prereqs:
#   1. Install the HF CLI:  pip install -U huggingface_hub  (provides `hf`)
#   2. Log in:              hf auth login   (paste a write token)
#   3. Ensure your .env has the API keys you need at runtime
#
# What this script does:
#   1. Creates the Space if it doesn't exist
#   2. Swaps in the Space-formatted README
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
# Create the repo (space). `hf repo create` supports creating spaces.
# If the command fails because the repo already exists, ignore the error.
hf repo create "${REPO_ID}" \
    --type space \
    --space_sdk docker \
    -y || true

echo "==> Swapping in Space-formatted README..."
cp README.md README.github.md
cp .huggingface/SPACE_README.md README.md

echo "==> Preparing git remote for the Space..."
HF_REMOTE_URL="https://huggingface.co/spaces/${REPO_ID}.git"
REMOTE_NAME="hf-space"

# Ensure we are in a git repo; if not, initialize one and commit everything.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    # remove existing remote with same name to avoid duplicates
    git remote remove "${REMOTE_NAME}" >/dev/null 2>&1 || true
else
    git init
    git add .
    git commit -m "Initial commit for Hugging Face Space" || true
fi

git remote add "${REMOTE_NAME}" "${HF_REMOTE_URL}" || git remote set-url "${REMOTE_NAME}" "${HF_REMOTE_URL}"

echo "==> Pushing project to the Space (this may overwrite remote)..."
# Force push current HEAD to the Space main branch. Change `main` if your Space uses a different default branch.
git push -f "${REMOTE_NAME}" HEAD:main

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
            # Use `hf space secret put` to upload secrets to the Space
            # If your `hf` version uses a slightly different subcommand, adjust accordingly.
            hf space secret put "${REPO_ID}" "${key}" "${value}" || true
        fi
    done < .env
fi

echo
echo "Done. Your Space: https://huggingface.co/spaces/${REPO_ID}"
