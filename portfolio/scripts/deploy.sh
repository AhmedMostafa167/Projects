#!/usr/bin/env bash
# Deploy this portfolio to your AhmedMostafa167.github.io repository.
#
# What it does:
#   1. Clones (or pulls) AhmedMostafa167.github.io into a temp directory
#   2. Replaces the contents with this portfolio's files
#   3. Commits the change with a sensible message
#   4. Pushes to the default branch -> GitHub Pages picks it up automatically
#
# Usage:
#   bash scripts/deploy.sh
#
# Prereqs:
#   - Git configured with credentials that can push to AhmedMostafa167.github.io
#   - (gh CLI is NOT required, plain git is enough)

set -euo pipefail

REPO_URL="https://github.com/AhmedMostafa167/AhmedMostafa167.github.io.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORTFOLIO_DIR="$(dirname "${SCRIPT_DIR}")"
TMP_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

echo "==> Cloning AhmedMostafa167.github.io..."
git clone --depth 1 "${REPO_URL}" "${TMP_DIR}/repo"
cd "${TMP_DIR}/repo"

DEFAULT_BRANCH="$(git remote show origin | awk '/HEAD branch/ {print $NF}')"
echo "    Default branch: ${DEFAULT_BRANCH}"

echo "==> Backing up existing repo metadata..."
mkdir -p "${TMP_DIR}/backup"
[[ -f .gitignore ]] && cp .gitignore "${TMP_DIR}/backup/" || true
[[ -f LICENSE ]] && cp LICENSE "${TMP_DIR}/backup/" || true
[[ -f CNAME ]] && cp CNAME "${TMP_DIR}/backup/" || true

echo "==> Wiping tracked files (preserving .git)..."
git rm -rf --quiet . 2>/dev/null || true
# Belt-and-braces: also remove any untracked leftovers from old templates
find . -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +

echo "==> Copying new portfolio in..."
# Use rsync if available for a cleaner copy, fall back to cp -R
if command -v rsync >/dev/null 2>&1; then
    rsync -a --exclude '.git' --exclude 'scripts/deploy.sh' \
          --exclude 'README.md' "${PORTFOLIO_DIR}/" ./
else
    cp -R "${PORTFOLIO_DIR}/." ./
    rm -f scripts/deploy.sh README.md
    # Remove empty scripts dir if it ended up empty
    rmdir scripts 2>/dev/null || true
fi

echo "==> Restoring repo metadata..."
[[ -f "${TMP_DIR}/backup/.gitignore" ]] && cp "${TMP_DIR}/backup/.gitignore" ./ || true
[[ -f "${TMP_DIR}/backup/LICENSE" ]] && cp "${TMP_DIR}/backup/LICENSE" ./ || true
[[ -f "${TMP_DIR}/backup/CNAME" ]] && cp "${TMP_DIR}/backup/CNAME" ./ || true

echo "==> Committing..."
git add -A
if git diff --cached --quiet; then
    echo "    No changes to deploy."
    exit 0
fi
git commit -m "Deploy portfolio refresh"

echo "==> Pushing to origin/${DEFAULT_BRANCH}..."
git push origin "HEAD:${DEFAULT_BRANCH}"

# GitHub Pages may be configured to deploy from `main` even when the repo
# default branch is still `master` (or vice versa, from older forks).
# Push to whichever well-known branch is missing so Pages picks it up
# regardless of how Settings -> Pages is configured.
for candidate in main master; do
    if [[ "${candidate}" != "${DEFAULT_BRANCH}" ]]; then
        if git ls-remote --exit-code --heads origin "${candidate}" >/dev/null 2>&1; then
            echo "==> Also syncing origin/${candidate} (in case Pages reads from there)..."
            git push origin "HEAD:${candidate}"
        fi
    fi
done

echo
echo "Done. Your site will be live in ~30-60s at:"
echo "    https://ahmedmostafa167.github.io/"
echo
echo "If the old site still shows: hard-refresh (Ctrl/Cmd + Shift + R)"
echo "or check Pages settings:"
echo "    https://github.com/AhmedMostafa167/AhmedMostafa167.github.io/settings/pages"
