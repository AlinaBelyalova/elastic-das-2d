#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SITE_DIR="${REPO_ROOT}/web_safod/site"
PAGES_CHECKOUT="${1:-${PAGES_CHECKOUT:-}}"

if [[ -z "${PAGES_CHECKOUT}" ]]; then
  echo "Usage: $0 /path/to/safod-event-explorer-pages"
  echo "or set PAGES_CHECKOUT."
  exit 2
fi

PYTHON="${PYTHON:-python}"

cd "${REPO_ROOT}"
"${PYTHON}" -m web_safod.scripts.validate_site

if [[ ! -d "${PAGES_CHECKOUT}/.git" ]]; then
  echo "Pages checkout is not a Git repository: ${PAGES_CHECKOUT}"
  exit 2
fi

rsync -av --delete --exclude='.git/' \
  "${SITE_DIR}/" \
  "${PAGES_CHECKOUT}/"

cd "${PAGES_CHECKOUT}"

git status --short
git add -A

if git diff --cached --quiet; then
  echo "No public-site changes to publish."
  exit 0
fi

git commit -m "Update SAFOD event explorer"
git push origin main

echo "Published SAFOD event explorer."
