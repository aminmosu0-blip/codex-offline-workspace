#!/usr/bin/env bash
set -euo pipefail

# Build a clean validator-local.zip next to this script.
# Excludes runtime/cache files even if present locally.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${ROOT_DIR%/}/validator-local.zip"

cd "$ROOT_DIR/.."

zip -r "$OUT" validator-local \
  -x "validator-local/.venv/*" \
     "validator-local/__pycache__/*" \
     "validator-local/*.pyc" \
     "validator-local/server.log" \
     "validator-local/data/*"

echo "Wrote: $OUT"
