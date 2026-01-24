#!/usr/bin/env bash
set -euo pipefail

cd /workspace/codex-offline-workspace

# This repo ships an offline validator runner and (optionally) offline git bundles.
# If you want to auto-create a bundle during setup, set workspace env vars:
#   BUNDLE_REPO_URL=https://github.com/arrow-py/arrow
#   PINNED_SHA=87a1a774aad0505d9da18ad1d16f6e571f262503
#   BUNDLE_NAME=arrow
#   BUNDLE_REFRESH=1   (optional)
bash scripts/codex_setup.sh
