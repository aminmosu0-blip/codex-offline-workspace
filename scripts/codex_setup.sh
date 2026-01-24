#!/usr/bin/env bash
set -euo pipefail

# codex_setup.sh
# Run this as the Codex workspace "setup script".
#
# What it does:
# - Creates stable symlinks expected by common prompts:
#     /workspace/validator-local-codex -> ./validator-local-codex
#     /workspace/*.bundle             -> ./bundles/*.bundle
# - Optionally fetches a repo bundle during setup while internet is enabled.
# - Bootstraps validator-local-codex during setup so later task prompts can run it offline.

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"

# Symlink validator runner to the hardcoded path used in many prompts.
if [[ -d "${root_dir}/validator-local-codex" ]]; then
  ln -sfn "${root_dir}/validator-local-codex" /workspace/validator-local-codex
fi

# If the user provided a repo URL (and optionally PINNED_SHA), fetch/refresh a bundle.
# Set these in Codex workspace env vars:
#   BUNDLE_REPO_URL (required to fetch)
#   PINNED_SHA (optional; validates the pinned commit exists in the clone)
#   BUNDLE_NAME (optional; defaults to repo name, e.g. arrow)
#   BUNDLE_REFRESH=1 (optional; force refresh even if bundle file exists)
if [[ -n "${BUNDLE_REPO_URL:-}" ]]; then
  bash "${root_dir}/scripts/ensure_bundle.sh" "${BUNDLE_REPO_URL}" "${PINNED_SHA:-}" "${BUNDLE_NAME:-}"
fi

# Expose any committed bundles at /workspace/*.bundle
shopt -s nullglob
for p in "${root_dir}/bundles/"*.bundle; do
  bn="$(basename "${p}")"
  ln -sfn "${p}" "/workspace/${bn}"
done
shopt -u nullglob

# Bootstrap validator during setup (internet is enabled during this step).
if [[ -d "${root_dir}/validator-local-codex" ]]; then
  marker="${root_dir}/validator-local-codex/.bootstrap_done"
  if [[ ! -f "${marker}" || "${BOOTSTRAP_REFRESH:-0}" == "1" ]]; then
    ( cd "${root_dir}/validator-local-codex" && bash scripts/task_bootstrap.sh )
    touch "${marker}"
  fi
fi

echo "Setup complete."
