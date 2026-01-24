#!/usr/bin/env bash
set -euo pipefail

# ensure_bundle.sh
# Create or refresh an offline git bundle in ./bundles/ from a remote repo URL.
#
# Usage:
#   bash scripts/ensure_bundle.sh <repo_url> [pinned_sha] [bundle_name]
#
# Notes:
# - Intended to run during Codex "setup script" while internet is enabled.
# - Produces: bundles/<bundle_name>.bundle
# - Also creates: /workspace/<bundle_name>.bundle symlink for prompts that hardcode /workspace paths.

repo_url="${1:-}"
pinned_sha="${2:-}"
bundle_name="${3:-}"

if [[ -z "${repo_url}" ]]; then
  echo "ERROR: repo_url is required" >&2
  exit 2
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bundles_dir="${root_dir}/bundles"
mkdir -p "${bundles_dir}"

if [[ -z "${bundle_name}" ]]; then
  # Derive from URL: owner/name(.git) -> name
  base="${repo_url##*/}"
  bundle_name="${base%.git}"
fi

bundle_path="${bundles_dir}/${bundle_name}.bundle"

if [[ -f "${bundle_path}" && "${BUNDLE_REFRESH:-0}" != "1" ]]; then
  echo "Bundle already present: ${bundle_path}"
else
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "${tmp_dir}"' EXIT

  echo "Cloning ${repo_url} ..."
  git clone --quiet "${repo_url}" "${tmp_dir}/repo"

  if [[ -n "${pinned_sha}" ]]; then
    git -C "${tmp_dir}/repo" cat-file -e "${pinned_sha}^{commit}" >/dev/null
  fi

  echo "Creating bundle: ${bundle_path}"
  git -C "${tmp_dir}/repo" bundle create "${bundle_path}" --all
fi

# Provide hardcoded /workspace/<name>.bundle for agent prompts.
ln -sfn "${bundle_path}" "/workspace/${bundle_name}.bundle"

# Convenience: if this is arrow.bundle, also expose /workspace/arrow.bundle.
if [[ "${bundle_name}" == "arrow" ]]; then
  ln -sfn "${bundle_path}" "/workspace/arrow.bundle"
fi

echo "OK: ${bundle_path}"
