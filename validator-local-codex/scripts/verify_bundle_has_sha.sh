#!/usr/bin/env bash
set -euo pipefail
if [ $# -ne 2 ]; then
  echo "Usage: $0 /abs/path/to/repo.bundle <PINNED_SHA>" >&2
  exit 2
fi
BUNDLE="$1"
SHA="$2"
TMP=$(mktemp -d)
cleanup(){ rm -rf "$TMP"; }
trap cleanup EXIT
# Create a temp repo and verify the commit is present.
git init -q "$TMP/repo"
cd "$TMP/repo"
git fetch -q "$BUNDLE" "refs/*:refs/*" || true
if git cat-file -e "$SHA^{commit}" 2>/dev/null; then
  echo "OK: bundle contains $SHA"
  exit 0
fi
echo "ERROR: bundle does not contain $SHA" >&2
exit 1
