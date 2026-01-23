#!/usr/bin/env bash
set -euo pipefail

echo "pwd=$(pwd)"
echo "python=$(python3 --version)"
if command -v docker >/dev/null 2>&1; then
  echo "docker=$(docker --version || true)"
  docker version >/dev/null 2>&1 && echo "docker_ok=1" || echo "docker_ok=0"
else
  echo "docker=missing"
  echo "docker_ok=0"
fi
