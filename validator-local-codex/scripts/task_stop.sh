#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${PID_FILE:-$ROOT/.task/validator.pid}"

if [ ! -f "$PID_FILE" ]; then
  echo "no pid file: $PID_FILE"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "stopped validator pid $PID"
else
  echo "pid not running: $PID"
fi
rm -f "$PID_FILE"
