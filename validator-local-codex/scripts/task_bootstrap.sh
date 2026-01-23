#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
LOG_DIR="${LOG_DIR:-$ROOT/.task}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PID_FILE="$LOG_DIR/validator.pid"
LOG_FILE="$LOG_DIR/validator.log"

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ]; then
  if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "validator already running (pid $(cat "$PID_FILE"))"
    echo "BASE_URL=http://$HOST:$PORT"
    echo "export BASE_URL=http://$HOST:$PORT"
    echo "export VLRUN=$ROOT/vlrun.py"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
. "$VENV_DIR/bin/activate"
python3 -m pip install -q -r "$ROOT/requirements.txt"

nohup python3 -m uvicorn app:app --host "$HOST" --port "$PORT" >"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "validator started (pid $(cat "$PID_FILE"))"
echo "log: $LOG_FILE"
echo "BASE_URL=http://$HOST:$PORT"
echo "export BASE_URL=http://$HOST:$PORT"
echo "export VLRUN=$ROOT/vlrun.py"
