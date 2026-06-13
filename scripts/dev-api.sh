#!/usr/bin/env bash
# Start the FastAPI backend for local development.
# Prefers the project virtualenv at apps/api/.venv, falling back to python3.
set -euo pipefail

cd "$(dirname "$0")/../apps/api"

if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"
else
  echo "No virtualenv found at apps/api/.venv. Create one with:" >&2
  echo "  cd apps/api && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  echo "Falling back to system python3." >&2
  PY="python3"
fi

exec "$PY" -m uvicorn main:app --reload --port "${API_PORT:-8080}"
