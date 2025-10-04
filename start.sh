#!/bin/bash
set -e

CPU_CORES=$(nproc)
MAX_WORKERS=${MAX_WORKERS:-$CPU_CORES}
MAX_REQUESTS=${MAX_REQUESTS:-2048}

# Use the smaller of MAX_WORKERS and CPU_CORES
if [ "$MAX_WORKERS" -lt "$CPU_CORES" ]; then
  WORKERS="$MAX_WORKERS"
else
  WORKERS="$CPU_CORES"
fi

# Build the Gunicorn command
CMD=(gunicorn app:app -k uvicorn.workers.UvicornWorker \
  --workers "$WORKERS" \
  --bind 0.0.0.0:8080 \
  --max-requests "$MAX_REQUESTS")

# Add timeout if provided
if [ -n "$TIMEOUT" ]; then
  CMD+=(--timeout "$TIMEOUT")
fi

exec "${CMD[@]}"