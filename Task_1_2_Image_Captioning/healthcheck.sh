#!/usr/bin/env bash
# Used by the Dockerfile HEALTHCHECK instruction. Checks the appropriate
# endpoint depending on which service the container is running.
set -uo pipefail

PORT="${PORT:-8000}"
SERVICE="${SERVICE:-api}"

if [ "$SERVICE" = "gradio" ]; then
  curl -fsS "http://localhost:${PORT}/" > /dev/null
else
  curl -fsS "http://localhost:${PORT}/health" | grep -q '"model_loaded":true' || \
    curl -fsS "http://localhost:${PORT}/health" > /dev/null
fi
