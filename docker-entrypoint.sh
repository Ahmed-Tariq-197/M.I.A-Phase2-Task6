#!/usr/bin/env bash
# Container entrypoint: make sure model artifacts are present (pulling
# them from the Hugging Face Hub if necessary), then launch the
# requested service.
#
# SERVICE=api    -> FastAPI on $PORT (default 8000)
# SERVICE=gradio -> Gradio demo on $PORT (default 7860)
set -euo pipefail

SERVICE="${SERVICE:-api}"
PORT="${PORT:-8000}"
HF_MODEL_REPO="${HF_MODEL_REPO:-}"

if [ -n "${HF_MODEL_REPO}" ]; then
  echo "Ensuring model artifacts are present (repo: ${HF_MODEL_REPO}) ..."
  python scripts/fetch_model_from_hub.py --repo-id "${HF_MODEL_REPO}" || \
    echo "WARNING: could not fetch model from the Hub; relying on any locally mounted artifacts/ instead."
fi

if [ ! -f "artifacts/checkpoints/best_model.pt" ] || [ ! -f "artifacts/vocab.json" ]; then
  echo "WARNING: no model checkpoint/vocabulary found at artifacts/. Mount a volume with"
  echo "trained artifacts, or set HF_MODEL_REPO to pull a published model from the Hub."
fi

case "$SERVICE" in
  api)
    exec uvicorn captioner.serving.api:app --host 0.0.0.0 --port "${PORT}"
    ;;
  gradio)
    export PORT
    exec python -m captioner.serving.app_gradio
    ;;
  *)
    echo "Unknown SERVICE '${SERVICE}'. Use 'api' or 'gradio'." >&2
    exit 1
    ;;
esac
