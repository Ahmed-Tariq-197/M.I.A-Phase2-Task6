# Flickr8k Image Caption Generator -- serving image.
#
# Builds a lightweight CPU inference image for the FastAPI and Gradio
# front ends. Training is not intended to run inside this image (it
# expects a GPU-equipped host or notebook environment); this image
# only needs the pretrained CNN backbone + a trained decoder checkpoint
# to serve captions.
#
# Build:
#   docker build -t flickr8k-caption-gen .
#
# Run the API (pulling a published model from the Hub):
#   docker run -p 8000:8000 -e HF_MODEL_REPO=your-username/flickr8k-caption-gen flickr8k-caption-gen
#
# Run the Gradio demo, using a locally trained model instead:
#   docker run -p 7860:7860 -e SERVICE=gradio -v $(pwd)/artifacts:/app/artifacts flickr8k-caption-gen

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

WORKDIR /app

# System dependencies needed by Pillow/torchvision image codecs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        libpng16-16 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install the CPU-only PyTorch build explicitly first -- the default PyPI
# wheel on Linux pulls in the full CUDA toolkit (~2GB of nvidia-* packages),
# which this serving image never uses (inference runs on CPU; see header).
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN grep -v -E '^(torch|torchvision)' requirements.txt > requirements-cpu.txt \
    && pip install --no-cache-dir -r requirements-cpu.txt

COPY src/ src/
COPY configs/ configs/
COPY scripts/fetch_model_from_hub.py scripts/fetch_model_from_hub.py
COPY docker-entrypoint.sh healthcheck.sh ./

RUN chmod +x docker-entrypoint.sh healthcheck.sh \
    && mkdir -p artifacts/checkpoints artifacts/features .cache/huggingface .cache/torch \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

ENV SERVICE=api \
    PORT=8000

EXPOSE 8000 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ./healthcheck.sh

ENTRYPOINT ["./docker-entrypoint.sh"]
