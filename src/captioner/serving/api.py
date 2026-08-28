"""FastAPI service exposing image-captioning inference over HTTP.

Endpoints:
    GET  /health          -- liveness/readiness probe
    GET  /                -- basic service info
    POST /caption         -- upload an image, get a generated caption

Run locally with:
    uvicorn captioner.serving.api:app --host 0.0.0.0 --port 8000

The model is loaded once at process startup (not per-request) via
FastAPI's lifespan hook, so repeated requests are fast.
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from captioner.config import load_config
from captioner.inference.predict import CaptionPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_state: dict = {"predictor": None, "load_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    try:
        _state["predictor"] = CaptionPredictor(cfg=cfg)
    except Exception as exc:  # noqa: BLE001 - surfaced via /health instead of crashing the process
        logger.exception("Failed to load CaptionPredictor at startup")
        _state["load_error"] = str(exc)
    yield
    _state["predictor"] = None


app = FastAPI(
    title="Flickr8k Image Caption Generator",
    description="Upload an image and receive an automatically generated natural-language caption.",
    version="1.0.0",
    lifespan=lifespan,
)


class CaptionResponse(BaseModel):
    caption: str
    beam_size: Optional[int] = None
    decoding: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    detail: Optional[str] = None


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "flickr8k-caption-gen",
        "docs": "/docs",
        "endpoints": ["/health", "/caption"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    if _state["predictor"] is not None:
        return HealthResponse(status="ok", model_loaded=True)
    return HealthResponse(status="degraded", model_loaded=False, detail=_state["load_error"])


@app.post("/caption", response_model=CaptionResponse, tags=["inference"])
async def caption_image(
    file: UploadFile = File(..., description="Image file (jpg/png)."),
    beam: bool = Query(True, description="Use beam search (higher quality, slightly slower)."),
    beam_size: Optional[int] = Query(None, ge=1, le=10, description="Override the configured beam width."),
):
    predictor: Optional[CaptionPredictor] = _state["predictor"]
    if predictor is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {_state['load_error']}")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    result = predictor.predict(image, use_beam=beam, beam_size=beam_size)
    return CaptionResponse(
        caption=result.caption,
        beam_size=beam_size or (predictor.cfg.evaluation.beam_size if beam else None),
        decoding="beam" if beam else "greedy",
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Unhandled error while processing request")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
