import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import captioner.serving.api as api_module
from captioner.models.caption_model import GenerationResult


class _StubPredictor:
    """Stands in for CaptionPredictor so API tests don't need a trained
    checkpoint or the CNN backbone weights on disk."""

    class _StubCfg:
        class evaluation:
            beam_size = 3

    cfg = _StubCfg()

    def predict(self, image, use_beam=True, beam_size=None):
        return GenerationResult(token_ids=[1, 2, 3], caption="a stub generated caption", score=-1.23)


@pytest.fixture()
def client():
    api_module._state["predictor"] = _StubPredictor()
    api_module._state["load_error"] = None
    with TestClient(api_module.app) as c:
        yield c
    api_module._state["predictor"] = None


def _sample_image_bytes():
    img = Image.new("RGB", (32, 32), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health_endpoint_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_root_endpoint_lists_docs(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "endpoints" in resp.json()


def test_caption_endpoint_returns_generated_text(client):
    files = {"file": ("test.jpg", _sample_image_bytes(), "image/jpeg")}
    resp = client.post("/caption", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["caption"] == "a stub generated caption"
    assert body["decoding"] == "beam"


def test_caption_endpoint_rejects_non_image(client):
    files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    resp = client.post("/caption", files=files)
    assert resp.status_code == 400


def test_health_reports_degraded_when_model_missing(client):
    api_module._state["predictor"] = None
    api_module._state["load_error"] = "checkpoint not found"
    resp = client.get("/health")
    assert resp.json()["status"] == "degraded"


def test_caption_endpoint_503_when_model_missing(client):
    api_module._state["predictor"] = None
    files = {"file": ("test.jpg", _sample_image_bytes(), "image/jpeg")}
    resp = client.post("/caption", files=files)
    assert resp.status_code == 503
