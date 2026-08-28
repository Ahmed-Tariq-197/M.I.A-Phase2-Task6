"""Gradio demo: upload an image, get a generated caption.

Run with:
    python -m captioner.serving.app_gradio

This is the front end referenced in the README's demo/video section and
is also what gets deployed as a Hugging Face Space.
"""

from __future__ import annotations

import logging

import gradio as gr

from captioner.config import load_config
from captioner.inference.predict import CaptionPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_cfg = load_config()
_predictor: CaptionPredictor | None = None
_load_error: str | None = None

try:
    _predictor = CaptionPredictor(cfg=_cfg)
except Exception as exc:  # noqa: BLE001
    logger.exception("Could not load model for the Gradio demo")
    _load_error = str(exc)


def generate_caption(image, decoding_strategy: str, beam_size: int):
    if _predictor is None:
        return f"Model unavailable: {_load_error}"
    if image is None:
        return "Please upload an image first."
    result = _predictor.predict(
        image, use_beam=(decoding_strategy == "Beam search"), beam_size=int(beam_size)
    )
    return result.caption


with gr.Blocks(title="Flickr8k Image Caption Generator", analytics_enabled=False) as demo:
    gr.Markdown(
        "# Image Caption Generator\n"
        "Upload a photo and the model will generate a natural-language description, "
        "trained on the Flickr8k dataset with a ResNet-based encoder and an "
        "attention LSTM/GRU decoder."
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload an image")
            decoding_choice = gr.Radio(
                ["Beam search", "Greedy"], value="Beam search", label="Decoding strategy"
            )
            beam_size_slider = gr.Slider(1, 10, value=_cfg.evaluation.beam_size, step=1, label="Beam size")
            submit_btn = gr.Button("Generate caption", variant="primary")
        with gr.Column():
            caption_output = gr.Textbox(label="Generated caption", lines=3)

    submit_btn.click(
        fn=generate_caption,
        inputs=[image_input, decoding_choice, beam_size_slider],
        outputs=caption_output,
    )
    gr.Examples(
        examples=[],  # populated at deploy time with a few sample Flickr8k test images
        inputs=image_input,
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", _cfg.serving.gradio_port))
    demo.launch(server_name=_cfg.serving.api_host, server_port=port)
