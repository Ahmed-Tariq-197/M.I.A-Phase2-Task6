"""Flickr8k image captioning package.

Modules:
    config      -- YAML configuration loading
    data        -- dataset download, splitting, vocabulary, PyTorch Dataset
    features    -- pretrained CNN encoders and feature caching
    models      -- encoder/decoder/attention caption model
    training    -- training loop, checkpointing, early stopping
    evaluation  -- BLEU / ROUGE / METEOR metrics and test-set evaluation
    inference   -- single-image caption generation (greedy + beam search)
    serving     -- FastAPI and Gradio front ends
"""

__version__ = "1.0.0"
