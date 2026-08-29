"""
Text normalisation utilities.

The cleaning rules (lower-casing, contraction expansion, character
whitelisting) are carried over from the original workshop notebook. They are
kept unchanged on purpose: the task is to modernise the *word representation*
used by the model, not the text-processing pipeline that already worked
correctly.
"""
from __future__ import annotations

import re

EN_CONTRACTIONS = {
    "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
    "it's": "it is", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "i will", "you'll": "you will", "he'll": "he will", "she'll": "she will",
    "we'll": "we will", "they'll": "they will",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "couldn't": "could not", "won't": "will not",
    "wouldn't": "would not", "shouldn't": "should not", "haven't": "have not",
    "hasn't": "has not", "hadn't": "had not", "let's": "let us",
}

FR_CONTRACTIONS = {
    "c'est": "ce est", "j'ai": "je ai", "n'est": "ne est",
    "qu'est": "que est", "d'accord": "de accord", "l'est": "le est",
}

# Letters kept for French (accents + oe ligature).
FR_KEEP = r"a-zàâäçéèêëîïôùûüœÿ\s"


def _expand(text: str, table: dict) -> str:
    for src_tok, dst_tok in table.items():
        text = re.sub(rf"\b{re.escape(src_tok)}\b", dst_tok, text)
    return text


def clean_english(text: str) -> str:
    text = str(text).lower().strip()
    text = _expand(text, EN_CONTRACTIONS)
    text = re.sub(r"[^a-z\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_french(text: str) -> str:
    text = str(text).lower().strip()
    text = _expand(text, FR_CONTRACTIONS)
    text = re.sub(rf"[^{FR_KEEP}]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
