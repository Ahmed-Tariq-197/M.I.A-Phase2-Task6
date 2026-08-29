"""Image-captioning evaluation metrics: BLEU-1..4, ROUGE-L, METEOR.

All functions take ``references: List[List[str]]`` (one list of raw
reference-caption strings per image -- Flickr8k gives 5) and
``hypotheses: List[str]`` (one generated caption string per image), in
matching order.
"""

from __future__ import annotations

from typing import List

import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer


def _ensure_nltk_data() -> None:
    for pkg, path in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
    for pkg, path in [("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def _tokenize(text: str) -> List[str]:
    _ensure_nltk_data()
    try:
        return word_tokenize(text.lower())
    except LookupError:
        return text.lower().split()


def corpus_bleu4(references: List[List[str]], hypotheses: List[str]) -> float:
    """Corpus-level BLEU-4 with NLTK's method-1 smoothing (avoids zero
    scores from short captions with no 4-gram overlap)."""
    _ensure_nltk_data()
    refs_tok = [[_tokenize(r) for r in refs] for refs in references]
    hyps_tok = [_tokenize(h) for h in hypotheses]
    smoothing = SmoothingFunction().method1
    return corpus_bleu(refs_tok, hyps_tok, smoothing_function=smoothing)


def bleu_n_scores(references: List[List[str]], hypotheses: List[str]) -> dict:
    """Returns corpus BLEU-1, BLEU-2, BLEU-3, BLEU-4."""
    _ensure_nltk_data()
    refs_tok = [[_tokenize(r) for r in refs] for refs in references]
    hyps_tok = [_tokenize(h) for h in hypotheses]
    smoothing = SmoothingFunction().method1
    weights = {
        "bleu1": (1.0, 0, 0, 0),
        "bleu2": (0.5, 0.5, 0, 0),
        "bleu3": (1 / 3, 1 / 3, 1 / 3, 0),
        "bleu4": (0.25, 0.25, 0.25, 0.25),
    }
    return {
        name: corpus_bleu(refs_tok, hyps_tok, weights=w, smoothing_function=smoothing)
        for name, w in weights.items()
    }


def rouge_l_score(references: List[List[str]], hypotheses: List[str]) -> float:
    """Average ROUGE-L F1 across images, taking the best-matching reference
    caption per image (standard practice for multi-reference captioning)."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for refs, hyp in zip(references, hypotheses):
        best = max(scorer.score(ref, hyp)["rougeL"].fmeasure for ref in refs)
        scores.append(best)
    return sum(scores) / max(len(scores), 1)


def meteor_score_avg(references: List[List[str]], hypotheses: List[str]) -> float:
    """Average METEOR score across images (max over the 5 references)."""
    _ensure_nltk_data()
    from nltk.translate.meteor_score import meteor_score

    scores = []
    for refs, hyp in zip(references, hypotheses):
        refs_tok = [_tokenize(r) for r in refs]
        hyp_tok = _tokenize(hyp)
        scores.append(meteor_score(refs_tok, hyp_tok))
    return sum(scores) / max(len(scores), 1)


def evaluate_all_metrics(references: List[List[str]], hypotheses: List[str]) -> dict:
    """Convenience wrapper computing every metric at once."""
    result = bleu_n_scores(references, hypotheses)
    result["rougeL"] = rouge_l_score(references, hypotheses)
    result["meteor"] = meteor_score_avg(references, hypotheses)
    return result
