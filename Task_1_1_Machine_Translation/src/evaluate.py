"""
Translation quality evaluation: BLEU and ROUGE.

Both metrics are computed on the model's native output space: lower-cased,
punctuation-stripped, whitespace-tokenised French -- the same normalisation
used throughout training. This keeps the comparison between the baseline
and the FastText-initialised model apples-to-apples; it also means the
absolute BLEU/ROUGE numbers reported here are not directly comparable to
numbers reported on raw, cased, punctuated text elsewhere.

ROUGE is computed without the (English, Porter) stemmer, since stemming
French text with an English stemmer would silently corrupt the metric.
"""
from __future__ import annotations

from typing import Dict, List

import sacrebleu
from rouge_score import rouge_scorer


def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    """Corpus-level BLEU (0-100 scale) via sacrebleu."""
    hyps = [h if h.strip() else "<empty>" for h in hypotheses]
    bleu = sacrebleu.corpus_bleu(hyps, [references])
    return bleu.score


def compute_rouge(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """Average ROUGE-1 / ROUGE-2 / ROUGE-L F-measure across the test set."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = len(references)
    for hyp, ref in zip(hypotheses, references):
        hyp = hyp if hyp.strip() else "<empty>"
        scores = scorer.score(ref, hyp)
        for key in totals:
            totals[key] += scores[key].fmeasure
    return {key: (value / max(1, n)) * 100 for key, value in totals.items()}


def evaluate_translations(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    metrics = {"bleu": compute_bleu(hypotheses, references)}
    metrics.update(compute_rouge(hypotheses, references))
    return metrics
