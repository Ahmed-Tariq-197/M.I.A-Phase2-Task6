from captioner.evaluation.metrics import bleu_n_scores, corpus_bleu4, evaluate_all_metrics, meteor_score_avg, rouge_l_score


def test_perfect_match_scores_near_one():
    references = [["a dog runs on the grass"]]
    hypotheses = ["a dog runs on the grass"]
    bleu4 = corpus_bleu4(references, hypotheses)
    assert bleu4 > 0.9


def test_completely_wrong_caption_scores_low():
    references = [["a dog runs on the grass"]]
    hypotheses = ["a rocket launches into orbit today"]
    bleu4 = corpus_bleu4(references, hypotheses)
    assert bleu4 < 0.2


def test_bleu_n_scores_returns_all_four_orders():
    references = [["a dog runs on the grass"], ["a cat sleeps on the mat"]]
    hypotheses = ["a dog runs on grass", "a cat sleeps on a mat"]
    scores = bleu_n_scores(references, hypotheses)
    assert set(scores.keys()) == {"bleu1", "bleu2", "bleu3", "bleu4"}
    for v in scores.values():
        assert 0.0 <= v <= 1.0
    # higher n-gram order should generally be harder to satisfy
    assert scores["bleu1"] >= scores["bleu4"]


def test_rouge_l_uses_best_reference():
    references = [["a completely unrelated sentence", "a dog runs on the grass"]]
    hypotheses = ["a dog runs on the grass"]
    score = rouge_l_score(references, hypotheses)
    assert score > 0.9  # should match the second, closer reference


def test_meteor_score_reasonable_range():
    references = [["a dog runs on the grass"]]
    hypotheses = ["a dog running on grass"]
    score = meteor_score_avg(references, hypotheses)
    assert 0.0 < score <= 1.0


def test_evaluate_all_metrics_returns_expected_keys():
    references = [["a dog runs on the grass"]]
    hypotheses = ["a dog runs on the grass"]
    result = evaluate_all_metrics(references, hypotheses)
    assert set(result.keys()) == {"bleu1", "bleu2", "bleu3", "bleu4", "rougeL", "meteor"}


def test_render_qualitative_examples_produces_png(tmp_path, synthetic_images_dir):
    from captioner.evaluation.evaluate import render_qualitative_examples

    images_dir, image_ids = synthetic_images_dir
    examples = [
        {
            "image_id": image_ids[0],
            "generated_caption": "a red square",
            "reference_captions": ["a red square on a plain background", "a bright red shape"],
        }
    ]
    out_path = tmp_path / "qualitative_examples.png"
    render_qualitative_examples(examples, images_dir, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
