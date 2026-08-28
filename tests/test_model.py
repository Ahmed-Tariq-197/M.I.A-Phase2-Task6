import torch

from captioner.models.caption_model import CaptionModel
from captioner.models.decoder import AttentionDecoder, BahdanauAttention


def _build_decoder(vocab):
    return AttentionDecoder(
        vocab_size=len(vocab),
        feature_dim=2048,
        embed_dim=32,
        hidden_dim=64,
        attention_dim=32,
        rnn_type="lstm",
        num_layers=1,
        dropout=0.0,
        pad_id=vocab.pad_id,
    )


def test_attention_output_shapes():
    attn = BahdanauAttention(feature_dim=2048, hidden_dim=64, attention_dim=32)
    features = torch.randn(4, 49, 2048)
    hidden = torch.randn(4, 64)
    context, weights = attn(features, hidden)
    assert context.shape == (4, 2048)
    assert weights.shape == (4, 49)
    # attention weights should sum to ~1 across image locations
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-4)


def test_decoder_forward_shapes(synthetic_vocab):
    decoder = _build_decoder(synthetic_vocab)
    features = torch.randn(3, 49, 2048)
    captions = torch.tensor([
        [synthetic_vocab.start_id, 5, 6, synthetic_vocab.end_id, synthetic_vocab.pad_id],
        [synthetic_vocab.start_id, 5, synthetic_vocab.end_id, synthetic_vocab.pad_id, synthetic_vocab.pad_id],
        [synthetic_vocab.start_id, 5, 6, 7, synthetic_vocab.end_id],
    ])
    lengths = torch.tensor([4, 3, 5])

    logits, alphas, decode_lengths = decoder(features, captions, lengths)
    assert logits.shape[0] == 3
    assert logits.shape[2] == len(synthetic_vocab)
    assert alphas.shape[-1] == 49
    assert torch.equal(decode_lengths, lengths - 1)


def test_decoder_gru_variant_runs(synthetic_vocab):
    decoder = AttentionDecoder(
        vocab_size=len(synthetic_vocab), feature_dim=2048, embed_dim=16, hidden_dim=32,
        attention_dim=16, rnn_type="gru", pad_id=synthetic_vocab.pad_id,
    )
    features = torch.randn(2, 49, 2048)
    captions = torch.tensor([
        [synthetic_vocab.start_id, 5, synthetic_vocab.end_id],
        [synthetic_vocab.start_id, 5, synthetic_vocab.end_id],
    ])
    lengths = torch.tensor([3, 3])
    logits, _, _ = decoder(features, captions, lengths)
    assert logits.shape == (2, 2, len(synthetic_vocab))


def test_caption_model_greedy_generation_terminates(synthetic_vocab):
    decoder = _build_decoder(synthetic_vocab)
    model = CaptionModel(encoder=None, decoder=decoder, vocab=synthetic_vocab)
    features = torch.randn(1, 49, 2048)
    result = model.generate_greedy(features, max_len=15)
    assert isinstance(result.caption, str)
    assert len(result.token_ids) <= 15


def test_caption_model_beam_search_terminates(synthetic_vocab):
    decoder = _build_decoder(synthetic_vocab)
    model = CaptionModel(encoder=None, decoder=decoder, vocab=synthetic_vocab)
    features = torch.randn(1, 49, 2048)
    result = model.generate_beam(features, beam_size=3, max_len=15)
    assert isinstance(result.caption, str)
    assert result.score <= 0  # log-probabilities are non-positive


def test_gradients_flow_through_decoder(synthetic_vocab):
    decoder = _build_decoder(synthetic_vocab)
    features = torch.randn(2, 49, 2048)
    captions = torch.tensor([
        [synthetic_vocab.start_id, 5, 6, synthetic_vocab.end_id],
        [synthetic_vocab.start_id, 5, synthetic_vocab.end_id, synthetic_vocab.pad_id],
    ])
    lengths = torch.tensor([4, 3])
    logits, _, decode_lengths = decoder(features, captions, lengths)
    targets = captions[:, 1:1 + logits.size(1)]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=synthetic_vocab.pad_id
    )
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in decoder.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(g >= 0 for g in grad_norms)
    assert sum(grad_norms) > 0
