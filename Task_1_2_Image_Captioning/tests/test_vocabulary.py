from captioner.data.vocabulary import Vocabulary, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    tokens = tokenize("A dog runs Through the Park!!")
    assert tokens == ["a", "dog", "runs", "through", "the", "park"]


def test_vocab_contains_special_tokens():
    vocab = Vocabulary.build(["a dog runs"], min_word_freq=1)
    assert vocab.pad_token in vocab
    assert vocab.start_token in vocab
    assert vocab.end_token in vocab
    assert vocab.unk_token in vocab


def test_rare_words_map_to_unk():
    captions = ["a common word"] * 10 + ["a rareword appears once"]
    vocab = Vocabulary.build(captions, min_word_freq=5)
    assert vocab.word_to_id("common") != vocab.unk_id
    assert vocab.word_to_id("rareword") == vocab.unk_id


def test_encode_adds_start_and_end_tokens():
    vocab = Vocabulary.build(["a dog runs"], min_word_freq=1)
    ids = vocab.encode("a dog runs")
    assert ids[0] == vocab.start_id
    assert ids[-1] == vocab.end_id


def test_encode_truncates_to_max_len():
    vocab = Vocabulary.build(["one two three four five six seven eight"], min_word_freq=1)
    ids = vocab.encode("one two three four five six seven eight", max_len=5)
    assert len(ids) <= 5
    assert ids[-1] == vocab.end_id


def test_decode_stops_at_end_token():
    vocab = Vocabulary.build(["a dog runs fast"], min_word_freq=1)
    ids = vocab.encode("a dog runs")
    decoded = vocab.decode(ids)
    assert decoded == "a dog runs"


def test_unknown_word_at_inference_maps_to_unk_not_crash():
    vocab = Vocabulary.build(["a dog runs"], min_word_freq=1)
    ids = vocab.encode("a dog flies through a wormhole")
    assert vocab.unk_id in ids  # "flies" and "wormhole" are unseen


def test_save_and_load_roundtrip(tmp_path):
    vocab = Vocabulary.build(["a dog runs in the park"], min_word_freq=1)
    path = tmp_path / "vocab.json"
    vocab.save(path)
    loaded = Vocabulary.load(path)
    assert len(loaded) == len(vocab)
    assert loaded.word_to_id("dog") == vocab.word_to_id("dog")
