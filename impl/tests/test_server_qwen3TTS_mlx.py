"""Tests for ``server_qwen3TTS_mlx.py``.

Covers the model-free HTTP surface: /capabilities (snapshot), /health, the
landing page, request-body validation (422s), text chunking behavior,
synthesis behavior and edge cases, and reference-audio pre-flight checks
(400s).

Real synthesis needs mlx-audio + Apple Silicon and is out of scope here --
this suite must never import MLX or load the model.
"""

import types

import numpy as np
import pytest
import server_qwen3TTS_mlx as srv
from fastapi.testclient import TestClient
from helpers import b64, load_snapshot, make_wav_bytes


@pytest.fixture(scope="module")
def client():
    # Deliberately NOT a context manager: entering it would run the FastAPI
    # lifespan, which loads the (stubbed) model. Not needed for these tests.
    return TestClient(srv.app)


@pytest.fixture
def fake_runtime(monkeypatch):
    runtime = types.SimpleNamespace(
        model=None,
        sample_rate=srv.SAMPLE_RATE,
        device=srv.DEVICE,
    )

    monkeypatch.setattr(srv, "_runtime", runtime)
    monkeypatch.setattr(
        srv,
        "_decode_wav",
        lambda raw: (object(), srv.SAMPLE_RATE),
    )
    monkeypatch.setattr(
        srv,
        "_prepare_ref_audio",
        lambda *args, **kwargs: object(),
    )

    return runtime


# ---------------------------------------------------------------------------
# GET /capabilities
# ---------------------------------------------------------------------------


def test_capabilities_matches_snapshot(client):
    response = client.get("/capabilities")
    assert response.status_code == 200
    doc = response.json()
    assert doc == load_snapshot("qwen3_mlx_capabilities.json")


def test_capabilities_language_enum_is_codes_plus_auto(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    enum = by_name["language"]["enum"]

    # The API contract is two-letter codes (docs/02) plus the engine's
    # 'auto' auto-detection sentinel; the engine-internal names must not
    # leak into the document.
    assert "auto" in enum
    assert "en" in enum
    assert "zh" in enum
    assert "english" not in enum
    assert doc["languages"] == sorted(srv.LANGUAGE_CODES)


def test_capabilities_required_fields(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}

    assert by_name["text"]["required"] is True
    assert by_name["audio_base64"]["required"] is True

    # ICL cloning only: without a transcript the request is invalid at the
    # boundary, not a fallback to some other cloning mode.
    assert by_name["reference_text"]["required"] is True

    assert by_name["language"]["required"] is False
    assert by_name["seed"]["required"] is False

    assert by_name["chunking_enabled"]["required"] is False
    assert by_name["chunk_min_chars"]["required"] is False
    assert by_name["chunk_max_chars"]["required"] is False
    assert by_name["chunk_silence_ms"]["required"] is False
    assert by_name["chunk_crossfade_ms"]["required"] is False


def test_capabilities_device_and_sample_rate(client):
    doc = client.get("/capabilities").json()

    assert doc["device"] == "mlx"
    assert doc["sample_rate"] == 24000
    assert doc["engine"] == "qwen3-tts-mlx"


# ---------------------------------------------------------------------------
# GET /health and landing page
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["serverType"] == "Qwen3-TTS-MLX"
    assert body["device"] == "mlx"
    assert body["model"] == srv.MODEL_NAME_OR_PATH


def test_root_landing_page(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Qwen3-TTS MLX" in response.text
    assert "/synthesize" in response.text


# ---------------------------------------------------------------------------
# Shared synthesis-test helpers
# ---------------------------------------------------------------------------


def _post(client, payload):
    return client.post("/synthesize", json=payload)


def _valid_payload():
    return {
        "text": "Hello there",
        "audio_base64": b64(make_wav_bytes(3.0)),
        "reference_text": "A short, exact transcript of the reference clip.",
    }


def _generated_audio(samples: int, sample_rate: int = srv.SAMPLE_RATE):
    return types.SimpleNamespace(
        audio=[0.1] * samples,
        sample_rate=sample_rate,
    )


# ---------------------------------------------------------------------------
# POST /synthesize — request-body validation (422)
# ---------------------------------------------------------------------------


def test_synthesize_unknown_field_rejected(client):
    payload = _valid_payload()
    payload["bogus_field"] = 1

    assert _post(client, payload).status_code == 422


def test_synthesize_missing_required_fields_rejected(client):
    assert _post(client, {}).status_code == 422


def test_synthesize_missing_reference_text_rejected(client):
    # ICL cloning only: without a transcript the request is invalid at the
    # boundary, not a 400 or a fallback to some other cloning mode.
    payload = _valid_payload()
    del payload["reference_text"]

    assert _post(client, payload).status_code == 422


def test_synthesize_empty_text_rejected(client):
    payload = _valid_payload()
    payload["text"] = ""

    assert _post(client, payload).status_code == 422


def test_synthesize_whitespace_text_rejected(client):
    payload = _valid_payload()
    payload["text"] = "   "

    assert _post(client, payload).status_code == 422


def test_synthesize_empty_audio_rejected(client):
    payload = _valid_payload()
    payload["audio_base64"] = ""

    assert _post(client, payload).status_code == 422


def test_synthesize_empty_reference_text_rejected(client):
    payload = _valid_payload()
    payload["reference_text"] = ""

    assert _post(client, payload).status_code == 422


def test_synthesize_whitespace_reference_text_rejected(client):
    payload = _valid_payload()
    payload["reference_text"] = "   "

    assert _post(client, payload).status_code == 422


def test_synthesize_unsupported_language_rejected(client):
    # A valid code that the Base checkpoint does not support.
    payload = _valid_payload()
    payload["language"] = "xx"

    assert _post(client, payload).status_code == 422


def test_synthesize_seed_below_min_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 0

    assert _post(client, payload).status_code == 422


def test_synthesize_seed_above_max_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 1001

    assert _post(client, payload).status_code == 422


def test_synthesize_chunk_min_greater_than_max_rejected(client):
    payload = _valid_payload()
    payload["chunking_enabled"] = True
    payload["chunk_min_chars"] = 500
    payload["chunk_max_chars"] = 250

    response = _post(client, payload)

    assert response.status_code == 422


def test_synthesize_negative_chunk_silence_rejected(client):
    payload = _valid_payload()
    payload["chunk_silence_ms"] = -1

    response = _post(client, payload)

    assert response.status_code == 422


def test_synthesize_chunk_crossfade_defaults_to_zero(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}

    assert by_name["chunk_crossfade_ms"]["default"] == 0
    assert by_name["chunk_crossfade_ms"]["required"] is False


def test_synthesize_negative_chunk_crossfade_rejected(client):
    payload = _valid_payload()
    payload["chunk_crossfade_ms"] = -1

    response = _post(client, payload)

    assert response.status_code == 422


def test_synthesize_chunk_crossfade_above_max_rejected(client):
    payload = _valid_payload()
    payload["chunk_crossfade_ms"] = 51

    response = _post(client, payload)

    assert response.status_code == 422


def test_synthesize_silence_and_crossfade_together_rejected(client):
    payload = _valid_payload()
    payload["chunk_silence_ms"] = 50
    payload["chunk_crossfade_ms"] = 10

    response = _post(client, payload)

    assert response.status_code == 422


# The shared docs/02 language contract (case, names, garbage, non-strings,
# null/empty -> 'en') is asserted once for every server in
# test_language_contract.py; keep only engine-specific language tests here.


# ---------------------------------------------------------------------------
# Text chunking helper
# ---------------------------------------------------------------------------


def test_split_text_chunks_short_text_is_single_chunk():
    text = "This is short."

    assert srv._split_text_chunks(
        text,
        min_chars=10,
        max_chars=100,
    ) == [text]


def test_split_text_chunks_prefers_sentence_boundary():
    text = (
        "This is the first sentence. "
        "This is the second sentence. "
        "This is the third sentence."
    )

    chunks = srv._split_text_chunks(
        text,
        min_chars=20,
        max_chars=55,
    )

    assert len(chunks) > 1
    assert chunks[0].endswith(".")
    assert all(len(chunk) <= 55 for chunk in chunks)


def test_split_text_chunks_prefers_soft_punctuation_when_needed():
    text = (
        "This section contains several words, "
        "and another useful phrase, "
        "followed by more material without a sentence ending"
    )

    chunks = srv._split_text_chunks(
        text,
        min_chars=20,
        max_chars=65,
    )

    assert len(chunks) > 1
    assert chunks[0].endswith(",")
    assert all(len(chunk) <= 65 for chunk in chunks)


def test_split_text_chunks_uses_word_boundary():
    text = "one two three four five six seven eight nine ten eleven twelve"

    chunks = srv._split_text_chunks(
        text,
        min_chars=10,
        max_chars=25,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 25 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_split_text_chunks_hard_splits_long_token():
    text = "x" * 120

    chunks = srv._split_text_chunks(
        text,
        min_chars=20,
        max_chars=50,
    )

    assert chunks == [
        "x" * 50,
        "x" * 50,
        "x" * 20,
    ]


# ---------------------------------------------------------------------------
# Crossfade helper
# ---------------------------------------------------------------------------


def test_crossfade_single_chunk_returned_unchanged():
    chunk = np.arange(100, dtype=np.float32)

    result = srv._crossfade_audio_chunks([chunk], 24000, 10)

    assert np.array_equal(result, chunk)


def test_crossfade_disabled_is_plain_concatenation():
    a = np.ones(1000, dtype=np.float32)
    b = np.zeros(1000, dtype=np.float32) + 2.0

    result = srv._crossfade_audio_chunks([a, b], 24000, 0)

    assert len(result) == len(a) + len(b)
    assert np.array_equal(result, np.concatenate([a, b]))


def test_crossfade_two_chunks_overlap_and_length():
    a = np.ones(1000, dtype=np.float32)
    b = np.ones(1000, dtype=np.float32) * 2.0

    result = srv._crossfade_audio_chunks([a, b], 24000, 10)

    overlap = round(24000 * 10 / 1000)
    assert overlap == 240
    assert len(result) == len(a) + len(b) - overlap


def test_crossfade_three_chunks_two_joins_length():
    a = np.ones(1000, dtype=np.float32)
    b = np.ones(1000, dtype=np.float32) * 2.0
    c = np.ones(1000, dtype=np.float32) * 3.0

    result = srv._crossfade_audio_chunks([a, b, c], 24000, 10)

    overlap = round(24000 * 10 / 1000)
    assert len(result) == len(a) + len(b) + len(c) - 2 * overlap


def test_crossfade_requested_overlap_larger_than_chunk_is_clamped():
    a = np.ones(10, dtype=np.float32)
    b = np.ones(1000, dtype=np.float32) * 2.0

    # 50 ms at 24 kHz would request 1200 samples of overlap, far larger than
    # `a`; the join must clamp to len(a) instead of raising or corrupting.
    result = srv._crossfade_audio_chunks([a, b], 24000, 50)

    assert len(result) == len(a) + len(b) - len(a)
    assert len(result) == len(b)


def test_crossfade_blend_is_smooth_and_gain_preserving():
    # Deterministic arrays so the blend itself can be checked, not just the
    # resulting length.
    a = np.ones(1000, dtype=np.float32)
    b = np.zeros(1000, dtype=np.float32)

    result = srv._crossfade_audio_chunks([a, b], 24000, 10)

    overlap = round(24000 * 10 / 1000)
    join_start = len(a) - overlap
    overlap_region = result[join_start : join_start + overlap]

    # No amplification/clipping: values in the overlap stay within [0, 1],
    # the range spanned by the two chunks being blended.
    assert np.all(overlap_region <= 1.0 + 1e-6)
    assert np.all(overlap_region >= 0.0 - 1e-6)

    # Correct fade direction: a (1.0) fading toward b (0.0) is monotonically
    # non-increasing across the overlap, starting near 1.0 and ending near 0.0.
    assert overlap_region[0] > overlap_region[-1]
    assert overlap_region[0] == pytest.approx(1.0, abs=1e-6)
    assert overlap_region[-1] == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(overlap_region) <= 1e-6)

    # Audio outside the overlap is untouched.
    assert np.array_equal(result[:join_start], a[:join_start])
    assert np.array_equal(result[join_start + overlap :], b[overlap:])


# ---------------------------------------------------------------------------
# POST /synthesize — synthesis behavior and edge cases
# ---------------------------------------------------------------------------


def test_synthesize_no_generated_audio_returns_500(client, fake_runtime):
    fake_runtime.model = types.SimpleNamespace(generate=lambda **kwargs: iter(()))

    payload = _valid_payload()
    response = _post(client, payload)

    assert response.status_code == 500
    assert "produced no audio" in response.json()["detail"].lower()


def test_synthesize_chunking_disabled_uses_single_generate_call(
    client,
    fake_runtime,
):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs["text"])
        yield _generated_audio(2400)

    fake_runtime.model = types.SimpleNamespace(generate=generate)

    payload = _valid_payload()
    payload["text"] = "Sentence one. Sentence two. Sentence three."
    payload["chunking_enabled"] = False
    payload["chunk_min_chars"] = 10
    payload["chunk_max_chars"] = 20

    response = _post(client, payload)

    assert response.status_code == 200
    assert calls == [payload["text"]]


def test_synthesize_chunking_enabled_uses_multiple_generate_calls(
    client,
    fake_runtime,
):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs["text"])
        yield _generated_audio(2400)

    fake_runtime.model = types.SimpleNamespace(generate=generate)

    payload = _valid_payload()
    payload["text"] = (
        "This is sentence number one. "
        "This is sentence number two. "
        "This is sentence number three."
    )
    payload["chunking_enabled"] = True
    payload["chunk_min_chars"] = 20
    payload["chunk_max_chars"] = 45

    expected_chunks = srv._split_text_chunks(
        payload["text"],
        min_chars=20,
        max_chars=45,
    )

    response = _post(client, payload)

    assert response.status_code == 200
    assert calls == expected_chunks
    assert len(calls) > 1


def test_synthesize_chunking_zero_silence_adds_no_gap(
    client,
    fake_runtime,
    monkeypatch,
):
    def generate(**kwargs):
        yield _generated_audio(100)

    fake_runtime.model = types.SimpleNamespace(generate=generate)

    captured = {}

    def fake_numpy_to_wav_bytes(wav, sample_rate):
        captured["wav"] = wav.copy()
        captured["sample_rate"] = sample_rate
        return b"fake wav"

    monkeypatch.setattr(
        srv,
        "_numpy_to_wav_bytes",
        fake_numpy_to_wav_bytes,
    )

    payload = _valid_payload()
    payload["text"] = "abcdefghij klmnopqrst uvwxyzabcd efghijklmn"
    payload["chunking_enabled"] = True
    payload["chunk_min_chars"] = 10
    payload["chunk_max_chars"] = 20
    payload["chunk_silence_ms"] = 0

    expected_chunks = srv._split_text_chunks(
        payload["text"],
        min_chars=10,
        max_chars=20,
    )

    response = _post(client, payload)

    assert response.status_code == 200
    assert len(expected_chunks) > 1
    assert len(captured["wav"]) == len(expected_chunks) * 100


def test_synthesize_chunking_inserts_requested_silence(
    client,
    fake_runtime,
    monkeypatch,
):
    def generate(**kwargs):
        yield _generated_audio(100)

    fake_runtime.model = types.SimpleNamespace(generate=generate)

    captured = {}

    def fake_numpy_to_wav_bytes(wav, sample_rate):
        captured["wav"] = wav.copy()
        captured["sample_rate"] = sample_rate
        return b"fake wav"

    monkeypatch.setattr(
        srv,
        "_numpy_to_wav_bytes",
        fake_numpy_to_wav_bytes,
    )

    payload = _valid_payload()
    payload["text"] = "abcdefghij klmnopqrst uvwxyzabcd efghijklmn"
    payload["chunking_enabled"] = True
    payload["chunk_min_chars"] = 10
    payload["chunk_max_chars"] = 20
    payload["chunk_silence_ms"] = 100

    expected_chunks = srv._split_text_chunks(
        payload["text"],
        min_chars=10,
        max_chars=20,
    )

    response = _post(client, payload)

    assert response.status_code == 200
    assert len(expected_chunks) > 1

    silence_samples = int(srv.SAMPLE_RATE * payload["chunk_silence_ms"] / 1000)

    expected_samples = (
        len(expected_chunks) * 100 + (len(expected_chunks) - 1) * silence_samples
    )

    assert len(captured["wav"]) == expected_samples


def test_synthesize_chunking_crossfade_regression(
    client,
    fake_runtime,
    monkeypatch,
):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs["text"])

        if len(calls) == 1:
            # First intentional chunk: the engine returns two generator
            # sub-results, which must still be concatenated normally (never
            # crossfaded against each other).
            yield _generated_audio(100)
            yield _generated_audio(100)
        else:
            yield _generated_audio(100)

    fake_runtime.model = types.SimpleNamespace(generate=generate)

    captured = {}

    def fake_numpy_to_wav_bytes(wav, sample_rate):
        captured["wav"] = wav.copy()
        captured["sample_rate"] = sample_rate
        return b"fake wav"

    monkeypatch.setattr(
        srv,
        "_numpy_to_wav_bytes",
        fake_numpy_to_wav_bytes,
    )

    payload = _valid_payload()
    payload["text"] = "abcdefghij klmnopqrst uvwxyzabcd efghijklmn"
    payload["chunking_enabled"] = True
    payload["chunk_min_chars"] = 10
    payload["chunk_max_chars"] = 20
    payload["chunk_crossfade_ms"] = 10

    expected_chunks = srv._split_text_chunks(
        payload["text"],
        min_chars=10,
        max_chars=20,
    )

    response = _post(client, payload)

    assert response.status_code == 200
    assert len(expected_chunks) > 1

    # The model is invoked exactly once per intentional text chunk,
    # regardless of how many generator sub-results the first call yields.
    assert calls == expected_chunks

    # The first intentional chunk's two sub-results (100 samples each) are
    # concatenated normally to 200 samples before any crossfade logic runs;
    # every subsequent intentional chunk is 100 samples. With a 10 ms
    # crossfade at 24 kHz the requested overlap (240) is clamped to the
    # shorter side of each join (100 samples), so every join after the first
    # keeps the running length constant at 200. If the two generator
    # sub-results had incorrectly been crossfaded against each other instead
    # of concatenated, the first chunk would collapse to 100 samples and the
    # final length would be 100 instead of 200.
    assert len(captured["wav"]) == 200


def test_synthesize_crossfade_with_chunking_disabled_is_noop(
    client,
    fake_runtime,
    monkeypatch,
):
    def generate(**kwargs):
        yield _generated_audio(100)

    fake_runtime.model = types.SimpleNamespace(generate=generate)

    captured = {}

    def fake_numpy_to_wav_bytes(wav, sample_rate):
        captured["wav"] = wav.copy()
        captured["sample_rate"] = sample_rate
        return b"fake wav"

    monkeypatch.setattr(
        srv,
        "_numpy_to_wav_bytes",
        fake_numpy_to_wav_bytes,
    )

    payload = _valid_payload()
    payload["text"] = "Sentence one. Sentence two. Sentence three."
    payload["chunking_enabled"] = False
    payload["chunk_crossfade_ms"] = 10

    response = _post(client, payload)

    # Chunking disabled means exactly one intentional text chunk, so the
    # crossfade path (which only runs for >1 chunks) never engages even
    # though chunk_crossfade_ms > 0.
    assert response.status_code == 200
    assert len(captured["wav"]) == 100


def test_synthesize_empty_middle_chunk_returns_500(
    client,
    fake_runtime,
):
    calls = 0

    def generate(**kwargs):
        nonlocal calls
        calls += 1

        if calls == 2:
            return iter(())

        return iter([_generated_audio(100)])

    fake_runtime.model = types.SimpleNamespace(generate=generate)

    payload = _valid_payload()
    payload["text"] = "abcdefghij klmnopqrst uvwxyzabcd efghijklmn"
    payload["chunking_enabled"] = True
    payload["chunk_min_chars"] = 10
    payload["chunk_max_chars"] = 20

    response = _post(client, payload)

    assert response.status_code == 500
    assert "produced no audio" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# POST /synthesize — reference-audio pre-flight (400)
# ---------------------------------------------------------------------------


def test_synthesize_undecodable_audio_rejected(client, fake_runtime):
    payload = _valid_payload()
    payload["audio_base64"] = b64(b"this is definitely not audio")

    response = _post(client, payload)

    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_synthesize_too_short_audio_rejected(client, fake_runtime):
    payload = _valid_payload()
    payload["audio_base64"] = b64(make_wav_bytes(0.5))  # 0.5 s < 2.0 s minimum

    response = _post(client, payload)

    assert response.status_code == 400
    assert "2" in response.json()["detail"]
