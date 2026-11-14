"""Tests for ``server_luxTTS.py``.

These exercise the HTTP surface that does NOT require a loaded model:
/capabilities (snapshot), /health, the landing page, request-body validation
(422s), and the reference-audio pre-flight checks (400s).  Synthesis itself
needs a real model + GPU, so it is intentionally out of scope here.
"""

import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server_luxTTS as srv
from helpers import b64, load_snapshot, make_wav_bytes


@pytest.fixture(scope="module")
def client():
    # Deliberately NOT used as a context manager: entering it would run the
    # FastAPI lifespan, which loads the (stubbed) model.  The endpoints under
    # test here don't need the model, so we skip lifespan entirely.
    return TestClient(srv.app)


# ---------------------------------------------------------------------------
# GET /capabilities — snapshot (single source of truth: the Pydantic model)
# ---------------------------------------------------------------------------


def test_capabilities_matches_snapshot(client):
    response = client.get("/capabilities")
    assert response.status_code == 200
    doc = response.json()
    assert doc == load_snapshot("lux_tts_capabilities.json")


def test_capabilities_core_fields_are_required(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    # The two fields the client must always supply.
    assert by_name["text"]["required"] is True
    assert by_name["audio_base64"]["required"] is True
    # The engine always transcribes the reference itself — no transcript
    # field exists (Chatterbox-style deliberate omission).
    assert "reference_text" not in by_name
    # Optional tuning knobs default to the model's own defaults.
    assert by_name["seed"]["required"] is False
    assert by_name["num_steps"]["default"] == 4
    assert by_name["t_shift"]["default"] == 0.5
    assert by_name["prompt_rms"]["default"] == 0.001


def test_capabilities_language_has_no_enum_and_no_advertised_list(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    # The engine has no language parameter: any two-letter code is accepted
    # (never forwarded), so there is neither a request enum nor an
    # advertised language list.
    assert by_name["language"]["enum"] is None
    assert doc["languages"] is None


def test_capabilities_sample_rate_is_48k(client):
    doc = client.get("/capabilities").json()
    assert doc["sample_rate"] == srv.SAMPLE_RATE == 48000


# ---------------------------------------------------------------------------
# GET /health and the landing page
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["serverType"] == "LuxTTS"
    assert body["device"] == srv.DEVICE
    assert body["model"] == srv.MODEL_NAME_OR_PATH
    # The tts-serve release version comes from the single shared constant;
    # pinning to srv.RELEASE_VERSION (not a literal) keeps this release-proof
    # while still catching a server that forgot to wire it in.
    assert body["version"] == srv.RELEASE_VERSION


def test_root_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "LuxTTS" in response.text
    assert "/synthesize" in response.text


# ---------------------------------------------------------------------------
# POST /synthesize — request-body validation (422).  Validation happens before
# the handler runs, so no model is involved.
# ---------------------------------------------------------------------------


def _post(client, payload):
    return client.post("/synthesize", json=payload)


def _valid_payload():
    return {
        "text": "Hello there",
        "audio_base64": b64(make_wav_bytes(3.0)),
    }


def test_synthesize_unknown_field_rejected(client):
    payload = _valid_payload()
    payload["bogus_field"] = 1
    assert _post(client, payload).status_code == 422


def test_synthesize_missing_required_fields_rejected(client):
    assert _post(client, {}).status_code == 422


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


# The shared docs/02 language contract (case, names, garbage, non-strings,
# null/empty -> 'en', and well-formed codes passing) is asserted once for
# every server in test_language_contract.py; keep only engine-specific
# language tests here.  LuxTTS accepts *any* well-formed two-letter code
# (it is never forwarded), so there is no "unsupported code" 422 to test.


def test_synthesize_seed_below_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_seed_above_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 1001
    assert _post(client, payload).status_code == 422


def test_synthesize_num_steps_below_range_rejected(client):
    payload = _valid_payload()
    payload["num_steps"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_num_steps_above_range_rejected(client):
    payload = _valid_payload()
    payload["num_steps"] = 33
    assert _post(client, payload).status_code == 422


def test_synthesize_guidance_scale_above_range_rejected(client):
    payload = _valid_payload()
    payload["guidance_scale"] = 10.5
    assert _post(client, payload).status_code == 422


def test_synthesize_t_shift_zero_rejected(client):
    # t_shift=0 degenerates the solver's timestep schedule (0/0 at the final
    # step) — the bound is exclusive on purpose.
    payload = _valid_payload()
    payload["t_shift"] = 0.0
    assert _post(client, payload).status_code == 422


def test_synthesize_t_shift_above_range_rejected(client):
    payload = _valid_payload()
    payload["t_shift"] = 3.5
    assert _post(client, payload).status_code == 422


def test_synthesize_speed_below_range_rejected(client):
    payload = _valid_payload()
    payload["speed"] = 0.05
    assert _post(client, payload).status_code == 422


def test_synthesize_return_smooth_nonBoolean_rejected(client):
    # Pydantic v2 lax mode coerces "yes"/"true"/"1" — use a value it cannot.
    payload = _valid_payload()
    payload["return_smooth"] = "definitely not a bool"
    assert _post(client, payload).status_code == 422


def test_synthesize_prompt_duration_below_range_rejected(client):
    payload = _valid_payload()
    payload["prompt_duration"] = 0.5
    assert _post(client, payload).status_code == 422


def test_synthesize_prompt_duration_above_range_rejected(client):
    payload = _valid_payload()
    payload["prompt_duration"] = 1001.0
    assert _post(client, payload).status_code == 422


def test_synthesize_prompt_rms_above_range_rejected(client):
    payload = _valid_payload()
    payload["prompt_rms"] = 1.5
    assert _post(client, payload).status_code == 422


# ---------------------------------------------------------------------------
# POST /synthesize — the fake engine.
#
# synthesize() calls _get_runtime() before it touches the audio, so every
# handler test needs *some* runtime installed, even the ones that only exercise
# the audio pre-flights.  One fixture, one fake engine.
#
# The fake records what the server asked the engine to do, because a `**kwargs`
# signature silently absorbs a dropped `speed=` — pinning the forwarding is the
# reason a fake-model suite exists at all (AGENTS.md: "the /synthesize success
# path is exercised where a test installs a fake model").
#
# What the *tokenizer* decides is engine behaviour, measured on a real box
# against YatharthS/LuxTTS (see the "nothing to speak" section of
# impl/server_luxTTS.md): '1234', '....', '```', '[laughs]', emoji, and any
# non-Latin/non-Chinese script (Cyrillic, Japanese, Korean, Greek, ...) all
# yield zero tokens, while 'café', 'über groß' and CJK do.  What is pinned
# *here* is the server's contract: zero tokens is a client error the caller can
# act on, never a 500 from inside the vocoder, and the engine is never asked to
# speak an empty token list.
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """``EmiliaTokenizer`` stand-in: one fixed token-id list for every text.

    ``tokens=[]`` reproduces the engine dropping a whole text (digits,
    punctuation, emoji, non-Latin/non-Chinese scripts).
    """

    def __init__(self, tokens=(20, 4, 11)):
        self.tokens = list(tokens)
        self.calls: list[list[str]] = []

    def texts_to_token_ids(self, texts):
        self.calls.append(list(texts))
        return [list(self.tokens) for _ in texts]


class _FakeAudioArray:
    """Stands in for the numpy array the server flattens, encodes and counts."""

    def reshape(self, shape):
        # The server flattens to 1-D; the fake is already flat.
        return [0.0, 0.05, -0.05]


class _FakeWav:
    """Stands in for the (1, N) float tensor ``generate_speech`` returns."""

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return _FakeAudioArray()


class _FakeEngine:
    """Stands in for the loaded ``zipvoice.luxvoice.LuxTTS`` object.

    Only the three members the server touches: the tokenizer (pre-flight),
    ``encode_prompt`` (Whisper + features, returns the encode dict) and
    ``generate_speech``.  Call arguments are recorded for the assertions.
    """

    def __init__(self, *, tokens=(20, 4, 11), prompt_tokens=((20, 4),)):
        self.tokenizer = _FakeTokenizer(tokens=tokens)
        self.prompt_tokens = [[*row] for row in prompt_tokens]
        self.encode_calls = 0
        self.encode_args = None
        self.generate_calls = 0
        self.generate_args = None
        self.generate_kwargs = None

    def encode_prompt(self, prompt_audio_path, duration, rms):
        self.encode_calls += 1
        self.encode_args = {
            "prompt_audio_path": prompt_audio_path,
            "duration": duration,
            "rms": rms,
        }
        return {
            "prompt_tokens": self.prompt_tokens,
            "prompt_features_lens": [7],
            "prompt_features": None,
            "prompt_rms": rms,
        }

    def generate_speech(self, text, encode_dict, **kwargs):
        self.generate_calls += 1
        self.generate_args = {"text": text, "encode_dict": encode_dict}
        self.generate_kwargs = kwargs
        return _FakeWav()


@pytest.fixture
def fake_engine(monkeypatch):
    """Install a fake loaded runtime; hands back the engine for assertions."""
    engine = _FakeEngine()
    monkeypatch.setattr(
        srv,
        "_runtime",
        types.SimpleNamespace(
            model=engine, sample_rate=srv.SAMPLE_RATE, device=srv.DEVICE
        ),
    )
    # numpy/soundfile are stubs on a GPU-free box: skip the WAV encoding.
    monkeypatch.setattr(srv, "_numpy_to_wav_bytes", lambda arr, sr: b"RIFFfake")
    return engine


# ---------------------------------------------------------------------------
# POST /synthesize — reference-audio pre-flight (400).  The handler runs up to
# the audio check; the fake runtime is only there to skip the model load.
# ---------------------------------------------------------------------------


def test_synthesize_undecodable_audio_rejected(client, fake_engine):
    payload = _valid_payload()
    payload["audio_base64"] = b64(b"this is definitely not audio")
    response = _post(client, payload)
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_synthesize_too_short_audio_rejected(client, fake_engine):
    payload = _valid_payload()
    payload["audio_base64"] = b64(make_wav_bytes(0.5))  # 0.5 s < 3.0 s minimum
    response = _post(client, payload)
    assert response.status_code == 400
    assert "3" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /synthesize — the text pre-flight (422).
# ---------------------------------------------------------------------------


def test_synthesize_text_with_no_tokens_returns_422_before_encoding_the_prompt(
    client, fake_engine
):
    # GIVEN a tokenizer that drops the whole text (digits/punctuation/emoji/
    # other scripts), the case where the vocoder aborts on zero frames.
    fake_engine.tokenizer.tokens = []

    # WHEN the request is posted,
    response = _post(client, _valid_payload())

    # THEN it is a client error naming the engine's script coverage...
    assert response.status_code == 422
    detail = response.json()["detail"].lower()
    assert "nothing this engine can speak" in detail
    assert "chinese" in detail
    assert "japanese" in detail
    # ...and the engine is asked for nothing at all: no Whisper/feature pass
    # (the expensive part), no synthesis.  Pinning encode_calls also pins the
    # ordering — the cheap text check runs before the ASR pass, not after.
    assert fake_engine.tokenizer.calls == [["Hello there"]]
    assert fake_engine.encode_calls == 0
    assert fake_engine.generate_calls == 0


def test_synthesize_text_with_tokens_is_synthesized(client, fake_engine):
    # GIVEN a tokenizer that finds tokens in the text (fixture default),
    # WHEN the payload is posted,
    response = _post(client, _valid_payload())

    # THEN the guard lets the request through (200 with audio).
    assert response.status_code == 200
    assert response.json()["audio_base64"]
    assert fake_engine.generate_calls == 1


def test_synthesize_text_whose_only_token_is_the_pad_token_is_not_rejected(
    client, fake_engine
):
    # GIVEN a text that tokenizes to exactly one token, id 0 — in tokens.txt
    # that id is the pad token '_'.  "Has tokens" is a *count* question: id 0
    # is falsy, so a truthiness test would reject speakable text.
    fake_engine.tokenizer.tokens = [0]

    response = _post(client, _valid_payload())

    assert response.status_code == 200
    assert fake_engine.generate_calls == 1


def test_synthesize_tokenizer_raising_lets_the_engine_handle_the_text(
    client, fake_engine
):
    # A pre-flight must not invent an error the engine would not have
    # reported: if the tokenizer blows up, the text goes to the engine anyway.
    def _boom(texts):
        raise RuntimeError("tokenizer exploded")

    fake_engine.tokenizer.texts_to_token_ids = _boom

    response = _post(client, _valid_payload())

    assert response.status_code == 200
    assert fake_engine.generate_calls == 1


def test_synthesize_generate_speech_receives_every_tuning_parameter_of_the_request(
    client, fake_engine
):
    payload = _valid_payload()
    payload.update(
        {
            "num_steps": 7,
            "guidance_scale": 4.5,
            "t_shift": 1.25,
            "speed": 0.8,
            "return_smooth": True,
            "prompt_duration": 8.0,
            "prompt_rms": 0.01,
        }
    )

    response = _post(client, payload)

    # THEN encode_prompt got the request's conditioning window...
    assert response.status_code == 200
    assert fake_engine.encode_args["duration"] == 8.0
    assert fake_engine.encode_args["rms"] == 0.01
    # ...and generate_speech got the text and every knob.  The exact-dict
    # assertion is the point: `**kwargs` absorbs a renamed or dropped one.
    assert fake_engine.generate_args["text"] == "Hello there"
    assert fake_engine.generate_args["encode_dict"] is not None
    assert fake_engine.generate_kwargs == {
        "num_steps": 7,
        "guidance_scale": 4.5,
        "t_shift": 1.25,
        "speed": 0.8,
        "return_smooth": True,
    }


def test_synthesize_422_from_the_text_preflight_cleans_up_the_staged_reference_file(
    client, fake_engine, monkeypatch
):
    # GIVEN unspeakable text, plus a recorder wrapped around the real cleanup
    # that runs in synthesize()'s finally block.
    fake_engine.tokenizer.tokens = []
    cleaned: list[str] = []
    real_cleanup = srv.cleanup_temp

    def record_then_cleanup(path):
        cleaned.append(str(path))
        real_cleanup(path)

    monkeypatch.setattr(srv, "cleanup_temp", record_then_cleanup)

    response = _post(client, _valid_payload())

    # THEN the mid-request exit does not leak the staged reference clip.
    assert response.status_code == 422
    assert len(cleaned) == 1
    assert not Path(cleaned[0]).exists()


# ---------------------------------------------------------------------------
# POST /synthesize — the reference-clip pre-flight (400).
#
# The engine conditions on Whisper's transcript of the clip's first
# prompt_duration seconds; an empty transcript makes the duration prediction
# divide by the (zero) prompt-token count.  Only encode_prompt can discover
# that, so these tests pin "400 after the ASR pass, before synthesis".
# ---------------------------------------------------------------------------


def test_synthesize_reference_clip_with_empty_prompt_tokens_returns_400_after_encoding(
    client, fake_engine
):
    # GIVEN a clip whose transcript holds nothing the tokenizer keeps (silence,
    # a clip in another language) — prompt_tokens is [[]].
    fake_engine.prompt_tokens = [[]]

    # WHEN a perfectly speakable text is posted,
    response = _post(client, _valid_payload())

    # THEN the caller is told the *clip* is the problem (400, not 500)...
    assert response.status_code == 400
    detail = response.json()["detail"].lower()
    assert "reference audio contains no recognizable speech" in detail
    # ...after the ASR pass that discovers it, and before synthesis.
    assert fake_engine.encode_calls == 1
    assert fake_engine.generate_calls == 0


def test_synthesize_empty_prompt_tokens_400_names_the_prompt_duration_window(
    client, fake_engine
):
    # GIVEN the same untranscribable clip, and a request that narrows the
    # conditioning window to 2.5 s (only that window reaches Whisper).
    fake_engine.prompt_tokens = [[]]
    payload = _valid_payload()
    payload["prompt_duration"] = 2.5

    response = _post(client, payload)

    # THEN the 400 names the window the client can act on, not the clip's
    # total length — advising "a ~10 s clip works best" while the request
    # transcribes 2.5 s of it sends the user off to re-record something that
    # fails identically.
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "first 2.5 s" in detail
    assert "prompt_duration" in detail


def test_synthesize_reference_clip_with_prompt_tokens_reaches_generate_speech(
    client, fake_engine
):
    fake_engine.prompt_tokens = [[20, 4, 11]]

    response = _post(client, _valid_payload())

    assert response.status_code == 200
    assert fake_engine.encode_calls == 1
    assert fake_engine.generate_calls == 1


def test_synthesize_reference_clip_with_only_the_pad_prompt_token_is_not_rejected(
    client, fake_engine
):
    # Same reason as the text guard: id 0 is the pad token, emptiness is about
    # counts, and a 400 here would reject a cloneable clip.
    fake_engine.prompt_tokens = [[0]]

    response = _post(client, _valid_payload())

    assert response.status_code == 200
    assert fake_engine.generate_calls == 1


def test_synthesize_encode_prompt_with_tensor_shaped_prompt_tokens_is_left_alone(
    client, fake_engine
):
    # GIVEN an engine that returns prompt_tokens as a tensor (every *other*
    # entry of the encode dict already is one).  The guard's shape assumption
    # is the engine's to break, and a pre-flight must not become the new 500.
    class _TensorLike:
        def __iter__(self):
            raise RuntimeError("Boolean value of Tensor is ambiguous")

    fake_engine.prompt_tokens = _TensorLike()

    response = _post(client, _valid_payload())

    assert response.status_code == 200
    assert fake_engine.generate_calls == 1


def test_synthesize_encode_prompt_returning_a_tuple_is_left_alone(client, fake_engine):
    # The dict-shape check is a deliberate no-op; pin it as fail-open rather
    # than an accidental 500 on an engine that swaps its encode dict.
    def _tuple_encode(prompt_audio_path, duration, rms):
        fake_engine.encode_calls += 1
        return (fake_engine.prompt_tokens, [7], None, rms)

    fake_engine.encode_prompt = _tuple_encode

    response = _post(client, _valid_payload())

    assert response.status_code == 200
    assert fake_engine.encode_calls == 1
    assert fake_engine.generate_calls == 1
