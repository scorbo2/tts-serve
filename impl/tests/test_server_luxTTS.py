"""Tests for ``server_luxTTS.py``.

These exercise the HTTP surface that does NOT require a loaded model:
/capabilities (snapshot), /health, the landing page, request-body validation
(422s), and the reference-audio pre-flight checks (400s).  Synthesis itself
needs a real model + GPU, so it is intentionally out of scope here.
"""

import types

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
# POST /synthesize — reference-audio pre-flight (400).  The handler runs up to
# the audio check, so the (stubbed) runtime is faked out to skip model load.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setattr(
        srv, "_runtime", types.SimpleNamespace(sample_rate=srv.SAMPLE_RATE, device=srv.DEVICE)
    )


def test_synthesize_undecodable_audio_rejected(client, fake_runtime):
    payload = _valid_payload()
    payload["audio_base64"] = b64(b"this is definitely not audio")
    response = _post(client, payload)
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_synthesize_too_short_audio_rejected(client, fake_runtime):
    payload = _valid_payload()
    payload["audio_base64"] = b64(make_wav_bytes(0.5))  # 0.5 s < 3.0 s minimum
    response = _post(client, payload)
    assert response.status_code == 400
    assert "3" in response.json()["detail"]


# ---------------------------------------------------------------------------
# tail_padding — extra time on the engine's length estimate (short sentences)
# ---------------------------------------------------------------------------


def test_synthesize_tail_padding_below_range_rejected(client):
    payload = _valid_payload()
    payload["tail_padding"] = -0.1
    assert _post(client, payload).status_code == 422


def test_synthesize_tail_padding_above_range_rejected(client):
    payload = _valid_payload()
    payload["tail_padding"] = 2.5
    assert _post(client, payload).status_code == 422


def _engine_text_frames(prompt_frames, prompt_tokens, text_tokens, passed_speed):
    """The engine's own length estimate for the text part (zipvoice ratio
    duration), including its internal speed factor."""
    return prompt_frames / prompt_tokens * text_tokens / (passed_speed * srv.ENGINE_SPEED_FACTOR)


@pytest.mark.parametrize("text_tokens", [3, 12, 150])
@pytest.mark.parametrize("speed", [0.8, 1.0, 1.5])
def test_padded_speed_adds_exactly_the_padding(text_tokens, speed):
    prompt_frames, prompt_tokens, pad = 940.0, 110, 28.0
    padded = srv.padded_speed(prompt_frames, prompt_tokens, text_tokens, speed, pad)
    before = _engine_text_frames(prompt_frames, prompt_tokens, text_tokens, speed)
    after = _engine_text_frames(prompt_frames, prompt_tokens, text_tokens, padded)
    assert after == pytest.approx(before + pad)
    assert padded < speed


def test_padded_speed_matters_most_for_short_texts():
    # The same padding is a large share of a one-word sentence and a small
    # share of a long one, so short texts are slowed down far more.
    short = srv.padded_speed(940.0, 110, 3, 1.0, 28.0)
    long = srv.padded_speed(940.0, 110, 150, 1.0, 28.0)
    assert short < long < 1.0
    assert long > 0.9


@pytest.mark.parametrize(
    "args",
    [
        (940.0, 110, 12, 1.0, 0.0),  # no padding requested
        (940.0, 110, 0, 1.0, 28.0),  # empty text tokens
        (940.0, 0, 12, 1.0, 28.0),  # empty prompt tokens
        (0.0, 110, 12, 1.0, 28.0),  # empty prompt features
    ],
)
def test_padded_speed_degenerate_inputs_keep_speed(args):
    assert srv.padded_speed(*args) == args[3]


class _FakeWav:
    """Stands in for the engine's (1, N) tensor: detach().cpu().numpy().reshape()."""

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self

    def reshape(self, *_shape):
        return [0.0] * (srv.SAMPLE_RATE // 10)  # ~0.1 s of silence


class _FakeLuxModel:
    """Engine-shaped encode_prompt()/tokenizer; records generate_speech() kwargs."""

    PROMPT_FRAMES = 940.0
    PROMPT_TOKENS = 110

    def __init__(self):
        self.calls: list[dict] = []
        # One token per character: enough to make token counts predictable.
        self.tokenizer = types.SimpleNamespace(
            texts_to_token_ids=lambda texts: [[0] * len(texts[0])]
        )

    def encode_prompt(self, _path, duration, rms):
        return {
            "prompt_tokens": [[0] * self.PROMPT_TOKENS],
            "prompt_features_lens": [self.PROMPT_FRAMES],
            "prompt_features": None,
            "prompt_rms": rms,
        }

    def generate_speech(self, _text, _encode_dict, **kwargs):
        self.calls.append(kwargs)
        return _FakeWav()


@pytest.fixture
def fake_model(monkeypatch):
    """Install a recording model and a no-op WAV encoder (the stub
    numpy/soundfile refuse clip()/write() by design)."""
    model = _FakeLuxModel()
    monkeypatch.setattr(
        srv,
        "_runtime",
        srv.LuxTTSRuntime(model=model, sample_rate=srv.SAMPLE_RATE, device=srv.DEVICE),
    )
    monkeypatch.setattr(srv, "_numpy_to_wav_bytes", lambda arr, sr: b"RIFFfake")
    return model


def test_synthesize_forwards_padded_speed(client, fake_model):
    payload = _valid_payload()
    payload.update(text="Indeed.", speed=1.0, tail_padding=0.3)
    assert _post(client, payload).status_code == 200
    expected = srv.padded_speed(
        _FakeLuxModel.PROMPT_FRAMES,
        _FakeLuxModel.PROMPT_TOKENS,
        len("Indeed."),
        1.0,
        0.3 / srv.DEFAULT_FRAME_SHIFT_S,
    )
    assert fake_model.calls[-1]["speed"] == pytest.approx(expected)
    assert fake_model.calls[-1]["speed"] < 1.0


def test_synthesize_zero_tail_padding_forwards_speed_unchanged(client, fake_model):
    payload = _valid_payload()
    payload.update(speed=1.2, tail_padding=0.0)
    assert _post(client, payload).status_code == 200
    assert fake_model.calls[-1]["speed"] == 1.2


def test_synthesize_tail_padding_defaults_on(client, fake_model):
    payload = _valid_payload()  # no tail_padding: the 0.3 s default applies
    assert _post(client, payload).status_code == 200
    assert fake_model.calls[-1]["speed"] < 1.0
