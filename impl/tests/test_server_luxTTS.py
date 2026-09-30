"""Tests for ``server_luxTTS.py``.

These exercise the HTTP surface that does NOT require a loaded model:
/capabilities (snapshot), /health, the landing page, request-body validation
(422s), and the reference-audio pre-flight checks (400s).  Synthesis itself
needs a real model + GPU, so it is intentionally out of scope here; the
prompt cache around it is exercised with a recording fake model.
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
# Prompt cache — the encoded reference is reused across requests
# ---------------------------------------------------------------------------


def test_prompt_cache_key_depends_on_clip_duration_and_rms():
    clip, other = make_wav_bytes(3.0), make_wav_bytes(3.0, freq=220.0)
    base = srv.PromptCache.key(clip, 5.0, 0.001)
    assert srv.PromptCache.key(clip, 5.0, 0.001) == base
    assert srv.PromptCache.key(other, 5.0, 0.001) != base
    assert srv.PromptCache.key(clip, 10.0, 0.001) != base
    assert srv.PromptCache.key(clip, 5.0, 0.01) != base


def test_prompt_cache_evicts_least_recently_used():
    cache = srv.PromptCache(2)
    cache.put("a", {"n": 1})
    cache.put("b", {"n": 2})
    assert cache.get("a") == {"n": 1}  # touching "a" makes "b" the oldest
    cache.put("c", {"n": 3})
    assert cache.get("b") is None
    assert cache.get("a") == {"n": 1}
    assert cache.get("c") == {"n": 3}
    assert len(cache) == 2


def test_prompt_cache_size_zero_disables_caching():
    cache = srv.PromptCache(0)
    cache.put("a", {"n": 1})
    assert cache.get("a") is None
    assert len(cache) == 0


def test_prompt_cache_size_env_default_is_eight():
    assert srv.PROMPT_CACHE_SIZE == 8


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
    """Counts encode_prompt() calls; hands generate_speech() back a silent clip."""

    def __init__(self):
        self.encode_calls: list[dict] = []
        self.generated_with: list[dict] = []

    def encode_prompt(self, path, duration, rms):
        self.encode_calls.append({"path": path, "duration": duration, "rms": rms})
        return {"prompt_tokens": [[0]], "prompt_features_lens": [1.0],
                "prompt_features": None, "prompt_rms": rms, "id": len(self.encode_calls)}

    def generate_speech(self, _text, encode_dict, **_kwargs):
        self.generated_with.append(encode_dict)
        return _FakeWav()


@pytest.fixture
def fake_model(monkeypatch):
    """A recording model, a fresh cache per test, and a no-op WAV encoder (the
    stub numpy/soundfile refuse clip()/write() by design)."""
    model = _FakeLuxModel()
    monkeypatch.setattr(
        srv, "_runtime",
        types.SimpleNamespace(model=model, sample_rate=srv.SAMPLE_RATE, device=srv.DEVICE),
    )
    monkeypatch.setattr(srv, "_prompt_cache", srv.PromptCache(8))
    monkeypatch.setattr(srv, "_numpy_to_wav_bytes", lambda arr, sr: b"RIFFfake")
    return model


def test_same_reference_is_encoded_once(client, fake_model):
    payload = _valid_payload()
    for text in ("First sentence.", "Second sentence.", "Third."):
        payload["text"] = text
        assert _post(client, payload).status_code == 200
    assert len(fake_model.encode_calls) == 1
    # Every generation reused that one encoding.
    assert [e["id"] for e in fake_model.generated_with] == [1, 1, 1]


def test_different_reference_is_encoded_again(client, fake_model):
    payload = _valid_payload()
    assert _post(client, payload).status_code == 200
    payload["audio_base64"] = b64(make_wav_bytes(3.0, freq=220.0))
    assert _post(client, payload).status_code == 200
    assert len(fake_model.encode_calls) == 2


@pytest.mark.parametrize("field,value", [("prompt_duration", 10.0), ("prompt_rms", 0.01)])
def test_changed_prompt_setting_is_encoded_again(client, fake_model, field, value):
    payload = _valid_payload()
    assert _post(client, payload).status_code == 200
    payload[field] = value
    assert _post(client, payload).status_code == 200
    assert len(fake_model.encode_calls) == 2
    assert fake_model.encode_calls[-1]["duration" if field == "prompt_duration" else "rms"] == value


def test_cache_hit_writes_no_temp_file(client, fake_model, monkeypatch):
    written = []
    real_write = srv.write_temp_audio
    monkeypatch.setattr(
        srv, "write_temp_audio", lambda raw, d: written.append(1) or real_write(raw, d)
    )
    payload = _valid_payload()
    for _ in range(3):
        assert _post(client, payload).status_code == 200
    assert len(written) == 1


def test_cache_disabled_encodes_every_request(client, fake_model, monkeypatch):
    monkeypatch.setattr(srv, "_prompt_cache", srv.PromptCache(0))
    payload = _valid_payload()
    for _ in range(3):
        assert _post(client, payload).status_code == 200
    assert len(fake_model.encode_calls) == 3


def test_seed_is_applied_after_encoding(client, fake_model, monkeypatch):
    # Seeding after encode_prompt() keeps cache hits and misses identical for
    # the same seed: the seed only governs the solver's noise.
    order = []
    real_encode = fake_model.encode_prompt
    monkeypatch.setattr(fake_model, "encode_prompt",
                        lambda *a, **k: order.append("encode") or real_encode(*a, **k))
    monkeypatch.setattr(srv, "seed_everything", lambda seed: order.append("seed"))
    payload = _valid_payload()
    payload["seed"] = 42
    assert _post(client, payload).status_code == 200
    assert _post(client, payload).status_code == 200
    assert order == ["encode", "seed", "seed"]
