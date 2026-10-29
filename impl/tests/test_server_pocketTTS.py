"""Tests for ``server_pocketTTS.py``.

These exercise the HTTP surface that does NOT require a real model:
/capabilities (snapshot), /health, the landing page, request-body validation
(422s) and the reference-audio pre-flight checks (400s).  The synthesis path
is additionally exercised against a fake model (installed over the import-only
stub) to pin voice-source selection (clip > preset > default), seed handling,
voice-state caching and the response shape.  Real synthesis needs the model
weights and is out of scope here.
"""

import os
import subprocess
import sys
import types

import pytest
from fastapi.testclient import TestClient

import _bootstrap
import server_pocketTTS as srv
from helpers import b64, load_snapshot, make_wav_bytes


@pytest.fixture(scope="module")
def client():
    # Deliberately NOT used as a context manager: entering it would run the
    # FastAPI lifespan, which loads the (stubbed) model.
    return TestClient(srv.app)


def _post(client, payload):
    return client.post("/synthesize", json=payload)


def _payload(**extra):
    return {"text": "Hello there", **extra}


def _clip_payload(**extra):
    return _payload(audio_base64=b64(make_wav_bytes(3.0)), **extra)


# ---------------------------------------------------------------------------
# GET /capabilities -- snapshot (single source of truth: the Pydantic model)
# ---------------------------------------------------------------------------


def test_capabilities_matches_snapshot(client):
    response = client.get("/capabilities")
    assert response.status_code == 200
    assert response.json() == load_snapshot("pocket_tts_capabilities.json")


def test_capabilities_voice_sources_are_optional(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    assert by_name["text"]["required"] is True
    # A clip is optional: presets and the language default need none.
    assert by_name["audio_base64"]["required"] is False
    assert by_name["voice"]["required"] is False
    assert doc["reference_audio"]["required"] is False
    # The engine takes no transcript, so the field is deliberately absent.
    assert "reference_text" not in by_name


def test_capabilities_voice_enum_is_the_engines_preset_table(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    assert by_name["voice"]["enum"] == list(srv._ORIGINS_OF_PREDEFINED_VOICES)


def test_capabilities_advertise_the_one_loaded_language(client):
    doc = client.get("/capabilities").json()
    assert doc["languages"] == [srv.LANGUAGE_CODE] == ["en"]
    assert doc["device"] == "cpu"
    assert doc["sample_rate"] == 24000


# ---------------------------------------------------------------------------
# GET /health and the landing page
# ---------------------------------------------------------------------------


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["serverType"] == "PocketTTS"
    assert body["device"] == "cpu"
    assert body["model"] == srv.MODEL_LABEL


def test_root_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Pocket TTS" in response.text
    assert "/synthesize" in response.text


# ---------------------------------------------------------------------------
# POST /synthesize -- request-body validation (422)
# ---------------------------------------------------------------------------


def test_unknown_field_rejected(client):
    assert _post(client, _payload(bogus_field=1)).status_code == 422


def test_reference_text_is_not_a_field(client):
    # Deliberately absent (the engine has no use for a transcript) -> 422,
    # not silently ignored.
    assert _post(client, _clip_payload(reference_text="hi")).status_code == 422


def test_missing_text_rejected(client):
    assert _post(client, {}).status_code == 422


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_or_whitespace_text_rejected(client, text):
    assert _post(client, {"text": text}).status_code == 422


def test_empty_audio_rejected(client):
    assert _post(client, _payload(audio_base64="")).status_code == 422


def test_unknown_preset_voice_rejected(client):
    assert _post(client, _payload(voice="not_a_voice")).status_code == 422


def test_clip_and_preset_together_rejected(client):
    assert _post(client, _clip_payload(voice="alba")).status_code == 422


@pytest.mark.parametrize("seed", [0, -1, 1001])
def test_seed_out_of_range_rejected(client, seed):
    assert _post(client, _payload(seed=seed)).status_code == 422


# ---------------------------------------------------------------------------
# POST /synthesize -- pre-flight (400)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setattr(
        srv,
        "_runtime",
        types.SimpleNamespace(
            sample_rate=24000, device="cpu", has_voice_cloning=True
        ),
    )


def test_undecodable_audio_rejected(client, fake_runtime):
    response = _post(client, _payload(audio_base64=b64(b"definitely not audio")))
    assert response.status_code == 400
    assert "Could not decode" in response.json()["detail"]


def test_too_short_audio_rejected(client, fake_runtime):
    response = _post(client, _payload(audio_base64=b64(make_wav_bytes(0.3))))
    assert response.status_code == 400
    assert "at least" in response.json()["detail"]


def test_cloning_unavailable_without_gated_weights(client, monkeypatch):
    monkeypatch.setattr(
        srv,
        "_runtime",
        types.SimpleNamespace(
            sample_rate=24000, device="cpu", has_voice_cloning=False
        ),
    )
    response = _post(client, _clip_payload())
    assert response.status_code == 400
    assert "hf auth login" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /synthesize -- with a fake model installed
# ---------------------------------------------------------------------------


class _FakeAudio:
    """Quacks like the torch tensor chain the handler flattens."""

    def __init__(self, n=24000):
        self._n = n

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self

    def reshape(self, *_):
        return [0.0] * self._n


class _FakeModel:
    def __init__(self):
        self.states_built = []  # what get_state_for_audio_prompt was asked for
        self.generated = []  # (state, text, copy_state)

    def get_state_for_audio_prompt(self, source, truncate=False):
        self.states_built.append((str(source), truncate))
        return f"state<{len(self.states_built)}>"

    def generate_audio(self, state, text, copy_state=True):
        self.generated.append((state, text, copy_state))
        return _FakeAudio()


@pytest.fixture
def fake_model(monkeypatch):
    model = _FakeModel()
    seeds = []
    monkeypatch.setattr(
        srv,
        "_runtime",
        srv.PocketRuntime(
            model=model, sample_rate=24000, device="cpu", has_voice_cloning=True
        ),
    )
    monkeypatch.setattr(srv, "seed_everything", seeds.append)
    monkeypatch.setattr(srv, "_numpy_to_wav_bytes", lambda a, sr: b"RIFF-fake-wav")
    srv._voice_states.clear()
    model.seeds = seeds
    yield model
    srv._voice_states.clear()


def test_synthesize_default_voice_is_the_languages_default(client, fake_model):
    response = _post(client, _payload())
    assert response.status_code == 200
    assert fake_model.states_built == [("alba", False)]
    assert fake_model.generated[0][1] == "Hello there"
    assert fake_model.generated[0][2] is True  # never mutate a cached state


def test_synthesize_preset_voice_is_used(client, fake_model):
    assert _post(client, _payload(voice="george")).status_code == 200
    assert fake_model.states_built == [("george", False)]


def test_synthesize_clip_is_staged_and_truncated(client, fake_model):
    assert _post(client, _clip_payload()).status_code == 200
    ((path, truncate),) = fake_model.states_built
    assert truncate is True  # engine caps prompts at 30 s
    assert not os.path.exists(path)  # the staged temp clip is cleaned up


def test_synthesize_voice_state_cached_per_clip_and_preset(client, fake_model):
    clip = _clip_payload()
    for _ in range(3):
        assert _post(client, clip).status_code == 200
        assert _post(client, _payload(voice="george")).status_code == 200
    # One build per distinct voice source, however many requests.
    assert len(fake_model.states_built) == 2
    assert len(fake_model.generated) == 6


def test_synthesize_different_clips_get_different_states(client, fake_model):
    _post(client, _payload(audio_base64=b64(make_wav_bytes(3.0, freq=440.0))))
    _post(client, _payload(audio_base64=b64(make_wav_bytes(3.0, freq=880.0))))
    assert len(fake_model.states_built) == 2


def test_voice_state_cache_is_bounded(client, fake_model):
    names = list(srv._ORIGINS_OF_PREDEFINED_VOICES)[: srv.VOICE_STATE_CACHE_SIZE + 3]
    for name in names:
        assert _post(client, _payload(voice=name)).status_code == 200
    assert len(srv._voice_states) == srv.VOICE_STATE_CACHE_SIZE


def test_synthesize_seed_applied_and_echoed(client, fake_model):
    body = _post(client, _payload(seed=42)).json()
    assert body["seed"] == 42
    assert fake_model.seeds == [42]


def test_synthesize_omitted_seed_is_random_in_range_and_echoed(client, fake_model):
    body = _post(client, _payload()).json()
    assert srv.SEED_MIN <= body["seed"] <= srv.SEED_MAX
    assert fake_model.seeds == [body["seed"]]


def test_synthesize_response_shape(client, fake_model):
    body = _post(client, _payload()).json()
    assert set(body) == {"audio_base64", "sample_rate", "seed", "time_used", "rtf", "fid"}
    assert body["sample_rate"] == 24000


def test_synthesize_language_not_forwarded_and_wellformed_accepted(client, fake_model):
    # docs/02: a single-language engine ignores the client's code without error.
    assert _post(client, _payload(language="fr")).status_code == 200
    assert _post(client, _payload(language="x?")).status_code == 422


def test_synthesize_engine_failure_is_500(client, fake_model, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("engine blew up")

    monkeypatch.setattr(fake_model, "generate_audio", boom)
    response = _post(client, _payload())
    assert response.status_code == 500
    assert "engine blew up" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Config validation (import-time, fail-fast)
# ---------------------------------------------------------------------------


def _import_with_env(**env):
    full = dict(os.environ)
    full.update(env)
    full["PYTHONPATH"] = os.pathsep.join(
        [str(_bootstrap.COMMON_SRC), str(_bootstrap.IMPL_DIR), str(_bootstrap.STUBS_DIR)]
    )
    return subprocess.run(
        [sys.executable, "-c", "import server_pocketTTS as s; print(s.LANGUAGE_CODE)"],
        env=full,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_model_name_selects_the_language_code():
    result = _import_with_env(POCKET_TTS_MODEL="french_24l")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fr"


def test_bad_model_name_fails_fast():
    result = _import_with_env(POCKET_TTS_MODEL="klingon")
    assert result.returncode != 0
    assert "POCKET_TTS_MODEL" in result.stderr


def test_bad_quantize_value_fails_fast():
    result = _import_with_env(POCKET_TTS_QUANTIZE="yes")
    assert result.returncode != 0
    assert "POCKET_TTS_QUANTIZE" in result.stderr
