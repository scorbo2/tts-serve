"""Tests for ``server_kittenTTS.py``.

These exercise the HTTP surface that does NOT require a loaded model:
/capabilities (snapshot), /health, the landing page, request-body validation
(422s), reference-audio pre-flight (400s), and runtime table checks
(unknown voice / preset, 422s).  A couple of success-path tests use a fake
model purely to pin the argument forwarding to ``KittenTTS2.generate()``;
actual synthesis needs a real model (and roughly 6 GB RAM / 8 GB VRAM).
"""

import os
import subprocess
import sys
import types

import pytest
from fastapi.testclient import TestClient

import _bootstrap
import server_kittenTTS as srv
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
    # ``device`` is machine-dependent: KITTENTTS_DEVICE defaults to 'auto',
    # and the capabilities document mirrors the engine's pick (CUDA when
    # available, else CPU) at import time.  Compare everything else, and
    # assert the device is a sane value (the dots.tts pattern).
    response = client.get("/capabilities")
    assert response.status_code == 200
    doc = response.json()
    snapshot = load_snapshot("kittentts_capabilities.json")
    assert doc["device"] in ("cuda", "cpu")
    doc.pop("device")
    snapshot.pop("device")
    assert doc == snapshot


def test_capabilities_core_fields_and_optionality(client):
    doc = client.get("/capabilities").json()
    assert doc["engine"] == "kittentts"
    assert doc["sample_rate"] == 24000
    assert doc["watermarked"] is False
    by_name = {p["name"]: p for p in doc["parameters"]}
    # text is the only mandatory field: the speaker comes from the model's
    # built-in voices when no clip is sent.
    assert by_name["text"]["required"] is True
    assert by_name["audio_base64"]["required"] is False
    assert by_name["reference_text"]["required"] is False
    # Optional knobs default to the engine's own defaults; the None-defaulted
    # ones mean "the repository's own value" (voice/preset tables and preset
    # sampling numbers are per-repository data, unknown at import time).
    assert by_name["voice"]["default"] is None
    assert by_name["preset"]["default"] is None
    assert by_name["normalize"]["default"] is True
    assert by_name["max_new_tokens"]["default"] == 1000
    assert by_name["temperature"]["default"] is None
    assert by_name["top_k"]["default"] is None
    assert by_name["seed"]["required"] is False


def test_capabilities_language_not_forwarded(client):
    # docs/02 no-support case (LuxTTS/VoxCPM precedent): field present with
    # the 'en' default, no enum, languages advertised as null.
    doc = client.get("/capabilities").json()
    assert doc["languages"] is None
    by_name = {p["name"]: p for p in doc["parameters"]}
    assert by_name["language"]["type"] == "string"
    assert by_name["language"]["enum"] is None
    assert by_name["language"]["default"] == "en"


# ---------------------------------------------------------------------------
# GET /health and the landing page
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["serverType"] == "KittenTTS"
    assert body["device"] == srv.DEVICE
    assert body["model"] == srv.MODEL_NAME_OR_PATH
    # The tts-serve release version comes from the single shared constant;
    # pinning to srv.RELEASE_VERSION (not a literal) keeps this release-proof
    # while still catching a server that forgot to wire it in.
    assert body["version"] == srv.RELEASE_VERSION


def test_root_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "KittenTTS" in response.text
    assert "/synthesize" in response.text


# ---------------------------------------------------------------------------
# POST /synthesize — request-body validation (422).  Validation happens before
# the handler runs, so no model is involved.
# ---------------------------------------------------------------------------


def _post(client, payload):
    return client.post("/synthesize", json=payload)


def _valid_payload():
    # A 6 s clip: comfortably above the 5 s cloning floor.
    return {
        "text": "Hello there",
        "audio_base64": b64(make_wav_bytes(6.0)),
    }


def test_synthesize_unknown_field_rejected(client):
    payload = _valid_payload()
    payload["bogus_field"] = 1
    assert _post(client, payload).status_code == 422


def test_synthesize_missing_required_text_rejected(client):
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


def test_synthesize_voice_and_audio_together_rejected(client):
    # The engine's "give either voice= or reference=, not both" mapped to a
    # 422 at the boundary (Pocket TTS precedent), never a silent pick.
    payload = _valid_payload()
    payload["voice"] = "Bruno"
    response = _post(client, payload)
    assert response.status_code == 422
    assert "mutually exclusive" in str(response.json()["detail"])


def test_synthesize_reference_text_without_audio_rejected(client):
    payload = {"text": "Hello there", "reference_text": "a transcript with no clip"}
    response = _post(client, payload)
    assert response.status_code == 422
    assert "reference_text requires audio_base64" in str(response.json()["detail"])


def test_synthesize_seed_below_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_seed_above_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 1001
    assert _post(client, payload).status_code == 422


def test_synthesize_max_new_tokens_out_of_range_rejected(client):
    payload = _valid_payload()
    payload["max_new_tokens"] = 2001
    assert _post(client, payload).status_code == 422


def test_synthesize_temperature_above_range_rejected(client):
    payload = _valid_payload()
    payload["temperature"] = 2.5
    assert _post(client, payload).status_code == 422


def test_synthesize_top_p_above_range_rejected(client):
    payload = _valid_payload()
    payload["top_p"] = 1.5
    assert _post(client, payload).status_code == 422


def test_synthesize_normalize_bad_type_rejected(client):
    # Pydantic v2 lax mode coerces "yes"/"true"/"1" to booleans — use a value
    # it cannot coerce.
    payload = _valid_payload()
    payload["normalize"] = "definitely not a bool"
    assert _post(client, payload).status_code == 422


# The shared docs/02 language contract (case, names, garbage, non-strings,
# null/empty -> 'en', no 'auto') is asserted once for every server in
# test_language_contract.py; keep only engine-specific language tests here.


def test_synthesize_language_wellformed_nonEnglish_passes(client, monkeypatch):
    # The engine has no fixed language list (languages: null), so any
    # well-formed two-letter code clears validation; with undecodable audio
    # the handler then fails its 400 audio pre-flight -- proving we got past
    # validation (actual synthesis needs a real model + GPU).
    monkeypatch.setattr(
        srv, "_runtime",
        types.SimpleNamespace(sample_rate=srv.SAMPLE_RATE, device=srv.DEVICE),
    )
    payload = {
        "text": "Hello there",
        "audio_base64": b64(b"this is definitely not audio"),
        "language": "de",
    }
    response = _post(client, payload)
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# POST /synthesize -- reference-audio pre-flight (400).  The handler runs up to
# the audio check, so the (stubbed) runtime is faked out to skip model load.
# ---------------------------------------------------------------------------


class _StubModel:
    """Raises where the import-only engine stub would: at the generate()
    boundary.  The voice/preset tables mirror the kitten-tts-2 checkpoint so
    the runtime table checks (422) exercise their real shapes."""

    available_voices = ["Bruno", "Bella", "German"]
    decode_presets = {"stable": {}, "expressive": {}}

    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            "kittenml stub: generate() is not available in tests"
        )


@pytest.fixture
def fake_runtime(monkeypatch):
    # The model raises at generate(), so tests that reach it prove the request
    # cleared the 422 boundary (actual synthesis needs a real model + GPU).
    monkeypatch.setattr(
        srv,
        "_runtime",
        types.SimpleNamespace(
            sample_rate=srv.SAMPLE_RATE, device=srv.DEVICE, model=_StubModel()
        ),
    )


def test_synthesize_undecodable_audio_rejected(client, fake_runtime):
    payload = _valid_payload()
    payload["audio_base64"] = b64(b"this is definitely not audio")
    response = _post(client, payload)
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_synthesize_too_short_audio_rejected(client, fake_runtime):
    payload = _valid_payload()
    payload["audio_base64"] = b64(make_wav_bytes(2.0))  # 2.0 s < 5.0 s minimum
    response = _post(client, payload)
    assert response.status_code == 400
    assert "5" in response.json()["detail"]


def test_synthesize_default_voice_no_audio_reaches_model(client, fake_runtime):
    # Built-in default voice (no audio, no voice field): clears validation and
    # the (skipped) audio pre-flight; the stub model then raises -> 500.
    response = _post(client, {"text": "Hello there"})
    assert response.status_code == 500
    assert "not available in tests" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /synthesize -- runtime table checks (voice / preset names are per-model
# repository data, so they cannot be Literal enums; the handler 422s them).
# ---------------------------------------------------------------------------


def test_synthesize_unknown_voice_rejected(client, fake_runtime):
    response = _post(client, {"text": "Hello there", "voice": "NoSuchVoice"})
    assert response.status_code == 422
    assert "voice" in response.json()["detail"]


def test_synthesize_known_voice_passes_table_check(client, fake_runtime):
    response = _post(client, {"text": "Hello there", "voice": "Bruno"})
    assert response.status_code == 500  # reached generate(); cleared the 422
    assert "not available in tests" in response.json()["detail"]


def test_synthesize_unknown_preset_rejected(client, fake_runtime):
    payload = {"text": "Hello there", "preset": "dramatic"}
    response = _post(client, payload)
    assert response.status_code == 422
    assert "preset" in response.json()["detail"]


def test_synthesize_known_preset_passes_table_check(client, fake_runtime):
    response = _post(client, {"text": "Hello there", "preset": "expressive"})
    assert response.status_code == 500  # reached generate(); cleared the 422
    assert "not available in tests" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /synthesize -- success path with a recording fake model: pins the
# argument forwarding to KittenTTS2.generate().  The stub numpy/soundfile
# refuse clip()/write() by design, so the WAV encoder is stood in for; these
# tests assert only on what the server sent to the engine.
# ---------------------------------------------------------------------------


class _RecordingModel(_StubModel):
    """Records generate() kwargs; returns a silent waveform."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        # Like the engine: a 1-D float list at model.sample_rate (the server
        # feeds it through np.asarray + len(), which the patched asarray
        # below turns into a plain shim).
        return [0.0] * (srv.SAMPLE_RATE // 10)  # ~0.1 s of silence


class _ArrayShim:
    def __init__(self, values):
        self._values = list(values)

    def reshape(self, *_shape):
        return self

    def __len__(self):
        return len(self._values)


@pytest.fixture
def fake_model(monkeypatch):
    model = _RecordingModel()
    monkeypatch.setattr(
        srv,
        "_runtime",
        types.SimpleNamespace(
            model=model, sample_rate=srv.SAMPLE_RATE, device=srv.DEVICE
        ),
    )
    monkeypatch.setattr(srv.np, "asarray", lambda obj, dtype=None: _ArrayShim(obj))
    monkeypatch.setattr(srv, "_numpy_to_wav_bytes", lambda arr, sr: b"RIFFfake")
    return model


def test_synthesize_builtinVoice_defaultForwards(client, fake_model):
    # No voice, no clip: both forwarded as None; the engine falls back to the
    # repository's default voice.  The resolved seed must reach the engine's
    # advanced knobs AND be echoed in the response.
    response = _post(client, {"text": "Hello there", "seed": 42})
    assert response.status_code == 200
    call = fake_model.calls[-1]
    assert call["voice"] is None
    assert call["reference"] is None
    assert call["reference_text"] is None
    assert call["advanced"] == {"seed": 42}
    assert response.json()["seed"] == 42


def test_synthesize_randomSeedResolvedInRange(client, fake_model):
    response = _post(client, {"text": "Hello there"})
    assert response.status_code == 200
    seed = response.json()["seed"]
    assert srv.SEED_MIN <= seed <= srv.SEED_MAX
    assert fake_model.calls[-1]["advanced"] == {"seed": seed}


def test_synthesize_namedVoiceForwarded(client, fake_model):
    response = _post(client, {"text": "Hello there", "voice": "Bruno"})
    assert response.status_code == 200
    call = fake_model.calls[-1]
    assert call["voice"] == "Bruno"
    assert call["reference"] is None


def test_synthesize_cloning_stagesReferenceAndForwardsIt(client, fake_model):
    # The clip must reach the engine as an existing *file path* (the engine
    # refuses in-memory audio), content-hashed and kept on disk because the
    # engine caches prompt/embedding conditioning per path.
    payload = _valid_payload()
    payload["reference_text"] = "Six seconds of clean speech."
    response = _post(client, payload)
    assert response.status_code == 200
    call = fake_model.calls[-1]
    assert call["voice"] is None
    reference_path = call["reference"]
    assert isinstance(reference_path, str)
    assert os.path.isfile(reference_path)
    assert os.path.basename(reference_path).endswith(".wav")
    assert len(os.path.basename(reference_path)) == 64 + len(".wav")  # SHA-256 name
    assert call["reference_text"] == payload["reference_text"]


def test_synthesize_tuningKnobsForwardedVerbatim(client, fake_model):
    payload = {
        "text": "Hello there",
        "preset": "expressive",
        "normalize": False,
        "max_new_tokens": 500,
        "temperature": 1.1,
        "top_k": 0,
        "top_p": 0.95,
        "min_p": 0.05,
    }
    response = _post(client, payload)
    assert response.status_code == 200
    call = fake_model.calls[-1]
    assert call["preset"] == "expressive"
    assert call["normalize"] is False
    assert call["max_new_tokens"] == 500
    assert call["temperature"] == 1.1
    assert call["top_k"] == 0
    assert call["top_p"] == 0.95
    assert call["min_p"] == 0.05


def test_synthesize_unsetSamplingOverridesForwardedAsNone(client, fake_model):
    # None must be passed through, not replaced with invented defaults: the
    # engine merges non-None overrides over the repository's preset values.
    response = _post(client, {"text": "Hello there", "voice": "Bella"})
    assert response.status_code == 200
    call = fake_model.calls[-1]
    assert call["temperature"] is None
    assert call["top_p"] is None
    assert call["top_k"] is None
    assert call["min_p"] is None


# ---------------------------------------------------------------------------
# Runtime wiring: _get_runtime guards and self-consistency
# ---------------------------------------------------------------------------


def test_get_runtime_rejects_legacy_onnx_repo(monkeypatch):
    # GIVEN a model repo that loads to the legacy ONNX backend (no
    # decode_presets attribute -- that is KittenTTS2-only):
    monkeypatch.setattr(srv, "_runtime", None)
    monkeypatch.setattr(
        srv, "KittenTTS", lambda *a, **k: types.SimpleNamespace(device="cpu")
    )

    # WHEN the runtime is loaded, THEN it fails fast with an actionable
    # message instead of accepting a model whose generate() this server
    # cannot drive.
    with pytest.raises(ValueError, match="not a KittenTTS 2 repository"):
        srv._get_runtime()


def test_get_runtime_readsSampleRateAndDeviceFromModel(monkeypatch):
    # GIVEN a KittenTTS2-shaped load whose reported rate/device differ from
    # the import-time constants (a hypothetical future checkpoint):
    monkeypatch.setattr(srv, "_runtime", None)
    loaded = types.SimpleNamespace(
        decode_presets={"stable": {}},
        available_voices=["Bruno"],
        sample_rate=48000,
        device="cpu",
    )
    monkeypatch.setattr(srv, "KittenTTS", lambda *a, **k: loaded)

    # THEN the runtime echoes what the loaded model says, not the guesses:
    runtime = srv._get_runtime()
    assert runtime.sample_rate == 48000
    assert runtime.device == "cpu"


# ---------------------------------------------------------------------------
# Import-time failure: bad KITTENTTS_DEVICE / KITTENTTS_WEIGHTS values fail
# the module import (fail-fast before any model download).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_name,bad_value",
    [
        pytest.param("KITTENTTS_DEVICE", "tpu", id="bad-device"),
        pytest.param("KITTENTTS_WEIGHTS", "quantum", id="bad-weights"),
    ],
)
def test_import_with_bad_env_var_fails_fast(env_name, bad_value):
    # GIVEN the same import environment the suite uses, plus a bad env var:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_bootstrap.COMMON_SRC), str(_bootstrap.IMPL_DIR), str(_bootstrap.STUBS_DIR)]
    )
    env[env_name] = bad_value

    # WHEN the server module is imported, THEN it exits non-zero naming the
    # offending variable -- not partway through a model download.
    result = subprocess.run(
        [sys.executable, "-c", "import server_kittenTTS"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert env_name in result.stderr
