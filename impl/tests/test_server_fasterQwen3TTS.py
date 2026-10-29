"""Tests for ``server_fasterQwen3TTS.py``.

Covers the model-free HTTP surface: /capabilities (snapshot), /health, the
landing page, request-body validation (422s), and the reference-audio
pre-flight checks (400s).  The shared staging helper is tested in
``tts-engine-common/tests/test_staging.py`` (it is engine-agnostic).

A handful of /synthesize success-path tests use a fake model (below) purely
to pin the argument forwarding to ``generate_voice_clone``; the engine call
itself still needs the model + GPU and is out of scope here.
"""

import types

import pytest
from fastapi.testclient import TestClient

import server_fasterQwen3TTS as srv
from helpers import b64, load_snapshot, make_wav_bytes


@pytest.fixture(scope="module")
def client():
    # Deliberately NOT a context manager: entering it would run the FastAPI
    # lifespan, which loads the (stubbed) model.  Not needed for these tests.
    return TestClient(srv.app)


# ---------------------------------------------------------------------------
# GET /capabilities
# ---------------------------------------------------------------------------


def test_capabilities_matches_snapshot(client):
    response = client.get("/capabilities")
    assert response.status_code == 200
    doc = response.json()
    assert doc == load_snapshot("faster_qwen3_capabilities.json")


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
    # The transcript is a hard requirement *in ICL mode* (enforced by a
    # cross-field validator), but not at the schema level: x-vector mode —
    # the default — ignores it.
    assert by_name["reference_text"]["required"] is False
    assert by_name["language"]["required"] is False
    assert by_name["seed"]["required"] is False


def test_capabilities_xvec_only_spec(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    spec = by_name["xvec_only"]
    assert spec["type"] == "boolean"
    assert spec["required"] is False
    # x-vector mode is the default: it is the stable-voice mode, the one
    # that fixes sentence-by-sentence streaming.
    assert spec["default"] is True


# ---------------------------------------------------------------------------
# GET /health and the landing page
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["serverType"] == "faster-qwen3-tts"
    assert body["device"] == srv.DEVICE
    assert body["model"] == srv.MODEL_NAME_OR_PATH
    # The tts-serve release version comes from the single shared constant;
    # pinning to srv.RELEASE_VERSION (not a literal) keeps this release-proof
    # while still catching a server that forgot to wire it in.
    assert body["version"] == srv.RELEASE_VERSION


def test_root_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Faster Qwen3-TTS" in response.text
    assert "/synthesize" in response.text


# ---------------------------------------------------------------------------
# POST /synthesize — request-body validation (422)
# ---------------------------------------------------------------------------


def _post(client, payload):
    return client.post("/synthesize", json=payload)


def _valid_payload():
    return {
        "text": "Hello there",
        "audio_base64": b64(make_wav_bytes(3.0)),
        "reference_text": "A short, exact transcript of the reference clip.",
    }


def test_synthesize_unknown_field_rejected(client):
    payload = _valid_payload()
    payload["bogus_field"] = 1
    assert _post(client, payload).status_code == 422


def test_synthesize_missing_required_fields_rejected(client):
    assert _post(client, {}).status_code == 422


def test_synthesize_missing_reference_text_xvec_mode_accepted(client, fake_model):
    # x-vector mode (the default) ignores the transcript, so omitting it is
    # valid — this is the sentence-streaming use case.
    payload = _valid_payload()
    del payload["reference_text"]
    response = _post(client, payload)
    assert response.status_code == 200
    # The engine still receives an explicit (empty) transcript.
    assert fake_model.calls[-1]["xvec_only"] is True
    assert fake_model.calls[-1]["ref_text"] == ""


def test_synthesize_explicit_null_reference_text_xvec_mode_accepted(client, fake_model):
    # An explicit null means 'no transcript' — it must validate like an
    # omitted field, not 500 inside the field validator (regression:
    # Pydantic runs after-mode validators for an explicitly-provided None).
    payload = _valid_payload()
    payload["reference_text"] = None
    response = _post(client, payload)
    assert response.status_code == 200
    assert fake_model.calls[-1]["xvec_only"] is True
    assert fake_model.calls[-1]["ref_text"] == ""


def test_synthesize_explicit_null_reference_text_icl_mode_rejected(client):
    # In ICL mode the transcript is a hard requirement: an explicit null is
    # 'no transcript', so it 422s with the cross-field message.
    payload = _valid_payload()
    payload["reference_text"] = None
    payload["xvec_only"] = False
    response = _post(client, payload)
    assert response.status_code == 422
    assert "reference_text" in response.text


def test_synthesize_icl_mode_missing_reference_text_rejected(client):
    # ICL (xvec_only=false) conditions on the transcript as well as the
    # audio, so without one the request is invalid at the boundary — not a
    # 400 and not a silent fallback to x-vector mode.
    payload = _valid_payload()
    del payload["reference_text"]
    payload["xvec_only"] = False
    response = _post(client, payload)
    assert response.status_code == 422
    assert "reference_text" in response.text


def test_synthesize_xvec_only_non_boolean_rejected(client):
    # Pydantic v2 lax mode would coerce "yes"/"true"/"1"; use a value that
    # is not coercible.
    payload = _valid_payload()
    payload["xvec_only"] = "banana"
    assert _post(client, payload).status_code == 422


def test_synthesize_icl_mode_forwarded_to_engine(client, fake_model):
    payload = _valid_payload()
    payload["xvec_only"] = False
    response = _post(client, payload)
    assert response.status_code == 200
    assert fake_model.calls[-1]["xvec_only"] is False
    assert fake_model.calls[-1]["ref_text"] == payload["reference_text"]


def test_synthesize_xvec_mode_transcript_not_forwarded(client, fake_model):
    # x-vector mode ignores the transcript — and must forward a canonical ""
    # to the engine, because the transcript is part of the engine's voice-
    # prompt cache key (forwarding it would create duplicate entries for the
    # same speaker embedding).
    payload = _valid_payload()  # includes reference_text; xvec_only defaults True
    response = _post(client, payload)
    assert response.status_code == 200
    assert fake_model.calls[-1]["xvec_only"] is True
    assert fake_model.calls[-1]["ref_text"] == ""


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


# The shared docs/02 language contract (case, names, garbage, non-strings,
# null/empty -> 'en') is asserted once for every server in
# test_language_contract.py; keep only engine-specific language tests here.


def test_synthesize_temperature_above_range_rejected(client):
    payload = _valid_payload()
    payload["temperature"] = 3.0
    assert _post(client, payload).status_code == 422


def test_synthesize_top_p_above_range_rejected(client):
    payload = _valid_payload()
    payload["top_p"] = 1.5
    assert _post(client, payload).status_code == 422


def test_synthesize_repetition_penalty_below_range_rejected(client):
    payload = _valid_payload()
    payload["repetition_penalty"] = 0.5
    assert _post(client, payload).status_code == 422


def test_synthesize_seed_out_of_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 0
    assert _post(client, payload).status_code == 422


# ---------------------------------------------------------------------------
# POST /synthesize — reference-audio pre-flight (400)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setattr(
        srv,
        "_runtime",
        types.SimpleNamespace(sample_rate=srv.SAMPLE_RATE, device=srv.DEVICE),
    )


class _FakeModel:
    """Records ``generate_voice_clone`` kwargs; returns a silent waveform."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate_voice_clone(self, **kwargs):
        self.calls.append(kwargs)
        # Like the engine: a list of waveforms, wavs[0] being the 1-D
        # audio.  A plain list is enough for the endpoint's len() math.
        waveform = [0.0] * (srv.SAMPLE_RATE // 10)  # ~0.1 s of silence
        return [waveform], srv.SAMPLE_RATE


@pytest.fixture
def fake_model(monkeypatch):
    """Install a recording model and a no-op WAV encoder.

    The stub numpy/soundfile refuse ``clip()``/``write()`` by design (real
    synthesis needs the model + GPU), so the success-path tests stand in for
    the encoder: they assert only on what the server sent to the engine.
    """
    model = _FakeModel()
    monkeypatch.setattr(
        srv,
        "_runtime",
        types.SimpleNamespace(
            model=model, sample_rate=srv.SAMPLE_RATE, device=srv.DEVICE
        ),
    )
    monkeypatch.setattr(srv, "_numpy_to_wav_bytes", lambda arr, sr: b"RIFFfake")
    return model


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



