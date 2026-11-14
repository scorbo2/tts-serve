"""Tests for ``server_qwen3TTS_mlx.py``.

Covers the model-free HTTP surface: /capabilities (snapshot), /health, the
landing page, request-body validation (422s), synthesis edge cases (500s),
and the reference-audio pre-flight checks (400s). Real synthesis needs
mlx-audio + Apple Silicon and is out of scope here -- this suite must never
import MLX or load the model.
"""

import types

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
    """Install a model-free runtime and bypass full reference-audio decoding.

    The test soundfile stub supports header inspection via sf.info(), which is
    enough for the reference-audio pre-flight checks, but deliberately does not
    implement sf.read(). Synthesis-edge-case tests need to get past that decode
    step without depending on the real soundfile package or MLX.
    """
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


def test_capabilities_device_and_sample_rate(client):
    doc = client.get("/capabilities").json()
    assert doc["device"] == "mlx"
    assert doc["sample_rate"] == 24000
    assert doc["engine"] == "qwen3-tts-mlx"


# ---------------------------------------------------------------------------
# GET /health and the landing page
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["serverType"] == "Qwen3-TTS-MLX"
    assert body["device"] == "mlx"
    assert body["model"] == srv.MODEL_NAME_OR_PATH
    # The tts-serve release version comes from the single shared constant;
    # pinning to srv.RELEASE_VERSION (not a literal) keeps this release-proof
    # while still catching a server that forgot to wire it in.
    assert body["version"] == srv.RELEASE_VERSION


def test_root_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Qwen3-TTS MLX" in response.text
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


# The shared docs/02 language contract (case, names, garbage, non-strings,
# null/empty -> 'en') is asserted once for every server in
# test_language_contract.py; keep only engine-specific language tests here.


def test_synthesize_seed_below_min_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_seed_above_max_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 1001
    assert _post(client, payload).status_code == 422


# ---------------------------------------------------------------------------
# POST /synthesize — synthesis edge cases (500)
# ---------------------------------------------------------------------------


def test_synthesize_no_generated_audio_returns_500(client, fake_runtime):
    fake_runtime.model = types.SimpleNamespace(generate=lambda **kwargs: iter(()))

    payload = _valid_payload()
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
