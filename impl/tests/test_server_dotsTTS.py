"""Tests for ``server_dotsTTS.py``.

Covers the model-free HTTP surface: /capabilities (snapshot), /health, the
landing page, request-body validation (422s), and the reference-audio
pre-flight checks (400s).  Real synthesis needs the model + GPU and is out
of scope here.
"""

import types

import pytest
from fastapi.testclient import TestClient

import server_dotsTTS as srv
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
    snapshot = load_snapshot("dots_capabilities.json")

    # ``device`` is machine-dependent: the dots.tts runtime picks CUDA when
    # available, else CPU, and the server mirrors that decision at import
    # time.  Compare everything else, and assert the device is a sane value.
    assert doc["device"] in ("cuda", "cpu")
    doc.pop("device")
    snapshot.pop("device")
    assert doc == snapshot


def test_capabilities_ode_method_enum(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    assert by_name["ode_method"]["enum"] == ["euler", "midpoint", "rk4"]


def test_capabilities_required_fields(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    assert by_name["text"]["required"] is True
    assert by_name["audio_base64"]["required"] is True
    assert by_name["reference_text"]["required"] is False


def test_capabilities_uses_48khz(client):
    # dots.tts' AudioVAE vocoder is 48 kHz — the odd one out in tts-serve.
    doc = client.get("/capabilities").json()
    assert doc["sample_rate"] == 48000


# ---------------------------------------------------------------------------
# GET /health and the landing page
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["serverType"] == "dots.tts"
    assert body["device"] == srv.DEVICE
    assert body["model"] == srv.MODEL_NAME_OR_PATH


def test_root_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "dots.tts" in response.text
    assert "/synthesize" in response.text


# ---------------------------------------------------------------------------
# POST /synthesize — request-body validation (422)
# ---------------------------------------------------------------------------


def _post(client, payload):
    return client.post("/synthesize", json=payload)


def _valid_payload():
    # dots.tts is 48 kHz; the duration check only cares about seconds.
    return {
        "text": "Hello there",
        "audio_base64": b64(make_wav_bytes(3.0, sr=48000)),
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


def test_synthesize_unknown_ode_method_rejected(client):
    payload = _valid_payload()
    payload["ode_method"] = "rk2"
    assert _post(client, payload).status_code == 422


def test_synthesize_num_steps_below_range_rejected(client):
    payload = _valid_payload()
    payload["num_steps"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_guidance_scale_above_range_rejected(client):
    payload = _valid_payload()
    payload["guidance_scale"] = 5.5
    assert _post(client, payload).status_code == 422


def test_synthesize_speaker_scale_negative_rejected(client):
    payload = _valid_payload()
    payload["speaker_scale"] = -0.1
    assert _post(client, payload).status_code == 422


def test_synthesize_seed_out_of_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 1001
    assert _post(client, payload).status_code == 422


# ---------------------------------------------------------------------------
# POST /synthesize — reference-audio pre-flight (400)
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
    payload["audio_base64"] = b64(make_wav_bytes(0.5, sr=48000))  # 0.5 s < 2.0 s
    response = _post(client, payload)
    assert response.status_code == 400
    assert "2" in response.json()["detail"]
