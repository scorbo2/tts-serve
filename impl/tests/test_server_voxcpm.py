"""Tests for ``server_voxcpm.py``.

Covers the model-free HTTP surface: /capabilities (snapshot), /health, the
landing page, request-body validation (422s), and the reference-audio
pre-flight checks (400s).  Real synthesis needs the model + GPU and is out
of scope here.
"""

import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server_voxcpm as srv
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
    assert doc == load_snapshot("voxcpm_capabilities.json")


def test_capabilities_sample_rate_is_48k(client):
    doc = client.get("/capabilities").json()
    assert doc["sample_rate"] == 48000


def test_capabilities_languages_is_null(client):
    """VoxCPM auto-detects language from text — no forwarding table."""
    doc = client.get("/capabilities").json()
    assert doc["languages"] is None


def test_capabilities_reference_audio_optional(client):
    """Reference audio is optional for voice design mode."""
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    ref = by_name["audio_base64"]
    assert ref["required"] is False


def test_capabilities_required_fields(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    assert by_name["text"]["required"] is True
    assert by_name["audio_base64"]["required"] is False
    assert by_name["reference_text"]["required"] is False


# ---------------------------------------------------------------------------
# GET /health and the landing page
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["serverType"] == "VoxCPM"
    assert body["device"] == srv.DEVICE
    assert body["model"] == srv.MODEL_NAME_OR_PATH


def test_root_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "VoxCPM" in response.text
    assert "/synthesize" in response.text


# ---------------------------------------------------------------------------
# POST /synthesize — request-body validation (422)
# ---------------------------------------------------------------------------


def _post(client, payload):
    return client.post("/synthesize", json=payload)


def _valid_payload():
    return {
        "text": "Hello there",
    }


def _payload_with_ref():
    return {
        "text": "Hello there",
        "audio_base64": b64(make_wav_bytes(3.0)),
    }


def test_synthesize_unknown_field_rejected(client):
    payload = _valid_payload()
    payload["bogus_field"] = 1
    assert _post(client, payload).status_code == 422


def test_synthesize_missing_required_fields_rejected(client):
    # text is the only hard-required field
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
    # audio_base64 is optional, but if provided it must be non-empty
    payload = _valid_payload()
    payload["audio_base64"] = ""
    assert _post(client, payload).status_code == 422


def test_synthesize_cfg_value_above_range_rejected(client):
    payload = _valid_payload()
    payload["cfg_value"] = 11.0
    assert _post(client, payload).status_code == 422


def test_synthesize_cfg_value_below_range_rejected(client):
    payload = _valid_payload()
    payload["cfg_value"] = -1.0
    assert _post(client, payload).status_code == 422


def test_synthesize_inference_timesteps_above_range_rejected(client):
    payload = _valid_payload()
    payload["inference_timesteps"] = 51
    assert _post(client, payload).status_code == 422


def test_synthesize_inference_timesteps_below_range_rejected(client):
    payload = _valid_payload()
    payload["inference_timesteps"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_seed_out_of_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_seed_too_large_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 1001
    assert _post(client, payload).status_code == 422


# ---------------------------------------------------------------------------
# POST /synthesize — reference-audio pre-flight (400)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_runtime(monkeypatch):
    # Return a list of zeros — the handler calls len() and iterates,
    # which works fine on a list.  No need for real numpy in tests.
    mock_model = types.SimpleNamespace()
    mock_model.generate = lambda **kwargs: [0.0] * 48000
    monkeypatch.setattr(
        srv, "_runtime", types.SimpleNamespace(
            sample_rate=48000, device="cpu", model=mock_model
        )
    )


def test_synthesize_undecodable_audio_rejected(client, fake_runtime):
    payload = _payload_with_ref()
    payload["audio_base64"] = b64(b"this is definitely not audio")
    response = _post(client, payload)
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_synthesize_too_short_audio_rejected(client, fake_runtime):
    payload = _payload_with_ref()
    payload["audio_base64"] = b64(make_wav_bytes(0.5))  # 0.5 s < 2.0 s minimum
    response = _post(client, payload)
    assert response.status_code == 400
    assert "2" in response.json()["detail"]


def test_synthesize_voice_design_no_audio_passes_validation(client, fake_runtime):
    # Voice design (no reference audio) should pass request validation.
    # With undecodable audio we can't test the handler path, but with no
    # audio at all the handler proceeds to synthesis (which 500s on the
    # stub — proving we cleared validation).
    payload = _valid_payload()
    response = _post(client, payload)
    # The stub model raises NotImplementedError → 500, not 422.
    assert response.status_code == 500
    assert "not available in tests" in response.json()["detail"]


def test_synthesize_with_reference_audio_passes_validation(client, fake_runtime):
    # With valid reference audio the handler proceeds to synthesis (500 on stub).
    payload = _payload_with_ref()
    response = _post(client, payload)
    assert response.status_code == 500
    assert "not available in tests" in response.json()["detail"]


def test_synthesize_with_reference_text_passes_validation(client, fake_runtime):
    # Ultimate cloning mode: audio + transcript.
    payload = _payload_with_ref()
    payload["reference_text"] = "A short transcript of the reference clip."
    response = _post(client, payload)
    assert response.status_code == 500
    assert "not available in tests" in response.json()["detail"]
