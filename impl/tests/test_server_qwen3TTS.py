"""Tests for ``server_qwen3TTS.py``.

Covers the model-free HTTP surface: /capabilities (snapshot), /health, the
landing page, request-body validation (422s), and the reference-audio
pre-flight checks (400s).  Real synthesis needs the model + GPU and is out
of scope here.
"""

import types

import pytest
from fastapi.testclient import TestClient

import server_qwen3TTS as srv
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
    assert doc == load_snapshot("qwen3_capabilities.json")


def test_capabilities_language_enum_includes_auto(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    enum = by_name["language"]["enum"]
    assert "auto" in enum
    # Language values are lowercase *names* (the engine's own convention).
    assert "english" in enum
    assert "English" not in enum
    assert doc["languages"] == sorted(srv.LANGUAGE_NAMES)


def test_capabilities_required_fields(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    assert by_name["text"]["required"] is True
    assert by_name["audio_base64"]["required"] is True
    assert by_name["reference_text"]["required"] is False


# ---------------------------------------------------------------------------
# GET /health and the landing page
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["serverType"] == "Qwen3-TTS"
    assert body["device"] == srv.DEVICE
    assert body["model"] == srv.MODEL_NAME_OR_PATH


def test_root_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Qwen3-TTS" in response.text
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


def test_synthesize_unsupported_language_rejected(client):
    payload = _valid_payload()
    payload["language"] = "french-quebec"  # not a Base-checkpoint language
    assert _post(client, payload).status_code == 422


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


def test_synthesize_x_vector_only_wrong_type_rejected(client):
    # Note: "yes"/"true"/"1" are legitimately coerced to booleans by
    # Pydantic v2's lax validation — use a value that is not.
    payload = _valid_payload()
    payload["x_vector_only_mode"] = "banana"
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
    monkeypatch.setattr(srv, "_runtime", types.SimpleNamespace(device=srv.DEVICE))


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
