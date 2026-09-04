"""Tests for ``server_chatterbox.py``.

These exercise the HTTP surface that does NOT require a loaded model:
/capabilities (snapshot), /health, the landing page, request-body validation
(422s), and the reference-audio pre-flight checks (400s).  Synthesis itself
needs a real model + GPU, so it is intentionally out of scope here.
"""

import types

import pytest
from fastapi.testclient import TestClient

import server_chatterbox as srv
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
    assert doc == load_snapshot("chatterbox_capabilities.json")


def test_capabilities_core_fields_are_required(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    # The two fields the client must always supply.
    assert by_name["text"]["required"] is True
    assert by_name["audio_base64"]["required"] is True
    # Optional tuning knobs default to the model's own defaults.
    assert by_name["seed"]["required"] is False
    assert by_name["exaggeration"]["default"] == 0.5


def test_capabilities_language_enum_matches_engine_table(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    assert by_name["language"]["enum"] == list(srv.SUPPORTED_LANGUAGES)
    assert doc["languages"] == list(srv.SUPPORTED_LANGUAGES)


# ---------------------------------------------------------------------------
# GET /health and the landing page
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["serverType"] == "Chatterbox"
    assert body["device"] == srv.DEVICE
    assert body["model"] == srv.MODEL_LABEL


def test_root_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Chatterbox" in response.text
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


def test_synthesize_unsupported_language_rejected(client):
    payload = _valid_payload()
    payload["language"] = "xx"
    assert _post(client, payload).status_code == 422


def test_synthesize_seed_below_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_seed_above_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 1001
    assert _post(client, payload).status_code == 422


def test_synthesize_exaggeration_above_range_rejected(client):
    payload = _valid_payload()
    payload["exaggeration"] = 5.0
    assert _post(client, payload).status_code == 422


def test_synthesize_cfg_weight_above_range_rejected(client):
    payload = _valid_payload()
    payload["cfg_weight"] = 1.5
    assert _post(client, payload).status_code == 422


# ---------------------------------------------------------------------------
# POST /synthesize — reference-audio pre-flight (400).  The handler runs up to
# the audio check, so the (stubbed) runtime is faked out to skip model load.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setattr(
        srv, "_runtime", types.SimpleNamespace(sample_rate=srv.S3GEN_SR, device=srv.DEVICE)
    )


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
