"""Tests for ``server_omnivoice.py``.

Covers the model-free HTTP surface: /capabilities (snapshot), /health, the
landing page, request-body validation (422s), and the reference-audio
pre-flight checks (400s).  Real synthesis needs the model + GPU and is out
of scope here.
"""

import types

import pytest
from fastapi.testclient import TestClient

import server_omnivoice as srv
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
    assert doc == load_snapshot("omnivoice_capabilities.json")


def test_capabilities_language_has_no_enum(client):
    # OmniVoice covers ~646 languages, so the schema deliberately carries no
    # enum and the capabilities document no language list (the shared docs/02
    # contract — codes, default 'en' — is asserted in
    # test_language_contract.py).
    doc = client.get("/capabilities").json()
    assert doc["languages"] is None
    by_name = {p["name"]: p for p in doc["parameters"]}
    assert by_name["language"]["type"] == "string"
    assert by_name["language"].get("enum") is None


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
    assert body["serverType"] == "OmniVoice"
    assert body["device"] == srv.DEVICE
    assert body["model"] == srv.MODEL_NAME_OR_PATH


def test_root_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "OmniVoice" in response.text
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
    payload["text"] = "  \t "
    assert _post(client, payload).status_code == 422


def test_synthesize_empty_audio_rejected(client):
    payload = _valid_payload()
    payload["audio_base64"] = ""
    assert _post(client, payload).status_code == 422


def test_synthesize_num_steps_below_range_rejected(client):
    payload = _valid_payload()
    payload["num_steps"] = 3  # engine minimum is 4
    assert _post(client, payload).status_code == 422


def test_synthesize_num_steps_above_range_rejected(client):
    payload = _valid_payload()
    payload["num_steps"] = 129
    assert _post(client, payload).status_code == 422


def test_synthesize_guidance_scale_negative_rejected(client):
    payload = _valid_payload()
    payload["guidance_scale"] = -0.5
    assert _post(client, payload).status_code == 422


def test_synthesize_denoise_wrong_type_rejected(client):
    # Note: "yes"/"true"/"1" are legitimately coerced to booleans by
    # Pydantic v2's lax validation — use a value that is not.
    payload = _valid_payload()
    payload["denoise"] = "banana"
    assert _post(client, payload).status_code == 422


def test_synthesize_seed_out_of_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 1001
    assert _post(client, payload).status_code == 422


# The shared docs/02 language contract (case, names, garbage, non-strings,
# null/empty -> 'en') is asserted once for every server in
# test_language_contract.py; keep only engine-specific language tests here.


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
    payload["audio_base64"] = b64(make_wav_bytes(0.5))  # 0.5 s < 2.0 s minimum
    response = _post(client, payload)
    assert response.status_code == 400
    assert "2" in response.json()["detail"]
