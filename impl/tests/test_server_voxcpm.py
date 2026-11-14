"""Tests for ``server_voxcpm.py``.

These exercise the HTTP surface that does NOT require a loaded model:
/capabilities (snapshot), /health, the landing page, request-body validation
(422s), and the reference-audio pre-flight checks (400s).  The three synthesis
modes (voice design, controllable cloning, ultimate cloning) are additionally
proven to clear the 422 boundary and reach the engine call, which raises on
the import-only stub (500).  Actual synthesis needs a real model + GPU, so it
is intentionally out of scope here.
"""

import types

import pytest
from fastapi.testclient import TestClient

import server_voxcpm as srv
from helpers import b64, load_snapshot, make_wav_bytes


@pytest.fixture(scope="module")
def client():
    # Deliberately NOT used as a context manager: entering it would run the
    # FastAPI lifespan, which loads the (stubbed) model.  The endpoints under
    # test here don't need the model, so we skip lifespan entirely.
    return TestClient(srv.app)


# ---------------------------------------------------------------------------
# GET /capabilities -- snapshot (single source of truth: the Pydantic model)
# ---------------------------------------------------------------------------


def test_capabilities_matches_snapshot(client):
    response = client.get("/capabilities")
    assert response.status_code == 200
    doc = response.json()
    assert doc == load_snapshot("voxcpm_capabilities.json")


def test_capabilities_core_fields_are_required(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    # The one field every request must supply.
    assert by_name["text"]["required"] is True
    # Reference audio is optional: voice design mode needs no reference clip.
    assert by_name["audio_base64"]["required"] is False
    # Optional tuning knobs default to the model's own defaults.
    assert by_name["seed"]["required"] is False
    assert by_name["cfg_value"]["default"] == 2.0
    assert by_name["inference_timesteps"]["default"] == 10


def test_capabilities_sample_rate_and_languages(client):
    doc = client.get("/capabilities").json()
    # VoxCPM2's AudioVAE V2 outputs 48 kHz (confirmed against the engine).
    assert doc["sample_rate"] == 48000
    # The engine has no language parameter (auto-detects from text), so there
    # is no fixed list -- accept any well-formed two-letter code for consistency.
    assert doc["languages"] is None


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
    # The tts-serve release version comes from the single shared constant;
    # pinning to srv.RELEASE_VERSION (not a literal) keeps this release-proof
    # while still catching a server that forgot to wire it in.
    assert body["version"] == srv.RELEASE_VERSION


def test_root_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "VoxCPM" in response.text
    assert "/synthesize" in response.text


# ---------------------------------------------------------------------------
# POST /synthesize -- request-body validation (422).  Validation happens before
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
    # 'text' is the only required field; audio_base64 is optional (voice design).
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
    # audio_base64 has max_length validation: an empty string is below min_length.
    payload["audio_base64"] = ""
    assert _post(client, payload).status_code == 422


def test_synthesize_reference_text_without_audio_rejected(client):
    # A transcript with no clip is a client bug, not a mode switch: silently
    # degrading to voice design would hand back audio the client did not ask
    # for and mask the bug, so the boundary rejects it (422, not 500).
    payload = {
        "text": "Hello there",
        "reference_text": "Exact transcript of the reference clip.",
    }
    response = _post(client, payload)
    assert response.status_code == 422
    # The validator's message makes it unambiguous which pairing failed.
    assert "reference_text" in str(response.json()["detail"])


# The shared docs/02 language contract (case, names, garbage, non-strings,
# null/empty -> 'en') is asserted once for every server in
# test_language_contract.py; keep only engine-specific language tests here.


def test_synthesize_seed_below_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_seed_above_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 1001
    assert _post(client, payload).status_code == 422


def test_synthesize_cfg_value_above_range_rejected(client):
    payload = _valid_payload()
    payload["cfg_value"] = 10.1
    assert _post(client, payload).status_code == 422


def test_synthesize_cfg_value_below_range_rejected(client):
    payload = _valid_payload()
    payload["cfg_value"] = 0.0
    assert _post(client, payload).status_code == 422


def test_synthesize_inference_timesteps_above_range_rejected(client):
    payload = _valid_payload()
    payload["inference_timesteps"] = 101
    assert _post(client, payload).status_code == 422


def test_synthesize_inference_timesteps_below_range_rejected(client):
    payload = _valid_payload()
    payload["inference_timesteps"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_normalize_bad_type_rejected(client):
    # Pydantic v2 lax mode coerces "yes"/"true"/"1" to booleans — use a value
    # it cannot coerce.
    payload = _valid_payload()
    payload["normalize"] = "definitely not a bool"
    assert _post(client, payload).status_code == 422


# ---------------------------------------------------------------------------
# POST /synthesize -- reference-audio pre-flight (400).  The handler runs up to
# the audio check, so the (stubbed) runtime is faked out to skip model load.
# ---------------------------------------------------------------------------


class _StubModel:
    """Raises where the import-only engine stub would: at the generate() boundary."""

    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            "voxcpm stub: generate() is not available in tests"
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


def test_synthesize_voice_design_no_audio_passes_validation(client, fake_runtime):
    # Voice design (no reference audio) clears request validation; the stub
    # model then raises -> 500, which is the proof we got past the 422s.
    payload = {"text": "Hello there"}
    response = _post(client, payload)
    assert response.status_code == 500
    assert "not available in tests" in response.json()["detail"]


def test_synthesize_controllable_cloning_passes_validation(client, fake_runtime):
    # Reference audio only: valid 3 s clip clears validation and pre-flight,
    # then the stub model raises -> 500.
    response = _post(client, _valid_payload())
    assert response.status_code == 500
    assert "not available in tests" in response.json()["detail"]


def test_synthesize_ultimate_cloning_passes_validation(client, fake_runtime):
    # Ultimate cloning: reference audio + its transcript.
    payload = _valid_payload()
    payload["reference_text"] = "Exact transcript of the reference clip."
    response = _post(client, payload)
    assert response.status_code == 500
    assert "not available in tests" in response.json()["detail"]


def test_synthesize_undecodable_audio_rejected(client, fake_runtime):
    payload = _valid_payload()
    payload["audio_base64"] = b64(b"this is definitely not audio")
    response = _post(client, payload)
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_synthesize_too_short_audio_rejected(client, fake_runtime):
    payload = _valid_payload()
    payload["audio_base64"] = b64(make_wav_bytes(1.0))  # 1.0 s < 2.0 s minimum
    response = _post(client, payload)
    assert response.status_code == 400
    assert "2" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Engine-specific language tests
# ---------------------------------------------------------------------------


def test_synthesize_language_wellformed_nonEnglish_passes(client, fake_runtime):
    # The engine has no fixed language list (languages: null), so any
    # well-formed two-letter code clears validation; with undecodable audio
    # the handler then fails its 400 audio pre-flight -- proving we got past
    # validation (actual synthesis needs a real model + GPU).
    payload = {"text": "Hello there", "audio_base64": b64(b"this is definitely not audio"), "language": "de"}
    response = _post(client, payload)
    assert response.status_code == 400
