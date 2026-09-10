"""Tests for ``server_indexTTS.py``.

Covers the model-free HTTP surface: /capabilities (snapshot), /health, the
landing page, request-body validation (422s — including the emotion-source
exclusivity rules and the emotion-vector array contract), and the
reference-audio pre-flight checks (400s).  Real synthesis needs the model
+ GPU and is out of scope here.
"""

import types

import pytest
from fastapi.testclient import TestClient

import server_indexTTS as srv
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
    snapshot = load_snapshot("indextts_capabilities.json")

    # ``device`` is machine-dependent: with INDEXTTS_DEVICE unset the engine
    # auto-selects CUDA when available, else CPU (and the server mirrors
    # that decision at import time).  Compare everything else, and assert
    # the device is a sane value.
    assert doc["device"] in ("cuda", "cpu", "mps", "xpu")
    doc.pop("device")
    snapshot.pop("device")
    assert doc == snapshot


def test_capabilities_uses_22khz(client):
    # The 22 kHz BigVGAN vocoder — the third distinct rate in this repo.
    doc = client.get("/capabilities").json()
    assert doc["sample_rate"] == 22050


def test_capabilities_language_enum_matches_engine_set(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    assert by_name["language"]["enum"] == list(srv.LANGUAGE_CODES)
    assert doc["languages"] == list(srv.LANGUAGE_CODES)


def test_capabilities_core_fields(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    assert by_name["text"]["required"] is True
    assert by_name["audio_base64"]["required"] is True
    assert by_name["seed"]["required"] is False
    # IndexTTS-2.5 conditions on the reference audio alone — there is
    # deliberately no transcript field (like Chatterbox).
    assert "reference_text" not in by_name


def test_capabilities_emotion_vector_is_bounded_array(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    vector = by_name["emotion_vector"]
    assert vector["type"] == "array"
    assert vector["item_type"] == "number"
    assert vector["min_items"] == 8
    assert vector["max_items"] == 8
    # One display label per component, in request order — lets a client
    # render a labeled row per component instead of 8 unlabeled boxes.
    assert vector["item_labels"] == list(srv.EMOTION_VECTOR_LABELS)
    assert vector["required"] is False
    assert by_name["emotion_audio_base64"]["required"] is False
    assert by_name["emotion_text"]["required"] is False
    assert by_name["emotion_alpha"]["default"] == 1.0


def test_capabilities_reference_audio_spec(client):
    doc = client.get("/capabilities").json()
    ref_audio = doc["reference_audio"]
    assert ref_audio["required"] is True
    assert ref_audio["min_duration_s"] == 2.0
    assert doc["watermarked"] is False


# ---------------------------------------------------------------------------
# GET /health and the landing page
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["serverType"] == "index-tts"
    assert body["device"] == srv.DEVICE_REPORT
    assert body["model"] == srv.MODEL_DIR


def test_root_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "IndexTTS" in response.text
    assert "/synthesize" in response.text


# ---------------------------------------------------------------------------
# POST /synthesize — request-body validation (422)
# ---------------------------------------------------------------------------


def _post(client, payload):
    return client.post("/synthesize", json=payload)


def _valid_payload():
    # 3 s of 22.05 kHz audio — clears the 2 s minimum with room to spare.
    return {
        "text": "Hello there",
        "audio_base64": b64(make_wav_bytes(3.0, sr=22050)),
    }


def test_synthesize_unknown_field_rejected(client):
    payload = _valid_payload()
    payload["bogus_field"] = 1
    assert _post(client, payload).status_code == 422


def test_synthesize_reference_text_rejected(client):
    # The transcript field is a deliberate gap (audio-only conditioning) —
    # it must fail loudly as an unknown field, not be silently ignored.
    payload = _valid_payload()
    payload["reference_text"] = "A transcript nobody asked for."
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


# The shared docs/02 language contract (case, names, garbage, non-strings,
# null/empty -> 'en', and the 'auto' sentinel per declaration) is asserted
# once for every server in test_language_contract.py; keep only
# engine-specific language tests here.


def test_synthesize_unsupported_language_rejected(client):
    # 'fr' is a valid two-letter code but not one IndexTTS-2.5 advertises —
    # the engine would silently degrade to its 'common' vocabulary, so the
    # server rejects it at the boundary instead.
    payload = _valid_payload()
    payload["language"] = "fr"
    assert _post(client, payload).status_code == 422


def test_synthesize_seed_below_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_seed_above_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 1001
    assert _post(client, payload).status_code == 422


def test_synthesize_emotion_alpha_below_range_rejected(client):
    payload = _valid_payload()
    payload["emotion_alpha"] = -0.1
    assert _post(client, payload).status_code == 422


def test_synthesize_emotion_alpha_above_range_rejected(client):
    payload = _valid_payload()
    payload["emotion_alpha"] = 1.5
    assert _post(client, payload).status_code == 422


def test_synthesize_duration_factor_below_range_rejected(client):
    payload = _valid_payload()
    payload["duration_factor"] = 0.05
    assert _post(client, payload).status_code == 422


def test_synthesize_duration_factor_above_range_rejected(client):
    payload = _valid_payload()
    payload["duration_factor"] = 3.5
    assert _post(client, payload).status_code == 422


def test_synthesize_temperature_above_range_rejected(client):
    payload = _valid_payload()
    payload["temperature"] = 2.5
    assert _post(client, payload).status_code == 422


def test_synthesize_top_p_above_range_rejected(client):
    payload = _valid_payload()
    payload["top_p"] = 1.5
    assert _post(client, payload).status_code == 422


def test_synthesize_top_k_out_of_range_rejected(client):
    payload = _valid_payload()
    payload["top_k"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_repetition_penalty_below_range_rejected(client):
    payload = _valid_payload()
    payload["repetition_penalty"] = 0.5
    assert _post(client, payload).status_code == 422


def test_synthesize_whitespace_emotion_text_rejected(client):
    payload = _valid_payload()
    payload["emotion_text"] = "   "
    assert _post(client, payload).status_code == 422


def test_synthesize_emotion_vector_too_short_rejected(client):
    payload = _valid_payload()
    payload["emotion_vector"] = [0.1] * 7
    assert _post(client, payload).status_code == 422


def test_synthesize_emotion_vector_too_long_rejected(client):
    payload = _valid_payload()
    payload["emotion_vector"] = [0.1] * 9
    assert _post(client, payload).status_code == 422


def test_synthesize_emotion_vector_component_above_range_rejected(client):
    payload = _valid_payload()
    payload["emotion_vector"] = [0.0] * 7 + [1.5]
    assert _post(client, payload).status_code == 422


def test_synthesize_emotion_vector_component_below_range_rejected(client):
    payload = _valid_payload()
    payload["emotion_vector"] = [-0.1] + [0.0] * 7
    assert _post(client, payload).status_code == 422


def test_synthesize_audio_plus_vector_emotion_sources_rejected(client):
    payload = _valid_payload()
    payload["emotion_audio_base64"] = b64(make_wav_bytes(3.0, sr=22050))
    payload["emotion_vector"] = [0.5] * 8
    assert _post(client, payload).status_code == 422


def test_synthesize_vector_plus_text_emotion_sources_rejected(client):
    payload = _valid_payload()
    payload["emotion_vector"] = [0.5] * 8
    payload["emotion_text"] = "excited and cheerful"
    assert _post(client, payload).status_code == 422


def test_synthesize_audio_plus_text_emotion_sources_rejected(client):
    payload = _valid_payload()
    payload["emotion_audio_base64"] = b64(make_wav_bytes(3.0, sr=22050))
    payload["emotion_text"] = "excited and cheerful"
    assert _post(client, payload).status_code == 422


# ---------------------------------------------------------------------------
# POST /synthesize — pre-flight (400)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_runtime(monkeypatch):
    # The 400 pre-flight checks all run before the model is touched, so a
    # bare namespace is enough for _get_runtime() to skip model loading.
    monkeypatch.setattr(
        srv, "_runtime", types.SimpleNamespace(model=None, device="cpu")
    )


def test_synthesize_undecodable_audio_rejected(client, fake_runtime):
    payload = _valid_payload()
    payload["audio_base64"] = b64(b"this is definitely not audio")
    response = _post(client, payload)
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_synthesize_too_short_audio_rejected(client, fake_runtime):
    payload = _valid_payload()
    payload["audio_base64"] = b64(make_wav_bytes(0.5, sr=22050))  # 0.5 s < 2.0 s
    response = _post(client, payload)
    assert response.status_code == 400
    assert "2" in response.json()["detail"]


def test_synthesize_undecodable_emotion_audio_rejected(client, fake_runtime):
    payload = _valid_payload()
    payload["emotion_audio_base64"] = b64(b"definitely not audio either")
    response = _post(client, payload)
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_synthesize_too_short_emotion_audio_rejected(client, fake_runtime):
    payload = _valid_payload()
    payload["emotion_audio_base64"] = b64(make_wav_bytes(0.5, sr=22050))
    response = _post(client, payload)
    assert response.status_code == 400
    assert "2" in response.json()["detail"]


def test_synthesize_emotion_text_without_qwen_emo_rejected(client, fake_runtime):
    # With the default configuration (INDEXTTS_USE_QWEN_EMO off) the
    # QwenEmotion model is not loaded, so emotion_text is a configuration
    # 400 — rejected before any audio work.
    payload = _valid_payload()
    payload["emotion_text"] = "excited and cheerful"
    response = _post(client, payload)
    assert response.status_code == 400
    assert "INDEXTTS_USE_QWEN_EMO" in response.json()["detail"]
