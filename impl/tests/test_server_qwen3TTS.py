"""Tests for ``server_qwen3TTS.py``.

Covers the model-free HTTP surface: /capabilities (snapshot), /health, the
landing page, request-body validation (422s), and the reference-audio
pre-flight checks (400s).  Real synthesis needs the model + GPU and is out
of scope here.
"""

import types

import pytest
from fastapi.testclient import TestClient

from tts_engine_common.devices import parse_device

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
    # The tts-serve release version comes from the single shared constant;
    # pinning to srv.RELEASE_VERSION (not a literal) keeps this release-proof
    # while still catching a server that forgot to wire it in.
    assert body["version"] == srv.RELEASE_VERSION


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


# ---------------------------------------------------------------------------
# Device selection (docs/01 D8).
#
# The grammar and the shared message are asserted for every server by
# impl/tests/test_device_contract.py; what is engine-specific here is that the
# value lands on transformers' `device_map=`, whose documented grammar
# includes an indexed CUDA device -- so this suite pins that the ordinal
# actually reaches the loader, and that an ordinal the box does not have is
# refused before the loader is touched at all.
# ---------------------------------------------------------------------------


def _pin_cuda(monkeypatch, available: bool, count: int) -> None:
    """Pin the CUDA probes so these tests never inherit the host's GPU."""
    monkeypatch.setattr(srv.torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(srv.torch.cuda, "device_count", lambda: count)


class _RecordingLoader:
    """Stand-in for Qwen3TTSModel: records the load kwargs, downloads nothing."""

    def __init__(self, calls: list):
        self._calls = calls

    def from_pretrained(self, *args, **kwargs):
        self._calls.append(kwargs)
        return types.SimpleNamespace()


def _pin_device(monkeypatch, value: str) -> None:
    """Pin both halves of a configured device: the string the engine is handed
    and the spec the guard reads (the module builds that pair at import time)."""
    monkeypatch.setattr(srv, "DEVICE", value)
    monkeypatch.setattr(
        srv,
        "DEVICE_SPEC",
        parse_device(value, env_var=srv.DEVICE_ENV, families=srv.DEVICE_FAMILIES),
    )


def test_getRuntime_indexedCudaWithinDeviceCount_forwardsTheOrdinalToTheEngine(monkeypatch):
    # GIVEN a two-GPU box and QWEN3TTS_DEVICE='cuda:1' -- the second GPU used to
    # be unreachable here because the module tested membership in
    # ('cuda', 'mps', 'cpu'):
    monkeypatch.setattr(srv, "_runtime", None)
    _pin_device(monkeypatch, "cuda:1")
    _pin_cuda(monkeypatch, available=True, count=2)
    calls: list = []
    monkeypatch.setattr(srv, "Qwen3TTSModel", _RecordingLoader(calls))

    # WHEN the runtime is loaded,
    runtime = srv._get_runtime()

    # THEN the loader was asked for GPU 1, not for 'cuda' with the index
    # quietly dropped:
    assert calls, "the loader was never called"
    assert calls[0]["device_map"] == "cuda:1"
    assert runtime.device == "cuda:1"


def test_getRuntime_indexedCudaPastTheLastGpu_failsBeforeTouchingTheCheckpoint(monkeypatch):
    # GIVEN a one-GPU box and a request for GPU 9, plus a loader that records
    # (and refuses) any checkpoint access:
    monkeypatch.setattr(srv, "_runtime", None)
    _pin_device(monkeypatch, "cuda:9")
    _pin_cuda(monkeypatch, available=True, count=1)

    class _NeverTouchTheCheckpoint:
        def from_pretrained(self, *args, **kwargs):
            raise AssertionError(
                "the device guard must reject this box before the checkpoint "
                "is downloaded and loaded only to fail while placing weights"
            )

    monkeypatch.setattr(srv, "Qwen3TTSModel", _NeverTouchTheCheckpoint())

    # WHEN the runtime is requested, THEN the shared message names the ordinal
    # and what the box has (wording pinned once for all servers by the device
    # contract test),
    with pytest.raises(RuntimeError, match="asks for GPU 9, but this box exposes 1 CUDA device"):
        srv._get_runtime()

    # AND no half-built runtime is left behind to fail every later request.
    assert srv._runtime is None
