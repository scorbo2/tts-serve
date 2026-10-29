"""Tests for ``server_ming3TTS.py`` (Ming-3.0-tts 0.5b dense).

Covers the model-free HTTP surface: /capabilities (snapshot), /health, the
landing page, request-body validation (422s), and the reference-audio
pre-flight checks (400s).

A handful of /synthesize success-path tests use a fake model (below) purely
to pin the argument forwarding to ``MingTTS.synthesize`` — the task name,
the speaker-embedding flag, the sampler knobs, and the
transcript-omitted -> ``prompt_text=None`` fallback — plus the in-lock
re-seeding and the content-hashed prompt staging.  The engine call itself
still needs the model + GPU and is out of scope here.
"""

import os
import types

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import server_ming3TTS as srv
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
    assert doc == load_snapshot("ming3_tts_capabilities.json")


def test_capabilities_language_enum_is_codes_without_auto(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    enum = by_name["language"]["enum"]
    # Two-letter codes (docs/02) only: the engine's TTS task has no
    # auto-detection sentinel, so 'auto' must not be advertised.
    assert "auto" not in enum
    assert "en" in enum
    assert "zh" in enum
    # Exact set *and* order: the Literal is pinned to sorted codes because
    # pydantic's Literal cache conflates same-arg-set Literals across the
    # two Ming servers (import-order-dependent enum order otherwise).
    assert enum == sorted(srv.LANGUAGE_CODES)
    assert doc["languages"] == sorted(srv.LANGUAGE_CODES)


def test_capabilities_required_fields(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    assert by_name["text"]["required"] is True
    assert by_name["audio_base64"]["required"] is True
    # The transcript refines the clone (in-context) but is optional: an
    # omitted transcript is the documented speaker-embedding-only fallback.
    assert by_name["reference_text"]["required"] is False
    assert by_name["language"]["required"] is False
    assert by_name["seed"]["required"] is False
    assert doc["reference_audio"]["required"] is True


def test_capabilities_sample_rate_is_44100(client):
    # 44.1 kHz is the released 0.5b checkpoint's audio-VAE rate.
    assert client.get("/capabilities").json()["sample_rate"] == 44100


# ---------------------------------------------------------------------------
# GET /health and the landing page
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["serverType"] == "ming3-tts"
    assert body["device"] == srv.DEVICE
    assert body["model"] == srv.MODEL_NAME_OR_PATH
    # Pinning to srv.RELEASE_VERSION (not a literal) keeps this release-proof
    # while still catching a server that forgot to wire the shared constant in.
    assert body["version"] == srv.RELEASE_VERSION


def test_root_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Ming-3.0-tts" in response.text
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


def test_synthesize_omitted_reference_text_accepted(client, fake_model):
    # Omitting the transcript is the documented mode switch to speaker-
    # embedding-only cloning — forwarded to the engine as an explicit None.
    payload = _valid_payload()
    del payload["reference_text"]
    response = _post(client, payload)
    assert response.status_code == 200
    assert fake_model.calls[-1]["prompt_text"] is None


def test_synthesize_explicit_null_reference_text_accepted(client, fake_model):
    # An explicit null means 'no transcript' — it must validate like an
    # omitted field, not 500 inside the field validator (regression:
    # Pydantic runs after-mode validators for an explicitly-provided None).
    payload = _valid_payload()
    payload["reference_text"] = None
    response = _post(client, payload)
    assert response.status_code == 200
    assert fake_model.calls[-1]["prompt_text"] is None


def test_synthesize_whitespace_reference_text_rejected(client):
    payload = _valid_payload()
    payload["reference_text"] = "   "
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
    # A well-formed two-letter code the model does not support: 422 at the
    # boundary (the engine has no fallback for unknown languages).
    payload = _valid_payload()
    payload["language"] = "xx"
    assert _post(client, payload).status_code == 422


# The shared docs/02 language contract (case, names, garbage, non-strings,
# null/empty -> 'en', no 'auto') is asserted once for every server in
# test_language_contract.py; keep only engine-specific language tests here.


def test_synthesize_supported_language_code_forwarded_verbatim(client, fake_model):
    # Unlike the Qwen3-TTS servers, this engine's language_id accepts the
    # two-letter codes directly — the server must NOT map them to names.
    payload = _valid_payload()
    payload["language"] = "zh"
    response = _post(client, payload)
    assert response.status_code == 200
    assert fake_model.calls[-1]["language_id"] == "zh"


def test_synthesize_seed_out_of_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 0
    assert _post(client, payload).status_code == 422
    payload["seed"] = srv.SEED_MAX + 1
    assert _post(client, payload).status_code == 422


def test_synthesize_non_positive_cfg_rejected(client):
    payload = _valid_payload()
    payload["cfg"] = 0.0
    assert _post(client, payload).status_code == 422


def test_model_cfg_non_finite_rejected():
    # gt=0.0 passes +inf; the isfinite validator catches it.  Asserted at
    # model level (like every other server): strict JSON clients cannot
    # send Infinity, and pushing a non-strict body through HTTP merely
    # exercises FastAPI's own 422 serializer choking on echoing the inf
    # input — not our validator.
    with pytest.raises(ValidationError):
        srv.SynthesisRequest(**_valid_payload(), cfg=float("inf"))
    with pytest.raises(ValidationError):
        srv.SynthesisRequest(**_valid_payload(), cfg=float("nan"))


def test_synthesize_negative_sigma_rejected(client):
    payload = _valid_payload()
    payload["sigma"] = -0.1
    assert _post(client, payload).status_code == 422


def test_synthesize_negative_temperature_rejected(client):
    payload = _valid_payload()
    payload["temperature"] = -1.0
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


class _FakeWaveform:
    """Chainable stand-in for the engine's (1, N) waveform tensor.

    Implements exactly the ``_tensor_to_numpy`` call chain
    (``detach().cpu().float()`` + ``squeeze``/``ndim``/``shape``/``numpy``)
    so the conversion helper itself is exercised, not stubbed away.
    """

    def __init__(self, n_samples: int) -> None:
        self._samples = [0.0] * n_samples
        self.ndim = 2
        self.shape = (1, n_samples)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def squeeze(self, dim=0):
        flat = _FakeWaveform(0)
        flat._samples = self._samples
        flat.ndim = 1
        flat.shape = (len(self._samples),)
        return flat

    def numpy(self):
        return self._samples


class _FakeModel:
    """Records ``synthesize`` kwargs; returns a silent fake waveform."""

    def __init__(self):
        self.calls: list[dict] = []

    def synthesize(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeWaveform(srv.SAMPLE_RATE // 10)  # ~0.1 s of silence


@pytest.fixture
def fake_model(monkeypatch):
    """Install a recording model and a no-op WAV encoder.

    The stub numpy/soundfile refuse ``clip()``/``write()`` by design (real
    synthesis needs the model + GPU), so the success-path tests stand in for
    the encoder: they assert on what the server sent to the engine.
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


# ---------------------------------------------------------------------------
# POST /synthesize — engine-call pinning (fake model)
# ---------------------------------------------------------------------------


def test_synthesize_forwards_task_and_sampling_kwargs_to_engine(client, fake_model):
    payload = _valid_payload()
    payload["cfg"] = 1.5
    payload["sigma"] = 0.5
    payload["temperature"] = 0.7
    response = _post(client, payload)
    assert response.status_code == 200
    call = fake_model.calls[-1]
    # Voice cloning only: the engine's TTS task, speaker embedding always on
    # (it is what makes the mode a *clone* at all).
    assert call["task"] == "TTS"
    assert call["use_spk_emb"] is True
    assert call["text"] == payload["text"]
    assert call["prompt_text"] == payload["reference_text"]
    assert call["cfg"] == 1.5
    assert call["sigma"] == 0.5
    assert call["temperature"] == 0.7


def test_synthesize_reseeds_pipeline_with_request_seed(client, fake_model, monkeypatch):
    # The prompt-VAE posterior sampling consumes the global RNG, so the
    # server re-seeds inside the synthesis lock; pin that the request's
    # seed is what reaches it.
    seeded: list[int] = []
    monkeypatch.setattr(srv, "seed_everything", seeded.append)
    payload = _valid_payload()
    payload["seed"] = 7
    response = _post(client, payload)
    assert response.status_code == 200
    assert seeded == [7]
    assert response.json()["seed"] == 7


def test_seed_everything_cudaUnavailable_skipsCudaSeeding(monkeypatch):
    # GIVEN CUDA is unavailable: this server explicitly supports 'cpu' and
    # 'mps' via MING3_TTS_DEVICE, and on a torch build without CUDA the
    # torch.cuda accessors raise outright (the test stub's no-op
    # manual_seed_all hides this, so the convention must be pinned here —
    # an unguarded call would 500 every /synthesize, since seeding runs
    # per request inside the synthesis lock):
    def explode(seed: int) -> None:
        raise AssertionError("Torch not compiled with CUDA enabled")

    monkeypatch.setattr(srv.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(srv.torch.cuda, "manual_seed_all", explode)
    # WHEN the pipeline is seeded,
    # THEN the CUDA branch must be skipped and seeding must succeed:
    srv.seed_everything(7)


def test_seed_everything_cudaAvailable_seedsAllCudaDevices(monkeypatch):
    # GIVEN CUDA is available,
    seeded: list[int] = []
    monkeypatch.setattr(srv.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(srv.torch.cuda, "manual_seed_all", seeded.append)
    # WHEN the pipeline is seeded with 9,
    # THEN every CUDA device must get that seed (the guard must not
    # silently skip real GPU machines either):
    srv.seed_everything(9)
    assert seeded == [9]


def test_synthesize_response_echoes_in_range_seed_when_omitted(client, fake_model):
    # (_valid_payload deliberately omits seed.)
    response = _post(client, _valid_payload())
    assert response.status_code == 200
    assert srv.SEED_MIN <= response.json()["seed"] <= srv.SEED_MAX


def test_synthesize_stages_reference_audio_content_hashed_and_kept(client, fake_model):
    # The engine caches encoded prompts per path, so the same clip must land
    # at the same staged path across requests — and the file must survive
    # the request (deletion would race with a queued duplicate and defeat
    # the cache).
    payload = _valid_payload()
    first = _post(client, payload)
    second = _post(client, payload)
    assert first.status_code == second.status_code == 200
    first_path = fake_model.calls[-2]["prompt_wav_path"]
    second_path = fake_model.calls[-1]["prompt_wav_path"]
    assert first_path == second_path
    assert os.path.exists(first_path)
    # Leave the staging dir as we found it.
    os.remove(first_path)
