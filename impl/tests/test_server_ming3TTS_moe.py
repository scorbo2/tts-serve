"""Tests for ``server_ming3TTS_moe.py`` (Ming-3.0-tts 7.9B MoE "tiny").

Covers the model-free HTTP surface: /capabilities (snapshot), /health, the
landing page, request-body validation (422s), and the reference-audio
pre-flight checks (400s).

Success-path tests use a fake model (below) to pin the engine-call contract:
the ``seed``/``multilingual`` task selected from ``language``, the sampler
knobs by their engine names, the seed forwarded to ``generate`` (the engine
seeds itself — no server-side re-seeding), and the per-request temp-file
cleanup (one-shot engine: no path-keyed prompt cache).
"""

import os
import types

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import server_ming3TTS_moe as srv
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
    assert doc == load_snapshot("ming3_tts_moe_capabilities.json")


def test_capabilities_language_enum_is_codes_without_auto(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    enum = by_name["language"]["enum"]
    # Two-letter codes (docs/02) only; both cloning tasks require an
    # explicit target language, so no 'auto' sentinel is advertised.
    assert "auto" not in enum
    assert "en" in enum
    assert "ja" in enum
    # Exact set *and* order: the Literal is pinned to sorted codes because
    # pydantic's Literal cache conflates same-arg-set Literals across the
    # two Ming servers (the seed-then-multilingual task order here used to
    # leak the 0.5b server's enum into the /capabilities doc).
    assert enum == sorted(srv.LANGUAGE_CODES)
    assert doc["languages"] == sorted(srv.LANGUAGE_CODES)


def test_capabilities_reference_text_is_required(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    # ICL cloning only: conditioning on the clip and its transcript
    # together — the transcript is a hard requirement at schema level.
    assert by_name["reference_text"]["required"] is True
    assert by_name["audio_base64"]["required"] is True
    assert doc["reference_audio"]["required"] is True


def test_capabilities_sample_rate_is_44100(client):
    # The engine *enforces* 44100 via its release-contract validation.
    assert client.get("/capabilities").json()["sample_rate"] == 44100


# ---------------------------------------------------------------------------
# GET /health and the landing page
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["serverType"] == "ming3-tts-moe"
    assert body["device"] == srv.DEVICE
    assert body["model"] == srv.MODEL_NAME_OR_PATH
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


def test_synthesize_missing_reference_text_rejected(client):
    # In-context cloning only: a transcript-less request is a client error
    # at the boundary — no implicit speaker-embedding fallback exists for
    # the seed/multilingual tasks.
    payload = _valid_payload()
    del payload["reference_text"]
    response = _post(client, payload)
    assert response.status_code == 422
    assert "reference_text" in response.text


def test_synthesize_explicit_null_reference_text_rejected(client):
    # A required field sent as an explicit null is still 'no transcript'.
    payload = _valid_payload()
    payload["reference_text"] = None
    assert _post(client, payload).status_code == 422


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
    # A well-formed two-letter code neither cloning task may speak.
    payload = _valid_payload()
    payload["language"] = "xx"
    assert _post(client, payload).status_code == 422


# The shared docs/02 language contract (case, names, garbage, non-strings,
# null/empty -> 'en', no 'auto') is asserted once for every server in
# test_language_contract.py; keep only engine-specific language tests here.


def test_synthesize_seed_out_of_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 0
    assert _post(client, payload).status_code == 422
    payload["seed"] = srv.SEED_MAX + 1
    assert _post(client, payload).status_code == 422


def test_synthesize_non_positive_cfg_scale_rejected(client):
    payload = _valid_payload()
    payload["cfg_scale"] = 0.0
    assert _post(client, payload).status_code == 422


def test_model_cfg_scale_non_finite_rejected():
    # gt=0.0 passes +inf; the isfinite validator catches it.  Asserted at
    # model level (like every other server): strict JSON clients cannot
    # send Infinity, and pushing a non-strict body through HTTP merely
    # exercises FastAPI's own 422 serializer choking on echoing the inf
    # input — not our validator.
    with pytest.raises(ValidationError):
        srv.SynthesisRequest(**_valid_payload(), cfg_scale=float("inf"))
    with pytest.raises(ValidationError):
        srv.SynthesisRequest(**_valid_payload(), cfg_scale=float("nan"))


def test_synthesize_negative_sigma_rejected(client):
    payload = _valid_payload()
    payload["sigma"] = -0.1
    assert _post(client, payload).status_code == 422


def test_synthesize_negative_temperature_rejected(client):
    payload = _valid_payload()
    payload["temperature"] = -1.0
    assert _post(client, payload).status_code == 422


def test_synthesize_zero_flow_steps_rejected(client):
    payload = _valid_payload()
    payload["flow_steps"] = 0
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
    """Chainable stand-in for the engine's waveform tensor.

    Implements exactly the ``_tensor_to_numpy`` call chain
    (``detach().cpu().float()`` + ``squeeze``/``ndim``/``shape``/``numpy``),
    starting at (1, N) so the batch-dim normalization is exercised too.
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
    """Records ``generate`` kwargs; returns a fake output (waveform+stopped)."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(
            waveform=_FakeWaveform(srv.SAMPLE_RATE // 10),  # ~0.1 s of silence
            stopped=True,
        )


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


@pytest.mark.parametrize(
    "code,expected_task",
    [
        pytest.param("zh", "seed", id="chinese-seed-task"),
        pytest.param("en", "seed", id="english-seed-task"),
        pytest.param("de", "multilingual", id="german-multilingual-task"),
        pytest.param("es", "multilingual", id="spanish-multilingual-task"),
        pytest.param("fr", "multilingual", id="french-multilingual-task"),
        pytest.param("ja", "multilingual", id="japanese-multilingual-task"),
        pytest.param("ko", "multilingual", id="korean-multilingual-task"),
    ],
)
def test_task_for_language_selected_per_engine_task_spec(code, expected_task):
    # The engine's TASK_SPECS allow each language on exactly one cloning
    # task; picking the wrong task 422s *inside* the engine, so pin the
    # mapping table here.
    assert srv._task_for_language(code) == expected_task


def test_synthesize_default_language_selects_seed_task(client, fake_model):
    payload = _valid_payload()
    response = _post(client, payload)
    assert response.status_code == 200
    call = fake_model.calls[-1]
    assert call["task"] == "seed"
    # The two-letter code goes straight through as target_language; the
    # engine normalizes codes to canonical names itself (no server mapping).
    assert call["target_language"] == "en"


def test_synthesize_multilingual_language_selects_multilingual_task(client, fake_model):
    payload = _valid_payload()
    payload["language"] = "ja"
    response = _post(client, payload)
    assert response.status_code == 200
    assert fake_model.calls[-1]["task"] == "multilingual"
    assert fake_model.calls[-1]["target_language"] == "ja"


def test_synthesize_forwards_cloning_inputs_and_sampler_kwargs(client, fake_model):
    payload = _valid_payload()
    payload["cfg_scale"] = 3.0
    payload["flow_steps"] = 16
    payload["sigma"] = 0.5
    payload["temperature"] = 0.7
    response = _post(client, payload)
    assert response.status_code == 200
    call = fake_model.calls[-1]
    assert call["text"] == payload["text"]
    assert call["prompt_text"] == payload["reference_text"]
    assert call["cfg_scale"] == 3.0
    assert call["flow_steps"] == 16
    assert call["sigma"] == 0.5
    assert call["temperature"] == 0.7


def test_synthesize_seed_passed_to_engine_which_seeds_itself(client, fake_model):
    # Unlike the 0.5b server, the tiny engine calls seed_everything()
    # itself inside synthesize() — the server forwards the seed instead of
    # re-seeding the process.
    payload = _valid_payload()
    payload["seed"] = 42
    response = _post(client, payload)
    assert response.status_code == 200
    assert fake_model.calls[-1]["seed"] == 42
    assert response.json()["seed"] == 42


def test_synthesize_response_echoes_in_range_seed_when_omitted(client, fake_model):
    # (_valid_payload deliberately omits seed.)
    response = _post(client, _valid_payload())
    assert response.status_code == 200
    seed = response.json()["seed"]
    assert srv.SEED_MIN <= seed <= srv.SEED_MAX
    assert fake_model.calls[-1]["seed"] == seed


def test_synthesize_engine_output_not_stopped_response_echoes_stopped_false(client, fake_model, monkeypatch):
    # The stop-detector flag is response surface: clients need to know the
    # output may be truncated.  Force stopped=False through the fake.
    def _not_stopped(**kwargs):
        fake_model.calls.append(kwargs)
        return types.SimpleNamespace(
            waveform=_FakeWaveform(srv.SAMPLE_RATE // 10), stopped=False
        )

    monkeypatch.setattr(fake_model, "generate", _not_stopped)
    response = _post(client, _valid_payload())
    assert response.status_code == 200
    assert response.json()["stopped"] is False


def test_synthesize_reference_temp_file_deleted_after_request(client, fake_model):
    # One-shot engine: it re-encodes the reference per request with no
    # path-keyed cache, so the staged temp file must not outlive the
    # request (unlike the 0.5b server, which deliberately keeps it).
    response = _post(client, _valid_payload())
    assert response.status_code == 200
    staged_path = fake_model.calls[-1]["prompt_wav"]
    assert not os.path.exists(staged_path)


def test_synthesize_reference_temp_file_deleted_even_on_engine_failure(client, monkeypatch):
    # The cleanup lives in a finally block: a mid-synthesis engine blow-up
    # must not leak staged clips either.
    staged: list[str] = []
    real_write = srv.write_temp_audio

    def _recording_write(raw_bytes, directory):
        path = real_write(raw_bytes, directory)
        staged.append(path)
        return path

    def _explode(**kwargs):
        raise RuntimeError("simulated CUDA failure")

    monkeypatch.setattr(srv, "write_temp_audio", _recording_write)
    monkeypatch.setattr(
        srv,
        "_runtime",
        types.SimpleNamespace(
            model=types.SimpleNamespace(generate=_explode),
            sample_rate=srv.SAMPLE_RATE,
            device=srv.DEVICE,
        ),
    )
    monkeypatch.setattr(srv, "_numpy_to_wav_bytes", lambda arr, sr: b"RIFFfake")
    response = _post(client, _valid_payload())
    assert response.status_code == 500
    assert len(staged) == 1
    assert not os.path.exists(staged[0])
