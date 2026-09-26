"""Tests for ``server_breezeBlue.py``.

Covers the model-free HTTP surface: /capabilities (snapshot), /health, the
landing page, request-body validation (422s), and the reference-audio
pre-flight checks (400s).  A handful of /synthesize success-path tests use
a fake streaming runtime (below) purely to pin the request dict, template,
and seed the server forwards to the engine; actual synthesis still needs
the model + GPU.
"""

import base64
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import server_breezeBlue as srv
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
    assert doc == load_snapshot("breeze_blue_capabilities.json")


def test_capabilities_required_fields(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    # Both supported modes (voice clone, voice direction) are
    # reference-based, so the clip and its exact transcript are hard
    # requirements; the engine's voice-design mode is deliberately absent.
    assert by_name["text"]["required"] is True
    assert by_name["audio_base64"]["required"] is True
    assert by_name["reference_text"]["required"] is True
    assert by_name["language"]["required"] is False
    assert by_name["seed"]["required"] is False
    assert by_name["instruction"]["required"] is False
    assert by_name["cfg_scale"]["required"] is False


def test_capabilities_language_and_reference_audio(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    # The engine has no language parameter (bilingual auto-detection), so
    # no fixed code list is advertised — the no-support case of docs/02.
    assert doc["languages"] is None
    assert by_name["language"]["default"] == "en"
    assert by_name["language"]["enum"] is None
    ref = doc["reference_audio"]
    assert ref["required"] is True
    assert ref["min_duration_s"] == srv.MIN_PROMPT_DURATION_S
    assert doc["sample_rate"] == srv.SAMPLE_RATE
    assert doc["watermarked"] is False
    assert by_name["cfg_scale"]["default"] == 1.0


# ---------------------------------------------------------------------------
# GET /health and the landing page
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["serverType"] == "breeze-tts-2"
    assert body["device"] == srv.DEVICE
    assert body["model"] == srv.MODEL_NAME_OR_PATH


def test_root_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Breeze TTS 2" in response.text
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
    # Unlike most tts-serve engines, the transcript is a hard requirement:
    # both supported modes condition on the clip and its transcript.
    payload = _valid_payload()
    del payload["reference_text"]
    assert _post(client, payload).status_code == 422


def test_synthesize_missing_audio_rejected(client):
    # No voice-design mode here: every supported mode needs the reference
    # clip, so omitting it is invalid (not a mode switch).
    payload = _valid_payload()
    del payload["audio_base64"]
    assert _post(client, payload).status_code == 422


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


def test_synthesize_empty_reference_text_rejected(client):
    payload = _valid_payload()
    payload["reference_text"] = ""
    assert _post(client, payload).status_code == 422


def test_synthesize_whitespace_reference_text_rejected(client):
    payload = _valid_payload()
    payload["reference_text"] = "   "
    assert _post(client, payload).status_code == 422


def test_synthesize_whitespace_instruction_rejected(client):
    # A provided-but-blank instruction would silently switch the engine
    # into plain-clone mode; reject it loudly (explicit null is the way to
    # say 'no instruction').
    payload = _valid_payload()
    payload["instruction"] = "   "
    assert _post(client, payload).status_code == 422


def test_synthesize_cfg_scale_zero_rejected(client):
    payload = _valid_payload()
    payload["cfg_scale"] = 0.0
    assert _post(client, payload).status_code == 422


def test_synthesize_cfg_scale_negative_rejected(client):
    payload = _valid_payload()
    payload["cfg_scale"] = -1.0
    assert _post(client, payload).status_code == 422


def test_synthesize_cfgScaleNonDefault_withoutInstruction_rejected(client):
    # The engine's CFG branch only exists in voice-direction mode
    # (ref_clone_tata has no negative branch), so a non-default scale
    # without an instruction is a client error — 422 at the boundary,
    # not a 500 from deep inside prepare_inputs().
    payload = _valid_payload()
    payload["cfg_scale"] = 4.0
    response = _post(client, payload)
    assert response.status_code == 422
    assert "instruction" in str(response.json()["detail"]).lower()


def test_model_cfg_scale_non_finite_rejected():
    # +inf passes the gt=0.0 bound, so the explicit finiteness check must
    # catch it (the engine's own CLI rejects any non-finite scale).  NaN is
    # caught by the bound itself.  Tested at the model level: over HTTP the
    # 422 would echo the offending input, and FastAPI cannot JSON-encode a
    # non-finite value (and a strict-JSON client could not send one).
    with pytest.raises(ValidationError):
        srv.SynthesisRequest(**_valid_payload(), cfg_scale=float("inf"))
    with pytest.raises(ValidationError):
        srv.SynthesisRequest(**_valid_payload(), cfg_scale=float("nan"))


def test_synthesize_seed_below_min_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_seed_above_max_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 1001
    assert _post(client, payload).status_code == 422


@pytest.mark.parametrize("code", ["fr", "de"])
def test_synthesize_language_wellFormedCode_passesValidation(
    client, fake_runtime, code
):
    # The engine auto-detects from text (no code table), so any well-formed
    # code clears validation — with undecodable audio the handler then
    # fails the 400 audio pre-flight, proving validation was passed.
    payload = _valid_payload()
    payload["language"] = code
    payload["audio_base64"] = b64(b"this is definitely not audio")
    assert _post(client, payload).status_code == 400


# The shared docs/02 language contract (case, names, garbage, non-strings,
# null/empty -> 'en', the 'auto' sentinel) is asserted once for every
# server in test_language_contract.py; keep only engine-specific language
# tests here.


# ---------------------------------------------------------------------------
# POST /synthesize — reference-audio pre-flight (400)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_runtime(monkeypatch):
    # The pre-flight never reads runtime attributes; a bare placeholder
    # stands in for the loaded engine.
    monkeypatch.setattr(
        srv,
        "_runtime",
        types.SimpleNamespace(sample_rate=srv.SAMPLE_RATE, device=srv.DEVICE),
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


# ---------------------------------------------------------------------------
# POST /synthesize — success path with a fake streaming runtime
# ---------------------------------------------------------------------------


class _FakeChunk:
    """Stands in for models.fast_streaming.FastStreamingChunk."""

    def __init__(self, audio):
        self.audio = audio
        self.sample_rate = srv.SAMPLE_RATE
        self.codec_frames = 2
        self.is_final = False


class _FakeStreaming:
    """Records iter_audio_chunks calls; yields the configured audio once."""

    def __init__(self, audio):
        self.audio = audio  # None => yield no chunks (degenerate generation)
        self.calls: list[dict] = []

    def iter_audio_chunks(self, inputs, *, request_id=None, seed=None):
        self.calls.append({"inputs": inputs, "request_id": request_id, "seed": seed})
        if self.audio is None:
            return iter(())
        yield _FakeChunk(self.audio)


def _install_fake_streaming(monkeypatch, audio):
    """Install a recording streaming runtime + stand-ins for the engine's
    template/input functions (the stubs raise by design)."""
    streaming = _FakeStreaming(audio=audio)
    monkeypatch.setattr(
        srv,
        "_runtime",
        types.SimpleNamespace(
            tokenizer=object(),
            model=object(),
            audio_tokenizer=object(),
            streaming=streaming,
            sample_rate=srv.SAMPLE_RATE,
            device=srv.DEVICE,
        ),
    )

    captured = {}

    def _select_template_name(request):
        # Mirror the engine's selection for the two reference-based
        # templates (the reference is always present on this server).
        captured["template"] = (
            "ref_edit_tata" if request.get("instruction") else "ref_clone_tata"
        )
        return captured["template"]

    def _get_template(name):
        return name

    def _prepare_inputs(tokenizer, audio_tokenizer, model, requests, template, **kwargs):
        captured["requests"] = [dict(r) for r in requests]
        captured["kwargs"] = kwargs
        return {"fake": "inputs"}

    monkeypatch.setattr(srv, "select_template_name", _select_template_name)
    monkeypatch.setattr(srv, "get_template", _get_template)
    monkeypatch.setattr(srv, "prepare_inputs", _prepare_inputs)
    # The stub numpy/soundfile refuse clip()/write() by design; stand in
    # for the WAV encoder (real synthesis needs the model + GPU).
    monkeypatch.setattr(srv, "_numpy_to_wav_bytes", lambda arr, sr: b"RIFFfake")
    return types.SimpleNamespace(streaming=streaming, captured=captured)


@pytest.fixture
def fake_streaming(monkeypatch):
    # ~0.1 s of silence as a plain Python list: len() is all the endpoint's
    # RTF math needs, and the WAV encoder is monkeypatched away.
    return _install_fake_streaming(monkeypatch, [0.0] * (srv.SAMPLE_RATE // 10))


def test_synthesize_voiceClone_forwardedToEngine(client, fake_streaming):
    payload = _valid_payload()
    response = _post(client, payload)
    assert response.status_code == 200
    body = response.json()

    request = fake_streaming.captured["requests"][0]
    # Plain cloning: no instruction key at all — the engine selects
    # ref_clone_tata from the fields present.
    assert "instruction" not in request
    assert request["text"] == payload["text"]
    assert request["speaker"] == "S0"
    assert request["ref_text"] == payload["reference_text"]
    assert fake_streaming.captured["template"] == "ref_clone_tata"
    # Guidance mirrors the engine defaults; the ref/ins CFG branches are
    # not used by this server.
    assert fake_streaming.captured["kwargs"]["guidance_scale"] == 1.0
    assert fake_streaming.captured["kwargs"]["guidance_scale_ref"] is None
    assert fake_streaming.captured["kwargs"]["guidance_scale_ins"] is None
    # The seed and request id travel to the sampling loop, and the
    # response echoes them.
    call = fake_streaming.streaming.calls[0]
    assert call["seed"] == body["seed"]
    assert call["request_id"] == body["fid"]
    # Core response shape.
    assert body["sample_rate"] == srv.SAMPLE_RATE
    assert body["audio_base64"] == base64.b64encode(b"RIFFfake").decode("ascii")
    assert body["time_used"] >= 0
    assert body["rtf"] is not None  # 0.1 s of fake audio => computable RTF


def test_synthesize_voiceDirection_instructionForwarded(client, fake_streaming):
    payload = _valid_payload()
    payload["instruction"] = "  Speak slowly with a restrained, serious tone.  "
    payload["cfg_scale"] = 4.0
    response = _post(client, payload)
    assert response.status_code == 200

    request = fake_streaming.captured["requests"][0]
    # Stripped, like the engine's own API; its presence switches the
    # template to voice direction.
    assert request["instruction"] == "Speak slowly with a restrained, serious tone."
    assert fake_streaming.captured["template"] == "ref_edit_tata"
    assert fake_streaming.captured["kwargs"]["guidance_scale"] == 4.0


def test_synthesize_nullInstruction_clones(client, fake_streaming):
    # An explicit null means 'no instruction' — it must validate like an
    # omitted field, not flip the engine into direction mode.
    payload = _valid_payload()
    payload["instruction"] = None
    response = _post(client, payload)
    assert response.status_code == 200
    assert "instruction" not in fake_streaming.captured["requests"][0]
    assert fake_streaming.captured["template"] == "ref_clone_tata"


def test_synthesize_explicitSeed_forwardedToEngine(client, fake_streaming):
    payload = _valid_payload()
    payload["seed"] = 77
    response = _post(client, payload)
    assert response.status_code == 200
    assert response.json()["seed"] == 77
    assert fake_streaming.streaming.calls[0]["seed"] == 77


def test_synthesize_referenceFile_cleanedUpAfterRequest(client, fake_streaming):
    # One-shot engine (no path-keyed cache): the staged temp file must be
    # gone once the request completes.
    payload = _valid_payload()
    _post(client, payload)
    staged_path = fake_streaming.captured["requests"][0]["ref_audio_path"]
    assert not Path(staged_path).exists()


def test_synthesize_no_generated_audio_returns_500(client, monkeypatch):
    # Same edge the Qwen3-TTS MLX suite pins: a degenerate generation that
    # yields no chunks must be a clear 500, not an empty audio payload.
    _install_fake_streaming(monkeypatch, audio=None)

    response = _post(client, _valid_payload())
    assert response.status_code == 500
    assert "produced no audio" in response.json()["detail"].lower()
