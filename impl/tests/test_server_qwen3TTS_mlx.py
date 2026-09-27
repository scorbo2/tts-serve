"""Tests for ``server_qwen3TTS_mlx.py``.

Covers the model-free HTTP surface: /capabilities (snapshot), /health, the
landing page, request-body validation (422s), synthesis edge cases (500s),
the reference-audio pre-flight checks (400s), and the native-streaming
POST /stream endpoint (request validation, ICL/x-vector dispatch, chunk
transport, first-chunk prefetch, cleanup/cancellation). Real synthesis needs
mlx-audio + Apple Silicon and is out of scope here -- this suite must never
import MLX or load the model.
"""

import asyncio
import json
import struct
import threading
import types

import pytest
import server_qwen3TTS_mlx as srv
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect
from helpers import b64, load_snapshot, make_wav_bytes


@pytest.fixture(scope="module")
def client():
    # Deliberately NOT a context manager: entering it would run the FastAPI
    # lifespan, which loads the (stubbed) model. Not needed for these tests.
    return TestClient(srv.app)


@pytest.fixture
def fake_runtime(monkeypatch):
    """Install a model-free runtime and bypass full reference-audio decoding.

    The test soundfile stub supports header inspection via sf.info(), which is
    enough for the reference-audio pre-flight checks, but deliberately does not
    implement sf.read(). Synthesis-edge-case tests need to get past that decode
    step without depending on the real soundfile package or MLX.
    """
    runtime = types.SimpleNamespace(
        model=None,
        sample_rate=srv.SAMPLE_RATE,
        device=srv.DEVICE,
    )

    monkeypatch.setattr(srv, "_runtime", runtime)
    monkeypatch.setattr(
        srv,
        "_decode_wav",
        lambda raw: (object(), srv.SAMPLE_RATE),
    )
    monkeypatch.setattr(
        srv,
        "_prepare_ref_audio",
        lambda *args, **kwargs: object(),
    )

    return runtime


# ---------------------------------------------------------------------------
# GET /capabilities
# ---------------------------------------------------------------------------


def test_capabilities_matches_snapshot(client):
    response = client.get("/capabilities")
    assert response.status_code == 200
    doc = response.json()
    assert doc == load_snapshot("qwen3_mlx_capabilities.json")


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
    # ICL cloning only: without a transcript the request is invalid at the
    # boundary, not a fallback to some other cloning mode.
    assert by_name["reference_text"]["required"] is True
    assert by_name["language"]["required"] is False
    assert by_name["seed"]["required"] is False


def test_capabilities_device_and_sample_rate(client):
    doc = client.get("/capabilities").json()
    assert doc["device"] == "mlx"
    assert doc["sample_rate"] == 24000
    assert doc["engine"] == "qwen3-tts-mlx"


def test_capabilities_streaming_block(client):
    doc = client.get("/capabilities").json()

    assert "streaming" in doc
    streaming = doc["streaming"]
    assert streaming["endpoint"] == "/stream"
    assert streaming["format"] == "pcm_f32le"
    assert streaming["sample_rate"] == 24000
    assert streaming["channels"] == 1
    assert set(streaming["voice_conditioning"]) == {"icl", "x_vector"}

    # The existing top-level endpoint is /synthesize's, unchanged by adding
    # a streaming capability.
    assert doc["endpoint"] == "/synthesize"


def test_capabilities_nonstreaming_engine_omits_streaming_key(client):
    # Cross-engine proof (not just "this engine has it"): a sibling engine
    # that does not stream must see ITS capabilities response completely
    # unaffected -- the `streaming` key must be absent, not null/false.
    import server_qwen3TTS as sibling_srv

    doc = TestClient(sibling_srv.app).get("/capabilities").json()
    assert "streaming" not in doc


# ---------------------------------------------------------------------------
# GET /health and the landing page
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["serverType"] == "Qwen3-TTS-MLX"
    assert body["device"] == "mlx"
    assert body["model"] == srv.MODEL_NAME_OR_PATH


def test_root_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Qwen3-TTS MLX" in response.text
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
    # ICL cloning only: without a transcript the request is invalid at the
    # boundary, not a 400 or a fallback to some other cloning mode.
    payload = _valid_payload()
    del payload["reference_text"]
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


def test_synthesize_unsupported_language_rejected(client):
    # A valid code that the Base checkpoint does not support.
    payload = _valid_payload()
    payload["language"] = "xx"
    assert _post(client, payload).status_code == 422


# The shared docs/02 language contract (case, names, garbage, non-strings,
# null/empty -> 'en') is asserted once for every server in
# test_language_contract.py; keep only engine-specific language tests here.


def test_synthesize_seed_below_min_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_seed_above_max_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 1001
    assert _post(client, payload).status_code == 422


# ---------------------------------------------------------------------------
# POST /synthesize — synthesis edge cases (500)
# ---------------------------------------------------------------------------


def test_synthesize_no_generated_audio_returns_500(client, fake_runtime):
    fake_runtime.model = types.SimpleNamespace(generate=lambda **kwargs: iter(()))

    payload = _valid_payload()
    response = _post(client, payload)

    assert response.status_code == 500
    assert "produced no audio" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Regression guard: /synthesize's dispatch to the engine must be untouched
# by the addition of /stream (docs/streaming approved architecture: existing
# /synthesize behavior is protected).
# ---------------------------------------------------------------------------


def test_synthesize_still_calls_generate_with_stream_false(client, fake_runtime):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        yield types.SimpleNamespace(audio=[0.1] * 100, sample_rate=srv.SAMPLE_RATE)

    fake_runtime.model = types.SimpleNamespace(generate=generate)

    payload = _valid_payload()
    response = _post(client, payload)

    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0]["stream"] is False
    assert calls[0]["ref_text"] == payload["reference_text"]
    # /synthesize never sends a streaming_interval -- that field belongs to
    # StreamRequest only.
    assert "streaming_interval" not in calls[0]


# ---------------------------------------------------------------------------
# POST /synthesize — reference-audio pre-flight (400)
# ---------------------------------------------------------------------------


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
# POST /stream — shared fakes and fixtures
#
# This is NATIVE MLX model-level streaming (mlx-audio's own incremental
# decoder), not long-text chunking: no text splitting, no silence joining,
# no crossfade is involved anywhere below.
# ---------------------------------------------------------------------------


def _valid_stream_payload():
    return {
        "text": "Hello there",
        "audio_base64": b64(make_wav_bytes(3.0)),
        # reference_text deliberately omitted: x-vector is the default here
        # so ICL tests opt in explicitly, matching the "presence selects
        # ICL" contract under test.
    }


def _stream_chunk(values, sample_rate=srv.SAMPLE_RATE, is_final=False):
    """A fake GenerationResult-like object; ``audio`` is a plain float list."""
    return types.SimpleNamespace(
        audio=values,
        sample_rate=sample_rate,
        is_streaming_chunk=True,
        is_final_chunk=is_final,
    )


class _FakeStreamModel:
    """Records ``generate()`` kwargs; yields fake incremental chunks.

    ``chunks``: a list of float-lists, one per yielded GenerationResult.
    ``raise_at``: if set, raise ``raise_exc`` instead of yielding
    ``chunks[raise_at]`` (0 = fail before any chunk is produced; >0 = fail
    mid-stream, after ``raise_at`` chunks have already been yielded).
    """

    def __init__(self, chunks=None, raise_at=None, raise_exc=None):
        self.calls: list[dict] = []
        self._chunks = [[0.1, 0.2, 0.3]] if chunks is None else chunks
        self._raise_at = raise_at
        self._raise_exc = raise_exc or RuntimeError("boom")
        self.reset_calls = 0
        self.finalize_calls = 0
        self.speech_tokenizer = types.SimpleNamespace(
            decoder=types.SimpleNamespace(reset_streaming_state=self._reset)
        )

    def _reset(self):
        self.reset_calls += 1

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        try:
            for i, values in enumerate(self._chunks):
                if self._raise_at is not None and i == self._raise_at:
                    raise self._raise_exc
                yield _stream_chunk(values, is_final=(i == len(self._chunks) - 1))
        finally:
            # Fires on ANY termination of this generator -- normal
            # completion, an exception, or an explicit .close() -- so tests
            # can prove the server actually unwound it rather than merely
            # abandoning it.
            self.finalize_calls += 1


@pytest.fixture
def fake_pcm_conversion(monkeypatch):
    """Stdlib-only stand-in for ``_audio_chunk_to_pcm_bytes``.

    Packs a plain list of floats as little-endian float32 -- exactly what
    the real helper is documented to produce -- without depending on a real
    MLX array or (per repo convention, see ``_numpy_to_wav_bytes`` in the
    faster-qwen3-tts suite) a real numpy array in the request path.
    """

    def _fake(audio) -> bytes:
        return struct.pack(f"<{len(audio)}f", *audio)

    monkeypatch.setattr(srv, "_audio_chunk_to_pcm_bytes", _fake)
    return _fake


@pytest.fixture
def fake_streaming_runtime(monkeypatch, fake_pcm_conversion):
    """Like ``fake_runtime``, pre-wired with a working ``_FakeStreamModel``."""
    runtime = types.SimpleNamespace(
        model=_FakeStreamModel(),
        sample_rate=srv.SAMPLE_RATE,
        device=srv.DEVICE,
    )
    monkeypatch.setattr(srv, "_runtime", runtime)
    monkeypatch.setattr(srv, "_decode_wav", lambda raw: (object(), srv.SAMPLE_RATE))
    monkeypatch.setattr(srv, "_prepare_ref_audio", lambda *args, **kwargs: object())
    return runtime


# ---------------------------------------------------------------------------
# POST /stream — request validation (422)
# ---------------------------------------------------------------------------


def test_stream_unknown_field_rejected(client, fake_streaming_runtime):
    payload = _valid_stream_payload()
    payload["bogus_field"] = 1
    assert client.post("/stream", json=payload).status_code == 422


def test_stream_missing_required_fields_rejected(client):
    assert client.post("/stream", json={}).status_code == 422


def test_stream_empty_text_rejected(client):
    payload = _valid_stream_payload()
    payload["text"] = ""
    assert client.post("/stream", json=payload).status_code == 422


def test_stream_unsupported_language_rejected(client):
    payload = _valid_stream_payload()
    payload["language"] = "xx"
    assert client.post("/stream", json=payload).status_code == 422


def test_stream_seed_out_of_range_rejected(client):
    payload = _valid_stream_payload()
    payload["seed"] = 0
    assert client.post("/stream", json=payload).status_code == 422


def test_stream_reference_text_optional_only_for_stream_request(
    client, fake_streaming_runtime
):
    # StreamRequest: reference_text may be omitted entirely (x-vector).
    payload = _valid_stream_payload()
    assert "reference_text" not in payload
    assert client.post("/stream", json=payload).status_code == 200

    # SynthesisRequest is UNCHANGED: reference_text is still required.
    synth_payload = _valid_payload()
    del synth_payload["reference_text"]
    assert client.post("/synthesize", json=synth_payload).status_code == 422


@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "whitespace"])
def test_stream_blank_reference_text_rejected_not_treated_as_absent(
    client, fake_streaming_runtime, blank
):
    # A *supplied* blank reference_text is a malformed ICL request, never a
    # silent fallback to x-vector.
    payload = _valid_stream_payload()
    payload["reference_text"] = blank
    response = client.post("/stream", json=payload)
    assert response.status_code == 422
    assert fake_streaming_runtime.model.calls == []


def test_stream_explicit_null_reference_text_treated_as_absent(
    client, fake_streaming_runtime
):
    payload = _valid_stream_payload()
    payload["reference_text"] = None
    response = client.post("/stream", json=payload)
    assert response.status_code == 200
    assert fake_streaming_runtime.model.calls[-1]["ref_text"] is None


def test_stream_streaming_interval_defaults_to_2_0():
    req = srv.StreamRequest(text="Hello", audio_base64=b64(make_wav_bytes(3.0)))
    assert req.streaming_interval == 2.0


@pytest.mark.parametrize("value", [0.32, 1.0, 2.0])
def test_stream_streaming_interval_accepted(client, fake_streaming_runtime, value):
    payload = _valid_stream_payload()
    payload["streaming_interval"] = value
    response = client.post("/stream", json=payload)
    assert response.status_code == 200


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1.0, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_stream_streaming_interval_rejected(client, fake_streaming_runtime, value):
    payload = _valid_stream_payload()
    payload["streaming_interval"] = value
    # httpx's `json=` encoder rejects NaN/Infinity outright (allow_nan=False)
    # before the request is even sent, so build the body with the stdlib
    # encoder (which -- like most JSON parsers, including the one this
    # server's request goes through -- accepts these as a non-standard but
    # widely-supported extension) and send it as a raw JSON payload instead.
    body = json.dumps(payload).encode("utf-8")

    response = client.post(
        "/stream", content=body, headers={"content-type": "application/json"}
    )

    assert response.status_code == 422
    # Rejected before generation: the fake model must never be invoked.
    assert fake_streaming_runtime.model.calls == []


def test_stream_streaming_interval_not_exposed_streaming_context_size(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"] for p in doc["parameters"]}
    # streaming_context_size is a dead parameter for the single-request MLX
    # generation path this server uses -- it must never be exposed.
    assert "streaming_context_size" not in by_name


# ---------------------------------------------------------------------------
# POST /stream — ICL / x-vector dispatch
# ---------------------------------------------------------------------------


def test_stream_icl_dispatch_forwards_ref_text_and_stream_true(
    client, fake_streaming_runtime
):
    payload = _valid_stream_payload()
    payload["reference_text"] = "A short, exact transcript of the reference clip."
    payload["streaming_interval"] = 0.5

    response = client.post("/stream", json=payload)

    assert response.status_code == 200
    call = fake_streaming_runtime.model.calls[-1]
    assert call["ref_audio"] is not None
    assert call["ref_text"] == payload["reference_text"]
    assert call["stream"] is True
    assert call["streaming_interval"] == 0.5


def test_stream_xvector_dispatch_omits_ref_text_and_stream_true(
    client, fake_streaming_runtime
):
    payload = _valid_stream_payload()
    payload["streaming_interval"] = 0.5

    response = client.post("/stream", json=payload)

    assert response.status_code == 200
    call = fake_streaming_runtime.model.calls[-1]
    assert call["ref_audio"] is not None
    assert call["ref_text"] is None
    assert call["stream"] is True
    assert call["streaming_interval"] == 0.5


# ---------------------------------------------------------------------------
# POST /stream — streaming body (transport)
# ---------------------------------------------------------------------------


def test_stream_body_is_raw_little_endian_float32_pcm_in_order(
    client, fake_streaming_runtime
):
    values_a = [1.0, -1.0, 0.5]
    values_b = [0.25, -0.25]
    fake_streaming_runtime.model = _FakeStreamModel(chunks=[values_a, values_b])

    response = client.post("/stream", json=_valid_stream_payload())

    assert response.status_code == 200
    expected = struct.pack("<3f", *values_a) + struct.pack("<2f", *values_b)
    assert response.content == expected

    # No WAV (or any other) header: exactly 4 bytes per float32 sample.
    assert len(response.content) == 4 * (len(values_a) + len(values_b))
    assert response.content[:4] != b"RIFF"


def test_stream_response_headers_describe_pcm_f32le_mono_24khz(
    client, fake_streaming_runtime
):
    response = client.post("/stream", json=_valid_stream_payload())

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["X-Audio-Format"] == "pcm_f32le"
    assert response.headers["X-Sample-Rate"] == "24000"
    assert response.headers["X-Audio-Channels"] == "1"


def test_stream_final_shorter_chunk_accepted(client, fake_streaming_runtime):
    full = [0.1] * 10
    shorter_final = [0.2] * 3
    fake_streaming_runtime.model = _FakeStreamModel(chunks=[full, shorter_final])

    response = client.post("/stream", json=_valid_stream_payload())

    assert response.status_code == 200
    assert len(response.content) == 4 * (len(full) + len(shorter_final))


def test_stream_pcm_chunks_does_not_buffer_full_utterance(fake_pcm_conversion):
    produced: list[int] = []

    class _TrackingModel(_FakeStreamModel):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            for i, values in enumerate(self._chunks):
                produced.append(i)
                yield _stream_chunk(values, is_final=(i == len(self._chunks) - 1))

    model = _TrackingModel(chunks=[[0.1], [0.2], [0.3]])
    runtime = types.SimpleNamespace(
        model=model, sample_rate=srv.SAMPLE_RATE, device=srv.DEVICE
    )

    gen = srv._stream_pcm_chunks(
        runtime,
        text="hi",
        engine_language="english",
        ref_audio_for_model=object(),
        reference_text=None,
        seed=1,
        streaming_interval=2.0,
    )

    assert produced == []  # nothing generated before the first pull
    next(gen)
    assert produced == [0]  # exactly the first chunk, not all three
    next(gen)
    assert produced == [0, 1]
    next(gen)
    assert produced == [0, 1, 2]
    with pytest.raises(StopIteration):
        next(gen)


# ---------------------------------------------------------------------------
# POST /stream — first-chunk failures (must not commit HTTP 200)
# ---------------------------------------------------------------------------


def test_stream_invalid_setup_before_generation_returns_normal_error(
    client, fake_streaming_runtime
):
    payload = _valid_stream_payload()
    payload["audio_base64"] = b64(b"this is definitely not audio")

    response = client.post("/stream", json=payload)

    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()
    assert fake_streaming_runtime.model.calls == []
    assert not srv._synthesis_lock.locked()


def test_stream_model_exception_before_first_chunk_returns_500(
    client, fake_streaming_runtime
):
    fake_streaming_runtime.model = _FakeStreamModel(chunks=[[0.1]], raise_at=0)

    response = client.post("/stream", json=_valid_stream_payload())

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert fake_streaming_runtime.model.reset_calls == 1
    assert fake_streaming_runtime.model.finalize_calls == 1
    assert not srv._synthesis_lock.locked()


def test_stream_empty_generator_returns_error_not_200(client, fake_streaming_runtime):
    fake_streaming_runtime.model = _FakeStreamModel(chunks=[])

    response = client.post("/stream", json=_valid_stream_payload())

    assert response.status_code == 500
    assert "produced no audio" in response.json()["detail"].lower()
    assert fake_streaming_runtime.model.reset_calls == 1
    assert not srv._synthesis_lock.locked()


# ---------------------------------------------------------------------------
# POST /stream — mid-stream failure (status already committed)
# ---------------------------------------------------------------------------


def test_stream_mid_stream_failure_terminates_cleanly_and_cleans_up(
    client, fake_streaming_runtime
):
    fake_streaming_runtime.model = _FakeStreamModel(
        chunks=[[0.1, 0.2], [0.3, 0.4]], raise_at=1
    )

    # Once the first chunk has already been sent, a later failure cannot
    # change the HTTP status (headers are already committed) -- it can only
    # terminate the stream. TestClient's synchronous, in-process ASGI
    # transport surfaces that as the exception propagating directly to the
    # caller (a real network client would instead just see the connection
    # end early / a truncated body -- there is no way for either transport
    # to inject a JSON/error payload into an already-started raw PCM body).
    with pytest.raises(RuntimeError, match="boom"):
        client.post("/stream", json=_valid_stream_payload())

    # Server-side cleanup must have run regardless, and the lock must be free.
    assert fake_streaming_runtime.model.reset_calls == 1
    assert fake_streaming_runtime.model.finalize_calls == 1
    assert not srv._synthesis_lock.locked()

    # The shared model is usable again immediately afterward.
    fake_streaming_runtime.model = _FakeStreamModel(chunks=[[0.5]])
    followup = client.post("/stream", json=_valid_stream_payload())
    assert followup.status_code == 200


# ---------------------------------------------------------------------------
# POST /stream — client cancellation / disconnect (MANDATORY)
# ---------------------------------------------------------------------------


def test_stream_generator_close_after_first_chunk_cleans_up_and_frees_lock(
    fake_pcm_conversion,
):
    model = _FakeStreamModel(chunks=[[0.1], [0.2], [0.3]])
    runtime = types.SimpleNamespace(
        model=model, sample_rate=srv.SAMPLE_RATE, device=srv.DEVICE
    )

    gen = srv._stream_pcm_chunks(
        runtime,
        text="hi",
        engine_language="english",
        ref_audio_for_model=object(),
        reference_text=None,
        seed=1,
        streaming_interval=2.0,
    )

    first = next(gen)
    assert isinstance(first, (bytes, bytearray))

    # Lower-level guarantee: explicitly closing the inner generator must
    # unwind MLX state and release the lock. Response ownership/disconnect
    # propagation is covered separately below.
    gen.close()

    assert model.reset_calls == 1
    assert model.finalize_calls == 1
    assert not srv._synthesis_lock.locked()


def test_stream_and_synthesize_succeed_after_prior_cancellation(
    client, fake_streaming_runtime
):
    model = _FakeStreamModel(chunks=[[0.1], [0.2], [0.3]])
    fake_streaming_runtime.model = model

    gen = srv._stream_pcm_chunks(
        fake_streaming_runtime,
        text="hi",
        engine_language="english",
        ref_audio_for_model=object(),
        reference_text=None,
        seed=1,
        streaming_interval=2.0,
    )
    next(gen)
    gen.close()
    assert not srv._synthesis_lock.locked()

    # A subsequent /stream request succeeds:
    fake_streaming_runtime.model = _FakeStreamModel(chunks=[[0.4]])
    stream_response = client.post("/stream", json=_valid_stream_payload())
    assert stream_response.status_code == 200

    # A subsequent /synthesize request can acquire/use the SAME lock:
    def generate(**kwargs):
        yield types.SimpleNamespace(audio=[0.1] * 100, sample_rate=srv.SAMPLE_RATE)

    fake_streaming_runtime.model = types.SimpleNamespace(generate=generate)
    synth_response = client.post("/synthesize", json=_valid_payload())
    assert synth_response.status_code == 200



class _StreamExchange:
    """Real app/Request/StreamingResponse ASGI execution, without a socket."""

    def __init__(self, payload=None, spec="2.4"):
        self.scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": spec},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/stream",
            "root_path": "",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
        }
        self.payload = _valid_stream_payload() if payload is None else payload
        self.messages = []
        self.network_threads = []
        self.disconnect_received = False
        self.on_send = None

    async def run(self):
        self.loop_thread = threading.get_ident()
        self.incoming = asyncio.Queue()
        self.disconnect_seen = asyncio.Event()
        self.incoming.put_nowait({
            "type": "http.request",
            "body": json.dumps(self.payload).encode(),
            "more_body": False,
        })

        async def receive():
            self.network_threads.append(threading.get_ident())
            message = await self.incoming.get()
            if message["type"] == "http.disconnect":
                self.disconnect_received = True
                self.disconnect_seen.set()
            return message

        async def send(message):
            self.network_threads.append(threading.get_ident())
            self.messages.append(message)
            if self.on_send:
                await self.on_send(message)

        await srv.app(self.scope, receive, send)

    def disconnect(self):
        self.incoming.put_nowait({"type": "http.disconnect"})

    @property
    def bodies(self):
        return [m["body"] for m in self.messages if m.get("body")]

    @property
    def statuses(self):
        return [m["status"] for m in self.messages if m["type"] == "http.response.start"]


@pytest.fixture
def affine_stream(monkeypatch, fake_streaming_runtime):
    """Every engine operation asserts affinity, including explicit close."""
    state = types.SimpleNamespace(
        events=[], owner=None, calls=[], pulls=[], before_next=None,
        chunks=[[0.1], [0.2], [0.3]], closed=False, cleaned=False, worker_calls=0,
    )

    def record(operation):
        ident = threading.get_ident()
        if state.owner is None:
            state.owner = ident
        assert ident == state.owner, (operation, state.owner, ident)
        state.events.append((operation, ident))

    def prepare(*args):
        record("prepare")
        return object()

    class EngineIterator:
        def __init__(self):
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            record("next")
            state.pulls.append(self.index)
            if state.before_next:
                state.before_next(self.index)
            if self.index == len(state.chunks):
                raise StopIteration
            chunk = state.chunks[self.index]
            self.index += 1
            return _stream_chunk(chunk)

        def close(self):
            record("engine_close")
            state.closed = True

    def generate(**kwargs):
        record("engine_create")
        state.calls.append(kwargs)
        return EngineIterator()

    def reset():
        record("decoder_reset")

    def clear_cache():
        record("cache_clear")
        state.cleaned = True

    original_run_sync = srv.anyio.to_thread.run_sync

    async def count_worker_calls(func, *args, **kwargs):
        if getattr(func, "__name__", "") == "_stream_on_worker":
            state.worker_calls += 1
        return await original_run_sync(func, *args, **kwargs)

    monkeypatch.setattr(srv.anyio.to_thread, "run_sync", count_worker_calls)
    original_chunks = srv._stream_pcm_chunks

    class OwnedIterator:
        def __init__(self, *args, **kwargs):
            record("iterator_create")
            self.inner = original_chunks(*args, **kwargs)

        def __next__(self):
            record("prefetch" if not state.pulls else "advance")
            return next(self.inner)

        def close(self):
            record("iterator_close")
            self.inner.close()

    def convert(audio):
        record("pcm")
        return struct.pack("<f", audio[0])

    monkeypatch.setattr(srv, "_prepare_ref_audio", prepare)
    monkeypatch.setattr(srv, "_stream_pcm_chunks", OwnedIterator)
    monkeypatch.setattr(srv, "_audio_chunk_to_pcm_bytes", convert)
    monkeypatch.setattr(srv.mx, "clear_cache", clear_cache)
    fake_streaming_runtime.model = types.SimpleNamespace(
        generate=generate,
        speech_tokenizer=types.SimpleNamespace(
            decoder=types.SimpleNamespace(reset_streaming_state=reset)
        ),
    )
    return state


def _assert_affine_cleanup(state, exchange):
    assert state.closed and state.cleaned
    assert state.worker_calls == 1
    assert state.owner != exchange.loop_thread
    assert {tid for _, tid in state.events} == {state.owner}
    assert set(exchange.network_threads) == {exchange.loop_thread}
    assert "iterator_close" in [op for op, _ in state.events]
    assert srv._synthesis_lock.acquire(blocking=False)
    srv._synthesis_lock.release()


@pytest.mark.parametrize("icl", [True, False], ids=["icl", "xvector"])
def test_stream_single_worker_multichunk_lifecycle(affine_stream, icl):
    payload = _valid_stream_payload()
    if icl:
        payload["reference_text"] = "Reference speech"
    else:
        payload.pop("reference_text", None)
    exchange = _StreamExchange(payload)

    async def exercise():
        async def inspect_start(message):
            if message["type"] == "http.response.start":
                assert affine_stream.pulls == [0]
                assert [op for op, _ in affine_stream.events].count("pcm") == 1
        exchange.on_send = inspect_start
        await exchange.run()
        _assert_affine_cleanup(affine_stream, exchange)

    asyncio.run(exercise())
    assert exchange.statuses == [200]
    assert exchange.bodies == [struct.pack("<f", x) for x in (0.1, 0.2, 0.3)]
    assert exchange.messages[-1]["more_body"] is False
    assert affine_stream.pulls == [0, 1, 2, 3]
    assert affine_stream.calls[0]["ref_text"] == ("Reference speech" if icl else None)
    assert {"prepare", "iterator_create", "engine_create", "prefetch", "advance",
            "pcm", "iterator_close", "engine_close", "decoder_reset",
            "cache_clear"} <= {op for op, _ in affine_stream.events}


@pytest.mark.parametrize("spec", ["2.4", "2.3"])
def test_stream_asgi_disconnect_cleanup_and_reuse(
    affine_stream, fake_streaming_runtime, spec
):
    exchange = _StreamExchange(spec=spec)

    async def exercise():
        async def disconnect_after_body(message):
            if message.get("body"):
                exchange.disconnect()
        exchange.on_send = disconnect_after_body
        await asyncio.wait_for(exchange.run(), 5)
        assert exchange.disconnect_received
        assert affine_stream.pulls == [0]
        _assert_affine_cleanup(affine_stream, exchange)

        # Affinity is per request, not a requirement to reuse the same
        # worker across different requests.
        affine_stream.owner = None
        # Keep the real lock; subsequent generation uses a fresh fake model.
        fake_streaming_runtime.model = _FakeStreamModel(chunks=[[0.4]])
        followup = _StreamExchange()
        await asyncio.wait_for(followup.run(), 5)
        assert followup.statuses == [200]
        assert followup.bodies == [struct.pack("<f", 0.4)]
        assert not srv._synthesis_lock.locked()

    asyncio.run(exercise())
    assert exchange.statuses == [200]
    assert len(exchange.bodies) == 1


@pytest.mark.parametrize("spec", ["2.4", "2.3"])
def test_stream_disconnect_during_next_waits_for_worker_cleanup(affine_stream, spec):
    exchange = _StreamExchange(spec=spec)
    release = threading.Event()

    async def exercise():
        started = asyncio.Event()
        loop = asyncio.get_running_loop()

        def block_second(index):
            if index == 1:
                loop.call_soon_threadsafe(started.set)
                assert release.wait(5), "test did not release active next()"
        affine_stream.before_next = block_second
        task = asyncio.create_task(exchange.run())
        try:
            await asyncio.wait_for(started.wait(), 5)
            exchange.disconnect()
            # Let the old-ASGI listener consume the message and cancel its
            # task group while next() is still blocked on the worker.
            if spec == "2.3":
                await asyncio.wait_for(exchange.disconnect_seen.wait(), 5)
            else:
                await asyncio.sleep(0)
            assert not task.done()
            assert not affine_stream.closed
            assert srv._synthesis_lock.locked()
        finally:
            release.set()
        await asyncio.wait_for(task, 5)
        assert exchange.disconnect_received
        assert affine_stream.pulls == [0, 1]
        _assert_affine_cleanup(affine_stream, exchange)
        assert len(exchange.bodies) == 1

    asyncio.run(exercise())


def test_stream_backpressure_prevents_next_chunk(affine_stream):
    exchange = _StreamExchange()

    async def exercise():
        sending = asyncio.Event()
        release = asyncio.Event()

        async def blocked_send(message):
            if message.get("body") and not sending.is_set():
                sending.set()
                await release.wait()
        exchange.on_send = blocked_send
        task = asyncio.create_task(exchange.run())
        try:
            await asyncio.wait_for(sending.wait(), 5)
            for _ in range(10):
                await asyncio.sleep(0)
            assert affine_stream.pulls == [0]
            assert not task.done()
        finally:
            release.set()
        await asyncio.wait_for(task, 5)
        _assert_affine_cleanup(affine_stream, exchange)

    asyncio.run(exercise())
    assert len(exchange.bodies) == 3


@pytest.mark.parametrize("fail_on", ["start", "body", "terminal"])
def test_stream_asgi24_send_failure_closes_on_worker(affine_stream, fail_on):
    exchange = _StreamExchange()

    async def exercise():
        async def failing_send(message):
            if ((fail_on == "start" and message["type"] == "http.response.start")
                or (fail_on == "body" and message.get("body"))
                or (fail_on == "terminal" and message.get("more_body") is False)):
                raise OSError("peer closed")
        exchange.on_send = failing_send
        with pytest.raises(ClientDisconnect):
            await exchange.run()
        _assert_affine_cleanup(affine_stream, exchange)

    asyncio.run(exercise())
    if fail_on != "terminal":
        assert affine_stream.pulls == [0]


@pytest.mark.parametrize("spec", ["2.4", "2.3"])
@pytest.mark.parametrize("failure", ["prefetch", "empty", "prepare", "create"])
def test_stream_asgi_startup_failure_never_sends_200(
    monkeypatch, fake_streaming_runtime, spec, failure
):
    if failure == "empty":
        fake_streaming_runtime.model = _FakeStreamModel(chunks=[])
    elif failure == "prefetch":
        fake_streaming_runtime.model = _FakeStreamModel(raise_at=0)
    else:
        def fail(*args, **kwargs):
            raise srv.HTTPException(status_code=500, detail="setup failed")
        monkeypatch.setattr(
            srv, "_prepare_ref_audio" if failure == "prepare" else "_stream_pcm_chunks",
            fail,
        )
    exchange = _StreamExchange(spec=spec)

    async def exercise():
        await asyncio.wait_for(exchange.run(), 5)
        assert exchange.statuses == [500]
        assert not srv._synthesis_lock.locked()

    asyncio.run(exercise())



def test_stream_old_asgi_disconnect_cancels_blocked_send(affine_stream):
    exchange = _StreamExchange(spec="2.3")

    async def exercise():
        sending = asyncio.Event()

        async def blocked_send(message):
            if message.get("body"):
                sending.set()
                await asyncio.Event().wait()  # Only cancellation releases it.

        exchange.on_send = blocked_send
        task = asyncio.create_task(exchange.run())
        await asyncio.wait_for(sending.wait(), 5)
        exchange.disconnect()
        await asyncio.wait_for(task, 5)
        assert exchange.disconnect_received
        assert affine_stream.pulls == [0]
        _assert_affine_cleanup(affine_stream, exchange)

    asyncio.run(exercise())
