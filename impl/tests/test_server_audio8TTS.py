"""Tests for ``server_audio8TTS.py``.

These exercise the HTTP surface that does NOT require a loaded model:
/capabilities (snapshot), /health, the landing page, request-body validation
(422s), and the reference-audio pre-flight checks (400s).  Synthesis itself
needs the real checkpoint + a GPU, so the engine boundary is faked instead:
a fake processor/model pair pins *exactly* what the server forwards (sampling
arguments, the seed handle, the in-memory reference clip, the bad-case retry)
and what it refuses before calling (an over-budget prompt).
"""

import os
import types

import pytest
from fastapi.testclient import TestClient

import server_audio8TTS as srv
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
    assert doc == load_snapshot("audio8_tts_capabilities.json")


def test_capabilities_core_fields_are_required(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    # 'text' is the only field every request must supply.
    assert by_name["text"]["required"] is True
    # Reference audio is optional: the model can speak with its own default voice.
    assert by_name["audio_base64"]["required"] is False
    assert by_name["reference_text"]["required"] is False
    # Optional tuning knobs default to the engine CLI's own values.
    assert by_name["seed"]["required"] is False
    assert by_name["max_new_tokens"]["default"] == 1024
    assert by_name["retry_max_new_tokens"]["default"] == 2000
    assert by_name["temperature"]["default"] == 0.8
    assert by_name["top_p"]["default"] == 0.95
    assert by_name["top_k"]["default"] == 50
    assert by_name["greedy"]["default"] is False


def test_capabilities_sample_rate_and_reference_spec(client):
    doc = client.get("/capabilities").json()
    # ArkttsConfig.codec_sample_rate: 44.1 kHz, confirmed against the
    # checkpoint's own config.json (not the 24 kHz of most engines).
    assert doc["sample_rate"] == 44100
    # The engine has no language parameter (it detects the language from the
    # text), so there is no fixed list to advertise.
    assert doc["languages"] is None
    reference = doc["reference_audio"]
    assert reference["required"] is False
    # The clip budget is arithmetic, not folklore: it must be advertised so a
    # client can see why 20 s is the ceiling.
    assert reference["min_duration_s"] == srv.MIN_PROMPT_DURATION_S
    assert reference["max_duration_s"] == srv.MAX_PROMPT_DURATION_S


def test_capabilities_text_cap_matches_context_budget(client):
    # text + reference clip + generated frames share the model's 2048 packed
    # positions; the advertised caps must be the ones the server enforces.
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    assert by_name["text"]["max_length"] == srv.MAX_TEXT_CHARACTERS
    assert by_name["max_new_tokens"]["max"] == srv.MAX_NEW_TOKENS_LIMIT
    assert srv.MAX_CONTEXT_POSITIONS == 2048


# ---------------------------------------------------------------------------
# GET /health and the landing page
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["serverType"] == "Audio8 TTS"
    assert body["device"] == srv.DEVICE
    assert body["model"] == srv.MODEL_NAME_OR_PATH
    # The tts-serve release version comes from the single shared constant;
    # pinning to srv.RELEASE_VERSION (not a literal) keeps this release-proof
    # while still catching a server that forgot to wire it in.
    assert body["version"] == srv.RELEASE_VERSION


def test_root_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Audio8" in response.text
    assert "/synthesize" in response.text


# ---------------------------------------------------------------------------
# POST /synthesize -- request-body validation (422).  Validation happens before
# the handler runs, so no model is involved.
# ---------------------------------------------------------------------------


def _post(client, payload):
    return client.post("/synthesize", json=payload)


def _stubbed(module) -> bool:
    """True when the server's `module` is one of tests/stubs' import-only fakes.

    A few paths (real encoding, real decoding) cannot be exercised faithfully
    by a stand-in, so their tests say so and stand down instead of pretending.
    """
    parts = (getattr(module, "__file__", "") or "").split(os.sep)
    return "stubs" in parts


def _valid_payload():
    # The engine refuses a reference clip without its transcript, so a valid
    # cloning request always carries both halves of the pair.
    return {
        "text": "Hello there",
        "audio_base64": b64(make_wav_bytes(3.0)),
        "reference_text": "The exact words spoken in the reference clip.",
    }


def test_synthesize_unknown_field_rejected(client):
    payload = _valid_payload()
    payload["bogus_field"] = 1
    assert _post(client, payload).status_code == 422


def test_synthesize_missing_required_fields_rejected(client):
    # 'text' is required; everything else has a default.
    assert _post(client, {}).status_code == 422


def test_synthesize_empty_text_rejected(client):
    payload = _valid_payload()
    payload["text"] = ""
    assert _post(client, payload).status_code == 422


def test_synthesize_whitespace_text_rejected(client):
    payload = _valid_payload()
    payload["text"] = "   "
    assert _post(client, payload).status_code == 422


def test_synthesize_text_overContextCap_rejected(client):
    # The cap keeps prompt + generation inside the model's 2048 positions.
    payload = _valid_payload()
    payload["text"] = "a" * (srv.MAX_TEXT_CHARACTERS + 1)
    assert _post(client, payload).status_code == 422


def test_synthesize_empty_audio_rejected(client):
    payload = _valid_payload()
    # audio_base64 has min_length validation: an empty string is below it.
    payload["audio_base64"] = ""
    assert _post(client, payload).status_code == 422


def test_synthesize_referenceAudioWithoutTranscript_rejected(client):
    # The processor raises "reference_text is required when a reference voice
    # is provided"; catching the pairing at the boundary turns that 500 into a
    # 422 that names the missing field.
    payload = _valid_payload()
    del payload["reference_text"]
    response = _post(client, payload)
    assert response.status_code == 422
    assert "reference_text" in str(response.json()["detail"])


def test_synthesize_referenceTranscriptWithoutAudio_rejected(client):
    # A transcript with no clip is a client bug, not a mode switch: the
    # processor would silently ignore it and hand back default-voice audio.
    payload = {
        "text": "Hello there",
        "reference_text": "The exact words spoken in the reference clip.",
    }
    response = _post(client, payload)
    assert response.status_code == 422
    assert "reference_text" in str(response.json()["detail"])


def test_synthesize_whitespaceTranscript_rejected(client):
    payload = _valid_payload()
    payload["reference_text"] = "   "
    assert _post(client, payload).status_code == 422


def test_synthesize_seed_below_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_seed_above_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 1001
    assert _post(client, payload).status_code == 422


def test_synthesize_maxNewTokens_belowRange_rejected(client):
    payload = _valid_payload()
    payload["max_new_tokens"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_maxNewTokens_aboveContext_rejected(client):
    payload = _valid_payload()
    payload["max_new_tokens"] = srv.MAX_CONTEXT_POSITIONS + 1
    assert _post(client, payload).status_code == 422


def test_synthesize_retryBudget_belowFirstBudget_rejected(client):
    # The CLI's own rule: a smaller retry budget means the retry can only
    # produce shorter audio, which is a typo rather than a choice.
    payload = _valid_payload()
    payload["max_new_tokens"] = 1024
    payload["retry_max_new_tokens"] = 512
    assert _post(client, payload).status_code == 422


def test_synthesize_retryBudget_equalToFirstBudget_accepted(client, stub_runtime):
    # Equal budgets is how a client disables the retry, so it must not 422:
    # with undecodable audio the handler gets as far as its 400 audio check.
    payload = _valid_payload()
    payload["audio_base64"] = b64(b"this is definitely not audio")
    payload["max_new_tokens"] = 1024
    payload["retry_max_new_tokens"] = 1024
    assert _post(client, payload).status_code == 400


def test_synthesize_temperature_zero_rejected(client):
    # The engine clamps temperature to a 1e-5 floor but rejects <= 0 outright.
    payload = _valid_payload()
    payload["temperature"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_topP_aboveOne_rejected(client):
    payload = _valid_payload()
    payload["top_p"] = 1.5
    assert _post(client, payload).status_code == 422


def test_synthesize_topP_zero_rejected(client):
    payload = _valid_payload()
    payload["top_p"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_topK_zero_rejected(client):
    # Not a range nitpick: ArkttsLegacyTopKTopPLogitsProcessor raises
    # "top_k must be positive" even for greedy decoding, so 0 is a 500.
    payload = _valid_payload()
    payload["top_k"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_greedy_badType_rejected(client):
    # Pydantic v2 lax mode coerces "yes"/"true"/"1" to booleans — use a value
    # it cannot coerce.
    payload = _valid_payload()
    payload["greedy"] = "definitely not a bool"
    assert _post(client, payload).status_code == 422


# The shared docs/02 language contract (case, names, garbage, non-strings,
# null/empty -> 'en') is asserted once for every server in
# test_language_contract.py; keep only engine-specific language tests here.


# ---------------------------------------------------------------------------
# POST /synthesize -- reference-audio pre-flight (400)
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_runtime(monkeypatch):
    # Enough of the runtime for the handler to reach its audio pre-flight.
    monkeypatch.setattr(
        srv, "_runtime", types.SimpleNamespace(sample_rate=srv.SAMPLE_RATE, device="cpu")
    )


def test_synthesize_undecodableAudio_rejected(client, stub_runtime):
    payload = _valid_payload()
    payload["audio_base64"] = b64(b"this is definitely not audio")
    response = _post(client, payload)
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_synthesize_tooShortAudio_rejected(client, stub_runtime):
    payload = _valid_payload()
    payload["audio_base64"] = b64(make_wav_bytes(1.0))  # 1.0 s < 2.0 s minimum
    response = _post(client, payload)
    assert response.status_code == 400
    assert "2" in response.json()["detail"]


def test_synthesize_tooLongAudio_rejected(client, stub_runtime):
    # 25 s of clip would eat ~540 of the 2048 context positions the text and
    # the generated speech need -- the engine truncates nothing it can measure,
    # so refuse.
    payload = _valid_payload()
    payload["audio_base64"] = b64(make_wav_bytes(25.0))
    response = _post(client, payload)
    assert response.status_code == 400
    assert "20" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Reference-audio duration: the container header is a claim, the decode is a
# fact, and only the fact is worth enforcing a context budget against
# ---------------------------------------------------------------------------


def test_checkDecodedDuration_headerMatchesDecode_passes():
    # A WAV (or a well-behaved compressed container) decodes to what it claims.
    assert srv._check_decoded_duration(6.43, 6.43) is None
    # Sub-frame padding/priming differences are normal, not corruption.
    assert srv._check_decoded_duration(10.05, 10.14) is None


def test_checkDecodedDuration_headerOverstatesDecode_rejectedNamingBothNumbers():
    # GIVEN a container that advertises more audio than its stream holds...
    # WHEN the gate runs on the samples that actually decoded,
    err = pytest.raises(srv.HTTPException, srv._check_decoded_duration, 5.0, 40.0)
    # THEN the client hears from the container, with both numbers on the table.
    assert err.value.status_code == 400
    assert "decoded to 5.00 s" in err.value.detail
    assert "advertises 40.00 s" in err.value.detail


def test_checkDecodedDuration_decodedLengthOverCap_rejectedEvenIfHeaderAdvertisesLess():
    # The 20 s window is context arithmetic, so it is enforced on the decoded
    # length: a header that lies downward must not smuggle a long clip through.
    err = pytest.raises(srv.HTTPException, srv._check_decoded_duration, 25.0, 6.7)
    assert err.value.status_code == 400
    assert "25.00 s" in err.value.detail
    assert "as decoded" in err.value.detail


def test_checkDecodedDuration_decodedLengthUnderFloor_rejected():
    err = pytest.raises(srv.HTTPException, srv._check_decoded_duration, 1.0, 1.0)
    assert err.value.status_code == 400
    assert "1.00 s" in err.value.detail


# ---------------------------------------------------------------------------
# MP3 references need a length table, or the decoder truncates them silently
# ---------------------------------------------------------------------------


def _mpegBytes(with_length_table: bool) -> bytes:
    """An ID3v2 tag, an MPEG frame sync, and payload -- Xing table or none.

    Built by hand so the check is exercised on a bare dev box: the hazard lives
    in the bytes, not in decoding.  The real thing was verified against a live
    GPU server -- a 40.05 s VBR MP3 written without a length table reads back as
    6.73 s in libsndfile, and the 40 s of audio past that guess never arrives.
    """
    tag = b"ID3\x03\x00\x00" + bytes(6)  # empty v2.3 header, tag size 0
    payload = b"\xff\xfb\x90\x00" + b"\x00" * 64
    if with_length_table:
        payload = payload[:4] + b"Xing" + payload[8:]
    return tag + payload


def test_synthesize_mpegWithoutLengthTable_rejectedBeforeAnyDecoding(client, stub_runtime):
    # GIVEN an MP3 whose duration can only be guessed from one frame's bitrate,
    payload = _valid_payload()
    payload["audio_base64"] = b64(_mpegBytes(with_length_table=False))

    response = _post(client, payload)

    # THEN it is refused outright, naming the real hazard: a clip silently
    # shortened to whatever the guess lands on.
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "length table" in detail and "Xing" in detail


def test_synthesize_mpegWithLengthTable_passesTheLengthCheck(client, stub_runtime):
    # The same bytes with a length table must clear this gate; whatever happens
    # next is the ordinary decode path (on a GPU-free box, the soundfile stub's
    # own "could not decode" 400) -- never the length-table refusal.
    payload = _valid_payload()
    payload["audio_base64"] = b64(_mpegBytes(with_length_table=True))

    response = _post(client, payload)

    assert "length table" not in str(response.json()["detail"])


def test_synthesize_wavReference_notTreatedAsMpeg(client, stub_runtime):
    # Regression guard for the sniff: a WAV starts with 'RIFF', not an MPEG
    # frame sync, so it must fall through to the ordinary duration gate.
    payload = _valid_payload()
    payload["audio_base64"] = b64(make_wav_bytes(1.0))

    response = _post(client, payload)

    assert response.status_code == 400
    assert "length table" not in response.json()["detail"]
    assert "1.00 s" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /synthesize -- the engine boundary, with a fake processor + model
# ---------------------------------------------------------------------------


def _generator_seed(generator):
    """Read back the seed a generator handle was seeded with.

    ``initial_seed()`` is the real torch API; the torch *stub* records the seed
    on the instance instead.  Either way the point of the assertion is that the
    request's seed is what the engine's sampler was handed.
    """
    if hasattr(generator, "initial_seed"):
        return int(generator.initial_seed())
    return generator.seed


class _FakeTensor:
    """Stands in for the small tensors the server reads off processor output."""

    def __init__(self, values):
        self._values = values

    def to(self, _device):
        return self

    def sum(self):
        return sum(self._values)

    def __getitem__(self, index):
        return self._values[index]


class _FakeProcessor:
    """Records the processor call and returns mask/length stand-ins."""

    def __init__(self, prefix_positions=41, suffix_positions=21, reference_samples=198656):
        self.calls = []
        self.prefix_positions = prefix_positions
        self.suffix_positions = suffix_positions
        self.reference_samples = reference_samples

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        inputs = {
            "prefix_input_ids": _FakeTensor(list(range(self.prefix_positions))),
            "prefix_attention_mask": _FakeTensor([1] * self.prefix_positions),
            "suffix_input_ids": _FakeTensor(list(range(self.suffix_positions))),
            "suffix_attention_mask": _FakeTensor([1] * self.suffix_positions),
        }
        if "reference_audio" in kwargs:
            inputs["reference_audio_values"] = _FakeTensor([[0.0] * 8])
            inputs["reference_audio_lengths"] = _FakeTensor([self.reference_samples])
        return inputs


class _FakeWaveform:
    """Chain-compatible stand-in for the padded waveform batch row.

    ``decode_audio`` returns a batch row that runs past the audio it decoded
    (the padding is whatever fills the batch), and reports the real length
    separately.  The fake keeps that padding deliberately visible, so a server
    that forgot to trim would ship it to the client -- and this test suite
    would notice.
    """

    def __init__(self, row):
        self._row = list(row)

    def __getitem__(self, index):
        row, frames = index
        if row != 0:
            raise IndexError("the fake waveform models a batch of one")
        return _FakeWaveform(self._row[frames])

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._row


class _FakeGenerateResult:
    def __init__(self, frames, finished):
        self.codes = ("codes", frames)
        self.code_lengths = [frames]
        self.finished = [finished]


class _FakeEngine:
    """Fake model: records generate()/decode_audio() calls, scripts results."""

    def __init__(self, results=None, decode_error=None):
        # Each queued (frames, finished) drives one generate() pass.  An empty
        # queue means the model stands in for the import-only stub and raises.
        self._results = list(results or [])
        self._decode_error = decode_error
        self.generate_calls = []
        self.decode_calls = []
        # What each generate pass says, and what the encoder is handed.
        self.wav_calls = []
        self.last_frames = 0
        self.waveform_samples = [0.0, 0.25, -0.25]
        self.waveform_padding = 37

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        if not self._results:
            raise NotImplementedError("audio8 stub: generate() is not available in tests")
        frames, finished = self._results.pop(0)
        self.last_frames = frames
        return _FakeGenerateResult(frames=frames, finished=finished)

    def decode_audio(self, codes):
        self.decode_calls.append(codes)
        if self._decode_error is not None:
            raise self._decode_error
        # The row runs ``waveform_padding`` samples past the audio; the server
        # is the only thing standing between that padding and the client.
        row_length = self.last_frames + self.waveform_padding
        row = self.waveform_samples + [0.0] * (row_length - len(self.waveform_samples))
        return _FakeWaveform(row), [self.last_frames]


@pytest.fixture
def fake_engine(monkeypatch):
    """Return a builder that installs a fake runtime and reports its fakes.

    Built as a factory (rather than a bare fixture) because the scenarios
    differ in what the fake processor reports and what the fake model returns.

    ``_decode_reference_audio`` is replaced too: decoding needs real
    soundfile/NumPy, which a dev box does not have.  The stand-in proves the
    server hands the engine an in-memory (samples, rate) pair, which is what
    the processor wants (no temp file, nothing to clean up).
    """

    def build(results=None, decode_error=None, prefix_positions=41,
              suffix_positions=21, reference_samples=198656):
        processor = _FakeProcessor(
            prefix_positions=prefix_positions,
            suffix_positions=suffix_positions,
            reference_samples=reference_samples,
        )
        engine = _FakeEngine(results=results, decode_error=decode_error)
        monkeypatch.setattr(
            srv,
            "_runtime",
            types.SimpleNamespace(
                model=engine,
                processor=processor,
                sample_rate=srv.SAMPLE_RATE,
                max_context_positions=srv.MAX_CONTEXT_POSITIONS,
                device="cpu",
            ),
        )
        monkeypatch.setattr(
            srv, "_decode_reference_audio", lambda raw, advertised: ("mono-samples", 24000)
        )

        # Record what the WAV encoder is handed instead of running it: the
        # point is the array's length (the padding must be gone) and the rate.
        def recordWave(array, rate):
            engine.wav_calls.append((array, rate))
            return b"wav-bytes"

        monkeypatch.setattr(srv, "_numpy_to_wav_bytes", recordWave)
        return processor, engine

    return build


def test_synthesize_cloning_forwardsEngineArguments(client, fake_engine):
    processor, engine = fake_engine(results=[(128, True)])

    # GIVEN a cloning request with non-default sampling knobs,
    payload = _valid_payload()
    payload.update({"seed": 7, "temperature": 0.5, "top_p": 0.8, "top_k": 20})

    # WHEN it is synthesized,
    response = _post(client, payload)

    # THEN the processor saw the text plus the in-memory clip and transcript,
    assert processor.calls == [
        {
            "text": "Hello there",
            "return_tensors": "pt",
            "reference_audio": [("mono-samples", 24000)],
            "reference_text": [_valid_payload()["reference_text"]],
        }
    ], "the reference clip must reach the processor as an in-memory (samples, rate) pair"
    # ...and generate() saw the engine CLI's argument shape, sampled decoding,
    # and the requested budget.
    (generate_kwargs,) = engine.generate_calls
    assert generate_kwargs["max_new_tokens"] == 1024
    assert generate_kwargs["temperature"] == 0.5
    assert generate_kwargs["top_p"] == 0.8
    assert generate_kwargs["top_k"] == 20
    assert generate_kwargs["do_sample"] is True
    assert generate_kwargs["return_dict_in_generate"] is True
    assert _generator_seed(generate_kwargs["generator"]) == 7
    assert "prefix_input_ids" in generate_kwargs
    assert "reference_audio_lengths" in generate_kwargs


def test_synthesize_greedy_forwardsDoSampleFalse(client, fake_engine):
    _, engine = fake_engine(results=[(128, True)])
    payload = _valid_payload()
    payload["greedy"] = True

    assert _post(client, payload).status_code == 200

    assert engine.generate_calls[0]["do_sample"] is False


def test_synthesize_withoutReference_omitsProcessorReferenceArguments(client, fake_engine):
    processor, engine = fake_engine(results=[(64, True)])

    # GIVEN a request with neither clip nor transcript (the model's default voice)...
    response = _post(client, {"text": "No reference voice here."})

    # THEN the processor is called without the reference fields at all, so the
    # engine picks its transcript-free prompt template.
    assert response.status_code == 200
    assert "reference_audio" not in processor.calls[0]
    assert "reference_text" not in processor.calls[0]
    assert "reference_audio_lengths" not in engine.generate_calls[0]


def test_synthesize_truncatedGeneration_retriesWithClampedLargerBudget(client, fake_engine):
    # GIVEN a model that fills its first frame budget without ever emitting
    # end-of-speech, then finishes on the retry...
    _, engine = fake_engine(results=[(1024, False), (1400, True)])

    response = _post(client, _valid_payload())

    # THEN exactly one retry ran, with the configured larger budget, and the
    # retry's (complete) audio is what the client gets.  Both budgets arrive
    # clamped to what the prompt leaves over: the fake prompt costs 41 + 21
    # tokens + ceil(198656 / 2048) = 97 clip frames = 159 positions, so the
    # 2000-frame retry request can never be more than 2048 - 159 = 1889.
    assert response.status_code == 200
    budgets = [call["max_new_tokens"] for call in engine.generate_calls]
    assert budgets == [1024, 1889]
    assert response.json()["finished"] is True
    assert len(engine.decode_calls) == 1, "only the retry may be decoded"
    # The retry's frames, with the batch padding trimmed off -- the padding is
    # silence, and the client has no way to strip it by hand.
    (wav, wav_rate), = engine.wav_calls
    assert len(wav) == 1400
    assert wav_rate == srv.SAMPLE_RATE

def test_synthesize_retrySkipped_whenEngineWouldClampBothBudgetsToSameCeiling(
    client, fake_engine
):
    # GIVEN a prompt so wide that the context, not the request, decides the
    # ceiling: 41 + 21 tokens + ceil(2224128 / 2048) = 1086 clip frames = 1148
    # positions, leaving 900 -- below the requested 1024.  The engine clamps
    # max_new_tokens to max_seq_len - prompt_width, so a "larger" retry budget
    # of 2000 lands on the very same 900.
    _, engine = fake_engine(results=[(900, False)], reference_samples=2_224_128)

    response = _post(client, _valid_payload())

    # THEN the request never asks the engine for a number it is going to clamp
    # away, and the pointless second pass is skipped instead of doubling the
    # client's wait for an identical-length re-roll.
    assert response.status_code == 200
    assert [call["max_new_tokens"] for call in engine.generate_calls] == [900]
    assert response.json()["finished"] is False


def test_synthesize_tooWidePromptLeavesNoRoomForSpeech_rejectedWithBudget(client, fake_engine):
    # GIVEN a request that passes every schema cap but whose prompt costs
    # 2030 of the 2048 positions (no reference clip: 1200 + 830 tokens)...
    processor, engine = fake_engine(prefix_positions=1200, suffix_positions=830)

    response = _post(client, {"text": "x" * 1200, "max_new_tokens": 1024})

    # THEN it is a 400 that names the arithmetic the client cannot see, rather
    # than a 200 whose audio is a fraction of a second (the engine happily
    # generates 18 frames and calls finished=True; measured at 0.05 s of audio
    # for the equivalent real request).
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "2030" in detail and "18 frames" in detail and "0.8 s" in detail
    assert engine.generate_calls == []
    assert processor.calls


def test_synthesize_promptLeavesLessRoomThanTheModelNeeds_rejected(client, fake_engine):
    # GIVEN the request shape that measured worst on a real GPU: a script that
    # tokenizes near one token per character (Devanagari) plus a clip at the
    # 20 s ceiling -- 1100 + 367 tokens and ceil(880640 / 2048) = 430 clip
    # frames = 1897 of 2048 positions, leaving 151 frames.  The real server
    # answered that with exactly one frame of audio (0.0 s) and finished=true.
    _, engine = fake_engine(
        prefix_positions=1100, suffix_positions=367, reference_samples=880_640
    )

    response = _post(client, _valid_payload())

    # THEN it is a 400 rather than an unlistenable 200, and the numbers name
    # the half the client can still shorten.
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "1897" in detail and "151 frames" in detail and "7.0 s" in detail
    assert "1467 text tokens" in detail and "430 reference-clip frames" in detail
    assert engine.generate_calls == []


def test_synthesize_promptLeavingExactlyTheFloor_is_answered(client, fake_engine):
    # GIVEN a prompt that leaves exactly MIN_GENERATION_FRAMES over: the floor
    # is a floor, not a threshold to overshoot (1490 - 430 clip frames leaves
    # 1792 - 1100 = 692 suffix tokens worth of the budget unspent).
    clip_frames = 430
    tokens = (
        srv.MAX_CONTEXT_POSITIONS - srv.MIN_GENERATION_FRAMES - clip_frames
    )
    _, engine = fake_engine(
        prefix_positions=1100,
        suffix_positions=tokens - 1100,
        reference_samples=clip_frames * srv.CODEC_FRAME_SIZE,
        results=[(srv.MIN_GENERATION_FRAMES, False)],
    )

    response = _post(client, _valid_payload())

    # THEN the request is generated (budget clamped to the frames on hand, and
    # the retry skipped because it would clamp to that same ceiling).
    assert response.status_code == 200
    assert [c["max_new_tokens"] for c in engine.generate_calls] == [
        srv.MIN_GENERATION_FRAMES
    ]
    assert response.json()["finished"] is False


def test_synthesize_equalBudgets_skipRetry(client, fake_engine):
    _, engine = fake_engine(results=[(1024, False)])
    payload = _valid_payload()
    payload["retry_max_new_tokens"] = payload["max_new_tokens"] = 1024

    response = _post(client, payload)

    assert response.status_code == 200
    assert len(engine.generate_calls) == 1
    assert response.json()["finished"] is False


def test_synthesize_responseCarriesSeedAndSampleRate(client, fake_engine):
    _, engine = fake_engine(results=[(128, True)])
    payload = _valid_payload()
    payload["seed"] = 42

    body = _post(client, payload).json()

    assert body["sample_rate"] == srv.SAMPLE_RATE
    # The seed the client asked for is the seed that was applied (no engine
    # side seed increments here).
    assert body["seed"] == 42
    assert body["audio_base64"] == b64(b"wav-bytes")
    assert body["finished"] is True
    # Exactly the 128 generated frames reach the WAV encoder: the batch padding
    # is silence the client has no way to strip by hand.
    (wav, wav_rate), = engine.wav_calls
    assert len(wav) == 128, "exactly the generated frames: the padding never ships"
    assert wav_rate == srv.SAMPLE_RATE


def test_encodeWaveform_outOfRangeSamples_neverShipAsFullScalePcm():
    # GIVEN a decode that left the sample range: NaN and inf sail straight
    # through np.clip, and libsndfile writes them as digital full scale
    # (measured on a GPU box: nan -> -32768, +inf -> +32767).
    if _stubbed(srv.np) or _stubbed(srv.sf):
        pytest.skip("the WAV encoder needs real NumPy + libsndfile, not the import-only stubs")

    real_numpy = srv.np
    samples = real_numpy.array([float("nan"), float("inf"), float("-inf"), 2.0, -3.0, 0.5])

    # WHEN the waveform is encoded,
    encoded = srv._numpy_to_wav_bytes(samples, srv.SAMPLE_RATE)

    # THEN it reads back finite and in range, with the NaN replaced by silence
    # rather than the loudest sample the format can hold.
    import io

    reloaded, rate = srv.sf.read(io.BytesIO(encoded))
    assert rate == srv.SAMPLE_RATE
    assert reloaded[0] == pytest.approx(0.0), "NaN must become silence, not full scale"
    assert all(abs(sample) <= 1.0 for sample in reloaded)


# ---------------------------------------------------------------------------
# Engine-specific language tests
# ---------------------------------------------------------------------------


def test_synthesize_promptOverContextBudget_rejectedBeforeGeneration(client, fake_engine):
    # GIVEN a prompt whose token count alone outgrows the model's context...
    processor, engine = fake_engine(prefix_positions=1200, suffix_positions=1000)

    response = _post(client, {"text": "x" * 1200})

    # THEN the request is a 400 that names the budget, and the model is never
    # asked (the engine's own "Prompt length ... must be smaller than 2048"
    # would have arrived as a 500).
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "2048" in detail and "positions" in detail
    assert engine.generate_calls == []
    assert processor.calls  # the check runs on the processor's own output


def test_synthesize_promptBudgetCountsReferenceFrames(client, fake_engine):
    # GIVEN a prompt that only overflows once the clip's frames are counted:
    # the token rows alone (41 + 1100) are comfortably inside the budget, but
    # 1,900,000 clip samples add ceil(1900000 / 2048) = 928 frames -> 2069.
    processor, engine = fake_engine(reference_samples=1_900_000, suffix_positions=1100)
    payload = _valid_payload()

    response = _post(client, payload)

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "2069" in detail
    assert "2048" in detail and "positions" in detail
    # The 400 must say which half to shorten, in units the client owns.
    assert "1141 text tokens" in detail and "928 reference-clip frames" in detail
    assert engine.generate_calls == []
    assert "reference_audio" in processor.calls[0]


def test_synthesize_engineFailure_returnsFiftyWithDetail(client, fake_engine):
    # GIVEN a fake runtime whose model stands in for the import-only stub...
    _, engine = fake_engine()

    # WHEN the request clears validation and the pre-flights, the engine call
    # raises, and the failure is reported rather than hidden.
    response = _post(client, _valid_payload())
    assert response.status_code == 500
    assert "not available in tests" in response.json()["detail"]
    assert len(engine.generate_calls) == 1


# ---------------------------------------------------------------------------
# Engine-specific language tests
# ---------------------------------------------------------------------------


def test_synthesize_languageWellFormedNonEnglish_passesValidation(client, stub_runtime):
    # languages: null -- the engine takes no language parameter, so any
    # well-formed two-letter code clears validation (the issue behind this
    # server asks for Thai/Swedish: the model detects those from the *text*).
    payload = _valid_payload()
    payload["audio_base64"] = b64(b"this is definitely not audio")
    payload["language"] = "th"
    assert _post(client, payload).status_code == 400
