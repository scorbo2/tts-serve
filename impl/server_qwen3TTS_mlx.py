"""
FastAPI REST server for Qwen3-TTS voice cloning via MLX (Apple Silicon).

This is the MLX-native counterpart to ``server_qwen3TTS.py`` (PyTorch/CUDA/
MPS), built for a controlled A/B comparison between the two backends on the
same Mac.  It wraps the same Qwen3-TTS Base checkpoint family through
``mlx-audio`` instead of ``qwen_tts``/PyTorch.

Loads the model once on startup, then exposes two POST endpoints:

POST /synthesize
    Complete-file synthesis (unchanged, protected behavior): the full
    request/response cycle, base64-encoded 24 kHz WAV in one JSON response.
    ICL (in-context learning) voice cloning only: ``reference_text`` is
    required, there is no speaker-embedding-only fallback, no long-text
    chunking, no streaming, and no voice-library / preset-voice modes.  (See
    ``server_qwen3TTS.py`` for those.)

POST /stream
    Native MLX model-level streaming (mlx-audio's own incremental decoder,
    ``Model.generate(..., stream=True)``) -- NOT long-text chunking; there is
    no text splitting, silence joining, or crossfade here.  ``reference_text``
    is optional: supplying it selects ICL voice cloning (matching
    /synthesize's conditioning); omitting it selects x-vector
    (speaker-embedding) voice cloning instead -- mlx-audio's `Model.generate()`
    routes on that same presence/absence signal internally.  The response
    body is raw headerless little-endian float32 PCM (``pcm_f32le``), mono,
    24 kHz, emitted incrementally as mlx-audio yields each chunk -- no WAV
    container, no buffering of the complete utterance.  See ``StreamRequest``
    and the ``streaming`` block of GET /capabilities for the full contract.

Capabilities: GET /capabilities returns a machine-readable description of
every request parameter, derived from the Pydantic request model so it can
never drift from what the server actually validates (see the
tts-engine-common README).  Its ``streaming`` block (present because this
engine supports it) documents the /stream endpoint's fixed wire format.

Model weights are downloaded from HuggingFace on first start unless a local
path is given.  The language list below is the Base checkpoint's (10
languages + auto) -- identical to server_qwen3TTS.py's table, since both
wrap the same Qwen3-TTS Base checkpoint family; the engine itself validates
against its own config.

Device: mlx-audio has no device-selection knob (MLX picks its own compute
backend, Metal on Apple Silicon) so there is no ``*_DEVICE`` environment
variable here.  This server reports device as the literal string ``"mlx"``
everywhere (health check, capabilities, logging) -- chosen over "metal"
because "mlx" names the framework actually in use, and mlx-audio itself has
no notion of "metal" as a selectable device value.

Configuration (environment variables):
    QWEN3TTS_MLX_MODEL   HuggingFace id or local path.
                        Default: mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit
    QWEN3TTS_MLX_HOST    Bind host for `python server_qwen3TTS_mlx.py`.
                        Default: 0.0.0.0
    QWEN3TTS_MLX_PORT    Bind port for `python server_qwen3TTS_mlx.py`.
                        Default: 7500

Extra dependencies beyond the mlx-audio package:
    pip install fastapi uvicorn loguru soundfile
    pip install ../tts-engine-common # in-repo copy; or: pip install -e ../tts-engine-common

Usage:
    python server_qwen3TTS_mlx.py
    # or: uvicorn server_qwen3TTS_mlx:app --host 0.0.0.0 --port 7500
"""

from __future__ import annotations

import base64
import io
import math
import os
import random
import threading
import time
import uuid
from concurrent.futures import CancelledError as FutureCancelledError
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import anyio
import mlx.core as mx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from loguru import logger
from mlx_audio.tts.utils import load_model
from mlx_audio.utils import resample_audio
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tts_engine_common import (
    DEFAULT_LANGUAGE,
    CoreSynthesisResponse,
    build_capabilities,
    capabilities_endpoint,
    compute_rtf,
    decode_base64,
    normalize_language,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME_OR_PATH = os.getenv(
    "QWEN3TTS_MLX_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
)

# No device env var: mlx-audio has no device-selection argument, and MLX
# itself auto-selects its compute backend (Metal on Apple Silicon).  "mlx" is
# the string reported everywhere a device value is expected.
DEVICE = "mlx"

# Verified against the installed mlx-audio source: Qwen3TTSModel.ModelConfig
# defaults sample_rate to 24000, and the target checkpoint's config.json does
# not override it (no "sample_rate" key present) -- same rate as
# server_qwen3TTS.py's PyTorch path.  The actual value used for each response
# still comes from the loaded model's own `.sample_rate` property at
# synthesis time (see synthesize()), never assumed.
SAMPLE_RATE = 24000

SEED_MIN = 1
SEED_MAX = 1000

# Heuristic lower bound: below this the reference codes degrade to
# near-garbage. Mirrors server_qwen3TTS.py's MIN_REF_DURATION_S.
MIN_REF_DURATION_S = 2.0

# Sanity valve for the request payload (~5 min of 24 kHz audio).
MAX_AUDIO_B64_LEN = 10_000_000

# mlx-audio's own Model.generate() default for `streaming_interval` (seconds
# of GENERATED AUDIO accumulated per streamed chunk -- not a wall-clock
# delay). Official mlx-audio examples use values as low as 0.32 for lower
# latency at the cost of more per-chunk overhead; no upper bound is
# documented, so none is enforced here beyond "finite and > 0".
DEFAULT_STREAMING_INTERVAL_S = 2.0

# /stream's fixed wire format (see StreamingCapability in tts-engine-common
# and the module docstring): raw headerless little-endian float32 PCM, mono,
# at the model's own sample rate. Not configurable in this first version --
# no output-format negotiation.
STREAM_AUDIO_FORMAT = "pcm_f32le"
STREAM_CHANNELS = 1

# Voice-conditioning modes /stream supports, selected implicitly by
# reference_text's presence/absence (no explicit mode parameter -- see
# StreamRequest).
STREAM_VOICE_CONDITIONING = ("icl", "x_vector")

# Identical to server_qwen3TTS.py's table: both servers wrap the same
# Qwen3-TTS Base checkpoint family, and mlx-audio's Qwen3TTS Model.generate()
# takes the same lowercase language *names* via its `lang_code` argument
# (verified in mlx_audio/tts/models/qwen3_tts/qwen3_tts.py -- `lang_code` is
# threaded straight through as the `language=` argument of
# `_generate_icl()`/`_prepare_generation_inputs()`, i.e. it is the target
# synthesis language, not a source-language hint).
#
# The API contract is two-letter codes (docs/02-language-handling.md); this
# table is this server's internal code -> engine-name mapping.
LANGUAGE_CODE_TO_NAME = {
    "zh": "chinese",
    "en": "english",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "pt": "portuguese",
    "ru": "russian",
    "es": "spanish",
}
LANGUAGE_CODES = tuple(LANGUAGE_CODE_TO_NAME)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

# Dynamic Literal over the Base checkpoint's language codes (+ the engine's
# 'auto' auto-detection sentinel, mlx-audio's own default for `lang_code`) so
# the request schema and the /capabilities enum share one source.
Language = Literal[("auto", *LANGUAGE_CODES)]


class SynthesisRequest(BaseModel):
    """A single synthesis request. Unknown fields are rejected (422)."""

    model_config = ConfigDict(extra="forbid")

    # --- core vocabulary (tts_engine_common.CORE_FIELDS) -------------------
    text: str = Field(
        ...,
        min_length=1,
        description="Text to synthesize, e.g. 'Hello there'.",
    )
    audio_base64: str = Field(
        ...,
        min_length=1,
        max_length=MAX_AUDIO_B64_LEN,
        description=(
            "Reference voice sample as a base64 string.  Any container "
            "soundfile can decode (WAV, MP3, OGG, FLAC, ...).  ~3 s is enough "
            "for high-quality cloning."
        ),
    )
    reference_text: str = Field(
        ...,
        min_length=1,
        description=(
            "Exact transcript of the reference audio.  Required: this server "
            "runs ICL (in-context learning) voice cloning only, which "
            "conditions on the reference audio and its transcript together."
        ),
    )
    language: Language | None = Field(
        DEFAULT_LANGUAGE,
        description=(
            "Two-letter language code, e.g. 'en', 'zh', or 'auto' for "
            f"auto-detection (supported: {', '.join(sorted(LANGUAGE_CODES))}).  "
            "Omitted or empty defaults to 'en'."
        ),
    )
    seed: int | None = Field(
        None,
        ge=SEED_MIN,
        le=SEED_MAX,
        description=(
            "Random seed for reproducibility (seeds MLX's global RNG via "
            "mx.random.seed(), which drives the talker's token sampling). "
            f"If omitted, a random seed in [{SEED_MIN}, {SEED_MAX}] is chosen "
            "and echoed in the response."
        ),
    )

    @field_validator("text")
    @classmethod
    def _validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain non-whitespace characters")
        return v

    @field_validator("reference_text")
    @classmethod
    def _validate_reference_text(cls, v: str) -> str:
        # ICL mode feeds the transcript straight into the model's context; a
        # blank one would mis-condition the clone, so reject it loudly.
        if not v.strip():
            raise ValueError("reference_text must contain non-whitespace characters")
        return v

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, v: object) -> str:
        # docs/02: null/empty means English.  'mode=before' means ``v`` is
        # the raw JSON value (pre-coercion): non-strings are rejected here
        # as ValueError (422), never downstream as AttributeError (500).
        # Runs before the Literal check so the normalized default ('en') is
        # always a valid member.
        return normalize_language(v)


class SynthesisResponse(CoreSynthesisResponse):
    """The synthesis result (core fields from tts_engine_common, plus fid)."""

    fid: str = Field(..., description="Request ID (internal).")


class StreamRequest(BaseModel):
    """A single POST /stream request. Unknown fields are rejected (422).

    Deliberately a SEPARATE model from ``SynthesisRequest`` (never weakened
    to support streaming, per the approved architecture): ``reference_text``
    is optional here because its presence/absence IS the conditioning-mode
    switch --

        reference_text supplied  -> ICL (in-context learning)
        reference_text omitted   -> x-vector (speaker-embedding)

    -- mirroring how mlx-audio's own ``Model.generate()`` decides internally.
    There is no explicit ``mode`` field. A *supplied but blank* value (``""``
    or whitespace) is a malformed ICL request and is rejected (422), never
    silently reinterpreted as "absent" / x-vector.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        description="Text to synthesize, e.g. 'Hello there'.",
    )
    audio_base64: str = Field(
        ...,
        min_length=1,
        max_length=MAX_AUDIO_B64_LEN,
        description=(
            "Reference voice sample as a base64 string.  Any container "
            "soundfile can decode (WAV, MP3, OGG, FLAC, ...).  ~3 s is enough "
            "for high-quality cloning."
        ),
    )
    reference_text: str | None = Field(
        None,
        min_length=1,
        description=(
            "Exact transcript of the reference audio.  If supplied, "
            "streaming uses ICL (in-context learning) voice cloning "
            "(matching /synthesize's conditioning); if omitted, streaming "
            "uses x-vector (speaker-embedding) voice cloning instead.  A "
            "present-but-blank value is rejected (422), never treated as "
            "absent."
        ),
    )
    language: Language | None = Field(
        DEFAULT_LANGUAGE,
        description=(
            "Two-letter language code, e.g. 'en', 'zh', or 'auto' for "
            f"auto-detection (supported: {', '.join(sorted(LANGUAGE_CODES))}).  "
            "Omitted or empty defaults to 'en'.  Same semantics as "
            "/synthesize."
        ),
    )
    seed: int | None = Field(
        None,
        ge=SEED_MIN,
        le=SEED_MAX,
        description=(
            "Random seed for reproducibility (seeds MLX's global RNG via "
            "mx.random.seed(), which drives the talker's token sampling). "
            f"If omitted, a random seed in [{SEED_MIN}, {SEED_MAX}] is chosen. "
            "Same semantics as /synthesize."
        ),
    )
    streaming_interval: float = Field(
        DEFAULT_STREAMING_INTERVAL_S,
        description=(
            "Seconds of GENERATED AUDIO accumulated per streamed chunk -- "
            "not a wall-clock delay.  For example, the default 2.0 means "
            "mlx-audio accumulates approximately two seconds of generated "
            "audio before yielding a normal full-sized chunk; the actual "
            "wall-clock time to produce that audio is typically much "
            "shorter.  Smaller values (mlx-audio's own examples use values "
            "as low as 0.32) trade lower latency for more per-chunk "
            "overhead; there is no documented upper bound.  Must be finite "
            "and greater than 0.  The final chunk of a stream may contain "
            "less audio than this."
        ),
    )

    @field_validator("text")
    @classmethod
    def _validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain non-whitespace characters")
        return v

    @field_validator("reference_text")
    @classmethod
    def _validate_reference_text(cls, v: str | None) -> str | None:
        # None means "not supplied" (x-vector) -- Pydantic does not invoke a
        # field_validator against an omitted field's default, so this only
        # ever runs with v=None when a client sends an explicit JSON `null`,
        # which is treated the same as omission. A *supplied* blank/
        # whitespace-only string is a malformed ICL request, not a mode
        # switch, so it is rejected loudly rather than silently falling back
        # to x-vector.
        if v is None:
            return None
        if not v.strip():
            raise ValueError("reference_text must contain non-whitespace characters")
        return v

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, v: object) -> str:
        return normalize_language(v)

    @field_validator("streaming_interval")
    @classmethod
    def _validate_streaming_interval(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            raise ValueError(
                "streaming_interval must be a finite number greater than 0"
            )
        return v


class HealthResponse(BaseModel):
    """Health / readiness check."""

    status: Literal["ok"] = "ok"
    serverType: Literal["Qwen3-TTS-MLX"] = "Qwen3-TTS-MLX"
    model: str = MODEL_NAME_OR_PATH
    device: str = DEVICE


# ---------------------------------------------------------------------------
# Capabilities (derived from SynthesisRequest — single source of truth)
# ---------------------------------------------------------------------------

CAPABILITIES = build_capabilities(
    SynthesisRequest,
    engine="qwen3-tts-mlx",
    model=MODEL_NAME_OR_PATH,
    device=DEVICE,
    sample_rate=SAMPLE_RATE,
    watermarked=False,
    endpoint="/synthesize",
    reference_audio={
        "required": True,
        "formats": ["wav", "mp3", "ogg", "flac"],
        "min_duration_s": MIN_REF_DURATION_S,
        "note": (
            "reference_text (the exact transcript) is required: this server "
            "runs ICL voice cloning only, conditioning on the reference "
            "audio and transcript together. ~3 s is enough for "
            "high-quality cloning."
        ),
    },
    languages=sorted(LANGUAGE_CODES),
    streaming={
        "endpoint": "/stream",
        "format": STREAM_AUDIO_FORMAT,
        "sample_rate": SAMPLE_RATE,
        "channels": STREAM_CHANNELS,
        "voice_conditioning": list(STREAM_VOICE_CONDITIONING),
    },
)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Pre-load the model on startup and free it on shutdown."""
    _get_runtime()
    yield
    global _runtime
    if _runtime is not None:
        del _runtime.model
        _runtime = None
    mx.clear_cache()
    logger.info("Model unloaded and MLX cache cleared.")


app = FastAPI(
    title="Qwen3-TTS MLX Voice Cloning API",
    description=(
        "REST API around Qwen3-TTS via mlx-audio (Apple Silicon native). "
        "Send text + a reference audio sample + its transcript and get back "
        "cloned speech.  Machine-readable parameter metadata at "
        "GET /capabilities."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_api_route(
    "/capabilities",
    capabilities_endpoint(CAPABILITIES),
    methods=["GET"],
    tags=["System"],
    summary="Machine-readable description of the request parameters",
)

# ---------------------------------------------------------------------------
# Runtime — thin wrapper around the mlx-audio Qwen3-TTS model
# ---------------------------------------------------------------------------


@dataclass
class Qwen3TTSMLXRuntime:
    """Holds the loaded model and its metadata for the lifetime of the server."""

    model: Any
    device: str
    sample_rate: int


_runtime: Qwen3TTSMLXRuntime | None = None

# mlx-audio's Model keeps an internal ICL cache (self._icl_cache) that is
# mutated during generation; serialize synthesis so concurrent requests don't
# race on shared model state (mirrors the *_synthesis_lock pattern used by
# the other engine servers with shared mutable model state).
_synthesis_lock = threading.Lock()


def _get_runtime() -> Qwen3TTSMLXRuntime:
    """Return the global runtime, loading the model once on first call."""
    global _runtime
    if _runtime is None:
        logger.info("Loading Qwen3-TTS MLX model '{}' ...", MODEL_NAME_OR_PATH)
        model = load_model(MODEL_NAME_OR_PATH)
        _runtime = Qwen3TTSMLXRuntime(
            model=model, device=DEVICE, sample_rate=model.sample_rate
        )
        logger.info(
            "Model loaded successfully (sample_rate={} Hz).", model.sample_rate
        )
    return _runtime


# ---------------------------------------------------------------------------
# Global exception handler — catches anything that slips past endpoint handlers
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def _unhandled_exception(_request, exc: Exception) -> JSONResponse:
    """Return a meaningful 500 instead of FastAPI's blank ``detail: ''``."""
    logger.error("Unhandled exception: {}", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )


def _sanitize_nonfinite(value: Any) -> Any:
    """Replace non-finite floats (NaN/±inf) with ``None``, recursively.

    Pydantic echoes the raw rejected value in each validation error's
    ``input`` field (e.g. ``-inf`` for a rejected ``streaming_interval``).
    Starlette's ``JSONResponse`` renders with ``allow_nan=False`` (strict
    JSON), so a 422 whose detail contains that raw value would otherwise
    itself fail to serialize and surface as an unrelated 500 -- exactly the
    class of bug ``CoreSynthesisResponse``'s existing ``rtf``/``time_used``
    sanitizers guard against for response bodies; this is the same fix for
    validation-error bodies.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _sanitize_nonfinite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_nonfinite(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(
    _request, exc: RequestValidationError
) -> JSONResponse:
    """The normal 422, even when the rejected input is non-finite (see
    ``_sanitize_nonfinite``)."""
    return JSONResponse(
        status_code=422,
        content={"detail": _sanitize_nonfinite(jsonable_encoder(exc.errors()))},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, tags=["System"])
def root() -> str:
    """A friendly landing page so browser visitors don't get the auto-generated docs."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head><title>Qwen3-TTS MLX REST API</title></head>
    <body>
        <h1>Qwen3-TTS MLX Voice Cloning REST API</h1>
        <p>Model: <code>{MODEL_NAME_OR_PATH}</code> on <code>{DEVICE}</code>.
        This server is a REST API, not a web server.</p>
        <p>Use a REST client like <strong>Postman</strong>, <strong>Insomnia</strong>,
        or <strong>curl</strong> to make requests (interactive docs at <a href="/docs">/docs</a>).</p>
        <ul>
            <li><code>GET /capabilities</code> &mdash; Machine-readable parameter metadata</li>
            <li><code>GET /health</code> &mdash; Check server status</li>
            <li><code>POST /synthesize</code> &mdash; Generate cloned speech</li>
        </ul>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Check whether the server is alive and the model is loaded."""
    if _runtime is None:
        logger.warning("Health check: model not yet loaded.")
    return HealthResponse()


@app.post(
    "/synthesize",
    response_model=SynthesisResponse,
    tags=["Synthesis"],
    summary="Synthesize speech from text + reference audio + transcript",
)
def synthesize(req: SynthesisRequest) -> SynthesisResponse:
    """
    Synthesize audio using the provided text, reference audio, and transcript.

    Qwen3-TTS (via mlx-audio) performs zero-shot voice cloning in ICL mode: a
    short reference audio clip and its exact transcript are kept in the
    model's context, and new speech is generated in the same voice.

    The full parameter list is documented at GET /capabilities; the request
    schema mirrors it exactly (same model, no drift).
    """
    runtime = _get_runtime()

    # Resolve randomised seed for reproducibility.
    seed = req.seed if req.seed is not None else random.randint(SEED_MIN, SEED_MAX)

    # docs/02: the API speaks two-letter codes; the engine wants lowercase
    # names.  'auto' passes through (the engine's own auto-detection mode).
    engine_language = LANGUAGE_CODE_TO_NAME.get(req.language, req.language)

    logger.info(
        "Synthesizing: seed={}, text_len={}, ref_text_len={}, "
        "lang={} (engine: {})",
        seed,
        len(req.text),
        len(req.reference_text),
        req.language,
        engine_language,
    )

    try:
        raw_audio = decode_base64(req.audio_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")

    _check_reference_audio(raw_audio)

    try:
        ref_wav, ref_sr = _decode_wav(raw_audio)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not decode reference audio: {exc}"
        )

    ref_audio_for_model = _prepare_ref_audio(ref_wav, ref_sr, runtime.sample_rate)

    try:
        t0 = time.perf_counter()

        with _synthesis_lock:
            mx.random.seed(seed)

            chunks: list[np.ndarray] = []
            sr = runtime.sample_rate
            for result in runtime.model.generate(
                text=req.text,
                lang_code=engine_language,
                ref_audio=ref_audio_for_model,
                ref_text=req.reference_text,
                stream=False,
            ):
                mx.eval(result.audio)
                chunks.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
                sr = result.sample_rate

        time_used = time.perf_counter() - t0

        if not chunks:
            logger.warning(
                "Synthesis produced no audio: seed={}, text={!r}, lang={} "
                "(engine: {})",
                seed,
                req.text,
                req.language,
                engine_language,
            )
            raise HTTPException(
                status_code=500,
                detail="The model produced no audio for the supplied text.",
            )

        wav = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

        rtf = compute_rtf(time_used, len(wav), sr)

        audio_bytes = _numpy_to_wav_bytes(wav, sr)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        audio_duration = len(wav) / sr if sr else 0.0
        logger.info(
            "Synthesis complete: {:.1f} s wall-clock, {:.1f} s audio, RTF={}",
            time_used,
            audio_duration,
            f"{rtf:.3f}" if rtf is not None else "n/a",
        )

        return SynthesisResponse(
            audio_base64=audio_b64,
            sample_rate=sr,
            seed=seed,
            fid=str(uuid.uuid4()),
            time_used=time_used,
            rtf=rtf,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Synthesis failed: {}", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/stream",
    tags=["Synthesis"],
    summary="Stream speech incrementally from text + reference audio",
)
def stream(req: StreamRequest, request: Request) -> StreamingResponse:
    """
    Native MLX model-level streaming (mlx-audio's own incremental decoder).

    This is NOT long-text chunking: there is no text splitting, no silence
    joining, no crossfade -- the model itself yields incremental audio
    chunks as it generates, and this endpoint forwards each chunk to the
    client as raw little-endian float32 PCM (pcm_f32le), mono, 24 kHz, as
    soon as it is produced.

    Conditioning mode is inferred from ``reference_text`` (no explicit mode
    field): supplying it selects ICL voice cloning, matching /synthesize's
    conditioning; omitting it selects x-vector (speaker-embedding) voice
    cloning instead.

    GET /capabilities describes transport and conditioning support under
    ``streaming``; the full request schema is StreamRequest (also in /docs).
    """
    runtime = _get_runtime()

    seed = req.seed if req.seed is not None else random.randint(SEED_MIN, SEED_MAX)
    engine_language = LANGUAGE_CODE_TO_NAME.get(req.language, req.language)
    icl = req.reference_text is not None

    logger.info(
        "Streaming: seed={}, text_len={}, icl={}, lang={} (engine: {}), "
        "streaming_interval={}",
        seed,
        len(req.text),
        icl,
        req.language,
        engine_language,
        req.streaming_interval,
    )

    try:
        raw_audio = decode_base64(req.audio_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")

    _check_reference_audio(raw_audio)

    try:
        ref_wav, ref_sr = _decode_wav(raw_audio)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not decode reference audio: {exc}"
        )

    return _MLXStreamingResponse(
        request, req, runtime, ref_wav, ref_sr, seed, engine_language
    )


class _MLXStreamingResponse(StreamingResponse):
    """Keep one request's MLX state on one worker for its complete lifetime.

    Only streaming execution is replaced. Starlette still owns __call__,
    ASGI-version handling, its disconnect listener, and background tasks.
    """

    def __init__(self, request, req, runtime, ref_wav, ref_sr, seed, engine_language):
        super().__init__(
            content=(),  # stream_response drives the worker, not body_iterator.
            media_type="application/octet-stream",
            headers={
                "X-Audio-Format": STREAM_AUDIO_FORMAT,
                "X-Sample-Rate": str(runtime.sample_rate),
                "X-Audio-Channels": str(STREAM_CHANNELS),
            },
        )
        self.request = request
        self.req = req
        self.runtime = runtime
        self.ref_wav = ref_wav
        self.ref_sr = ref_sr
        self.seed = seed
        self.engine_language = engine_language

    async def stream_response(self, send):
        # One invocation, not one dispatch per next(): pooled calls do not
        # guarantee affinity. Do not abandon a worker still owning the lock.
        await anyio.to_thread.run_sync(self._stream_on_worker, send, abandon_on_cancel=False)

    @staticmethod
    def _on_loop(callback, *args):
        try:
            return anyio.from_thread.run(callback, *args)
        except FutureCancelledError:
            # AnyIO's asyncio bridge translates loop cancellation into this
            # exception. Restore scope cancellation when the inherited older-
            # ASGI listener cancelled us; never swallow unrelated failures.
            anyio.from_thread.check_cancelled()
            raise

    def _disconnected(self):
        # On older ASGI the inherited listener may consume the disconnect
        # itself. Its scope cancellation must also stop this worker.
        anyio.from_thread.check_cancelled()
        return self._on_loop(self.request.is_disconnected)

    def _stream_on_worker(self, send):
        anyio.from_thread.check_cancelled()
        ref_audio = _prepare_ref_audio(
            self.ref_wav, self.ref_sr, self.runtime.sample_rate
        )
        chunk_iter = _stream_pcm_chunks(
            self.runtime,
            text=self.req.text,
            engine_language=self.engine_language,
            ref_audio_for_model=ref_audio,
            reference_text=self.req.reference_text,
            seed=self.seed,
            streaming_interval=self.req.streaming_interval,
        )
        try:
            # No HTTP 200 until generation has produced its first PCM chunk.
            try:
                first_chunk = next(chunk_iter)
            except StopIteration:
                logger.warning("Streaming produced no audio: seed={}", self.seed)
                raise HTTPException(
                    status_code=500,
                    detail="The model produced no audio for the supplied text.",
                )
            except Exception as exc:
                logger.error("Streaming setup failed: {}", exc, exc_info=True)
                raise HTTPException(status_code=500, detail=str(exc))

            if self._disconnected():
                return
            self._on_loop(send, {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            })
            self._on_loop(send, {
                "type": "http.response.body", "body": first_chunk, "more_body": True,
            })
            while not self._disconnected():
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    break
                # A disconnect during synchronous next() is handled only
                # after it finishes. Never interrupt MLX or migrate cleanup.
                if self._disconnected():
                    return
                self._on_loop(send, {
                    "type": "http.response.body", "body": chunk, "more_body": True,
                })
        finally:
            # Only iterator ownership belongs here. Engine cleanup stays in
            # _stream_pcm_chunks and runs on this same worker before return.
            chunk_iter.close()

        self._on_loop(send, {
            "type": "http.response.body", "body": b"", "more_body": False,
        })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_reference_audio(raw_bytes: bytes) -> None:
    """Header-only decode to reject undecodable or too-short reference clips."""
    try:
        info = sf.info(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not decode reference audio: {exc}"
        )
    duration = info.frames / info.samplerate if info.samplerate else 0.0
    if duration < MIN_REF_DURATION_S:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Reference audio is {duration:.2f} s long; at least "
                f"{MIN_REF_DURATION_S:.0f} s is required for usable voice cloning."
            ),
        )


def _decode_wav(raw_bytes: bytes) -> tuple[np.ndarray, int]:
    """Full decode to a mono 1-D float32 waveform plus its sample rate."""
    wav, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    if wav.ndim > 1:  # multi-channel -> mono (the engine expects 1-D)
        wav = wav.mean(axis=1)
    return wav, sr


def _prepare_ref_audio(ref_wav: np.ndarray, ref_sr: int, sample_rate: int):
    """Convert a decoded mono float32 waveform into an mx.array at the
    model's sample rate, resampling first if necessary.
    """
    arr = np.asarray(ref_wav, dtype=np.float32)
    if arr.ndim > 1:  # _decode_wav already returns mono; defensive only
        arr = arr.reshape(-1)

    if ref_sr != sample_rate:
        arr = resample_audio(arr, ref_sr, sample_rate)

    return mx.array(arr)


def _numpy_to_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    """Convert a numpy audio array to WAV-encoded bytes (PCM_16)."""
    buffer = io.BytesIO()
    # Clip to [-1, 1]: the PCM_16 conversion wraps out-of-range floats instead
    # of clamping them, which would produce crackling artifacts.
    sf.write(
        buffer,
        np.clip(audio_array, -1.0, 1.0),
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return buffer.getvalue()


def _audio_chunk_to_pcm_bytes(audio) -> bytes:
    """Convert one GenerationResult.audio chunk to raw PCM bytes.

    The fixed /stream wire format (pcm_f32le): IEEE-754 float32, explicit
    little-endian byte order (``"<f4"``, rather than relying on host
    endianness), mono, no header. A small, separately-named function so
    tests can monkeypatch it exactly like ``_numpy_to_wav_bytes`` above,
    without depending on a real MLX/numpy array in the request path.
    """
    mx.eval(audio)
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    return arr.astype("<f4", copy=False).tobytes()


def _stream_pcm_chunks(
    runtime: Qwen3TTSMLXRuntime,
    *,
    text: str,
    engine_language: str,
    ref_audio_for_model,
    reference_text: str | None,
    seed: int,
    streaming_interval: float,
):
    """Yield raw little-endian float32 PCM bytes for one /stream request.

    Holds the shared ``_synthesis_lock`` for the ENTIRE generator lifetime
    (not just around individual chunks), so /stream can never run
    concurrently with /synthesize or another /stream against the shared MLX
    model state -- mlx-audio's global RNG (seeded per request), its internal
    ICL cache, and (for stream-vs-stream specifically) the speech tokenizer
    decoder's incremental conv/KV-cache buffers, which mlx-audio
    unconditionally resets at the *start* of every streaming call. A `with`
    block correctly spans the `yield` statements below: the lock is held
    across suspension points and is only released once this generator is
    exhausted, raises, or is closed (client disconnect/cancellation delivers
    a GeneratorExit here via Python's normal generator-close machinery).

    Deterministic cleanup runs in `finally` on every exit path -- normal
    completion, an exception (before or after the first chunk), or an early
    `.close()`: the underlying MLX generator is explicitly closed, the
    streaming decoder state is reset, and the MLX cache is cleared. mlx-audio
    0.5.4 performs its own equivalent cleanup only on normal completion (no
    try/finally around its own streaming loop), so an aborted stream must
    not rely on it -- this is exactly that server-side safety net.
    """
    mlx_gen = None
    with _synthesis_lock:
        try:
            mx.random.seed(seed)
            mlx_gen = runtime.model.generate(
                text=text,
                lang_code=engine_language,
                ref_audio=ref_audio_for_model,
                ref_text=reference_text,
                stream=True,
                streaming_interval=streaming_interval,
            )
            for result in mlx_gen:
                yield _audio_chunk_to_pcm_bytes(result.audio)
        finally:
            if mlx_gen is not None:
                try:
                    mlx_gen.close()
                except Exception:
                    logger.exception("Error closing MLX streaming generator")
            try:
                runtime.model.speech_tokenizer.decoder.reset_streaming_state()
            except Exception:
                logger.exception("Error resetting MLX streaming decoder state")
            try:
                mx.clear_cache()
            except Exception:
                logger.exception("Error clearing MLX cache")


# ---------------------------------------------------------------------------
# Main (for running directly: python server_qwen3TTS_mlx.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("QWEN3TTS_MLX_HOST", "0.0.0.0")
    port = int(os.getenv("QWEN3TTS_MLX_PORT", "7500"))
    logger.info("Starting Qwen3-TTS MLX REST API server on %s:%d", host, port)
    # Pass the app object directly instead of a module path string,
    # so this works regardless of how the file is invoked.
    uvicorn.run(app, host=host, port=port, log_level="info")
