"""
FastAPI REST server for KittenTTS 2 (github.com/KittenML/KittenTTS).

KittenTTS 2 is a 1.7B-parameter 1-bit speech language model with in-context
voice cloning and expression control.  The server exposes its two speaker
sources through one endpoint: the model's built-in voices (115 of them,
selected by the `voice` field) and zero-shot cloning from a reference clip
(`audio_base64`, 5--30 s of a single speaker).  Omitting both uses the
checkpoint's own default voice (Bruno for KittenML/kitten-tts-2).  The
server returns base64-encoded 24 kHz WAV.

Expression control travels inside `text`: a leading `[emotion]` tag, inline
`<event>` tags and `(((emphasis)))` spans reach the model as markup
(engine docs: early beta -- they steer delivery, they do not guarantee it).

`seed` is meaningful: the engine seeds its sampler per generation
(`advanced={"seed": ...}`), so the same seed, text and speaker reproduce
each other.  Reproducibility is bit-exact on CPU; on CUDA some kernels are
non-deterministic, so do not promise bit-exact GPU output.

The language is carried by the *speaker*, not by a request parameter: the
engine has no `language` argument (non-English voices are named after their
language, e.g. "German").  This server therefore accepts any well-formed
two-letter code but does not forward it, and capabilities advertise
`languages: null` (the docs/02 no-support case).  Related gotcha: when a
clone omits `reference_text`, the engine transcribes the clip with Whisper
*in English mode* -- pass `reference_text` for non-English references.

NOTE: this server implements the KittenTTS **2** API.  The same `kittenml`
package also loads the legacy 0.8 ONNX models through the same constructor,
but those speak a different generate() signature (no cloning, different
voice names); pointing KITTENTTS_MODEL at an ONNX repository fails fast at
startup rather than confusing clients at request time.

Model weights are downloaded from HuggingFace (KittenML/kitten-tts-2, plus
the s3gen vocoder weights from ResembleAI/chatterbox-turbo) on first start.
The Python backend is not speed-optimized (engine docs): budget ~8 GB free
VRAM for CUDA, or ~6 GB of system RAM for CPU inference.

Configuration (environment variables):
    KITTENTTS_MODEL     HuggingFace id or local path to a KittenTTS 2 model
                        repository directory.  Default: KittenML/kitten-tts-2
    KITTENTTS_DEVICE    Device for the model: 'auto' (engine picks -- CUDA
                        when available, else CPU), 'cuda', 'mps' or 'cpu'.
                        Default: auto.  'cuda:<index>' is deliberately NOT
                        accepted: the engine's own device handling compares
                        the value against the literal 'cuda'
                        (kittenml/kittentts2/speaker.py::_choose_device), so
                        an indexed value silently runs the reference-clip
                        transcription on CPU instead.  Pin a GPU with
                        CUDA_VISIBLE_DEVICES instead.  The shared grammar
                        (docs/01 D8) is checked at import time; 'cuda' on a
                        box without CUDA fails before the model is loaded.
    KITTENTTS_DECODER   Audio decoder name from the model repository.
                        Unset keeps the repository default ('default', the
                        stock flow).  'student_w4' is a distilled, quantised
                        flow that saves ~420 MB of memory at a quality cost
                        (engine docs/decoders.md).
    KITTENTTS_WEIGHTS   Language-model weight packing: 'packed' (lossless,
                        default), 'emb4' (half the download, quantised
                        embedding) or 'full' (bf16).  Unset keeps the
                        repository default.
    KITTENTTS_HOST      Bind host for `python server_kittenTTS.py`.
                        Default: 0.0.0.0
    KITTENTTS_PORT      Bind port for `python server_kittenTTS.py`.
                        Default: 7500

Install the engine:
    pip install kittenml
Or use the uv environment in envs/kittentts/ (python3 tools/serve.py kittentts).

Extra dependencies beyond the KittenTTS package:
    pip install fastapi uvicorn loguru soundfile
    pip install ../tts-engine-common # in-repo copy; or: pip install -e ../tts-engine-common

Usage:
    python server_kittenTTS.py
    # or: uvicorn server_kittenTTS:app --host 0.0.0.0 --port 7500
"""

from __future__ import annotations

import base64
import io
import os
import random
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kittenml import KittenTTS

from tts_engine_common import (
    DEFAULT_LANGUAGE,
    CoreHealthResponse,
    RELEASE_VERSION,
    CoreSynthesisResponse,
    build_capabilities,
    capabilities_endpoint,
    compute_rtf,
    decode_base64,
    normalize_language,
    parse_device,
    require_cuda_available,
    stage_audio,
    temp_audio_dir,
    validate_language_code,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME_OR_PATH = os.getenv("KITTENTTS_MODEL", "KittenML/kitten-tts-2")
DEVICE_ENV = "KITTENTTS_DEVICE"
DEVICE_SETTING = os.getenv(DEVICE_ENV, "auto").strip().lower()

# Decoder / weight-pack overrides; empty means "the repository's own default"
# (config.json's default_decoder / default_weights).  Decoder *names* are
# per-repository data, so the engine validates them at load (its error lists
# the available names); weight packings are a fixed engine vocabulary.
DECODER = os.getenv("KITTENTTS_DECODER", "").strip()
WEIGHTS = os.getenv("KITTENTTS_WEIGHTS", "").strip().lower()

_SUPPORTED_WEIGHTS = ("", "packed", "emb4", "full")

# What this engine can serve.  Grammar rules are shared (docs/01 D8); this
# replaces the old membership tuple, which could not name the families the
# engine speaks in the engine's own order.
#
# 'cuda:<index>' is rejected on purpose, even though the language model itself
# would happily take one (kittentts2.model._resolve_device returns the string
# verbatim).  The *transcriber* does not: speaker.py::Whisper._choose_device
# asks `if self.device != "cuda": return "cpu"`, so 'cuda:1' silently moves the
# on-the-fly reference-clip transcription to the CPU -- slower, and per the
# engine's own warning a different transcript and therefore a different cloned
# voice.  A syntax that works for one subsystem and silently degrades another
# is worse than no syntax: use CUDA_VISIBLE_DEVICES=1 to pin the GPU.
DEVICE_FAMILIES = ("auto", "cuda", "mps", "cpu")
DEVICE_ALLOW_INDEX = False
DEVICE_SPEC = parse_device(
    DEVICE_SETTING,
    env_var=DEVICE_ENV,
    families=DEVICE_FAMILIES,
    allow_index=False,
    note="the engine's transcriber compares the device to the literal 'cuda'; "
    "pin a GPU with CUDA_VISIBLE_DEVICES",
)


def _check_device_available() -> None:
    """Raise unless this box can serve the configured CUDA request.

    'auto' is the engine's own CUDA-else-CPU pick, so nothing to check there;
    an explicit 'cuda:N' is an instruction, and torch would only notice the
    ordinal is missing once the weights are already loaded.
    """
    require_cuda_available(
        DEVICE_SPEC,
        is_available=torch.cuda.is_available,
        device_count=torch.cuda.device_count,
        env_var=DEVICE_ENV,
    )


def _validate_config() -> None:
    """Fail fast on bad configuration instead of partway through a model download."""
    if WEIGHTS not in _SUPPORTED_WEIGHTS:
        raise ValueError(
            "KITTENTTS_WEIGHTS must be empty (repository default), or one of "
            f"{_SUPPORTED_WEIGHTS[1:]}, got {WEIGHTS!r}"
        )


_validate_config()

# With 'auto', mirror the engine's own decision (kittentts2.model._resolve_device:
# CUDA when available, else CPU) for the capabilities/health documents.  The
# device the model actually lands on is re-read from the loaded model at runtime.
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if DEVICE_SETTING == "auto" else DEVICE_SETTING

# KittenTTS 2 decodes through the s3gen vocoder at 24 kHz.  Confirmed against
# the engine's own S3GEN_SR (kittenml/kittentts2/s3gen.py) and the checkpoint's
# config.json "sample_rate", not assumed.  Responses echo the loaded model's
# own `sample_rate` property; this constant describes /capabilities at import time.
SAMPLE_RATE = 24000

SEED_MIN = 1
SEED_MAX = 1000

# Engine docs: clone from 5--30 s of a single speaker.  Below 6 s the s3gen
# conditioning *tiles* the clip (repeats the speech rather than padding
# silence), and below 5 s the clone quality drops off the documented range;
# reject the documented floor with a clear 400 instead of tiling garbage.
MIN_PROMPT_DURATION_S = 5.0

# Sanity valve for the request payload (~5 min of 24 kHz audio).  Not a model
# limitation -- conditioning uses far less.
MAX_AUDIO_B64_LEN = 10_000_000

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class SynthesisRequest(BaseModel):
    """A single synthesis request. Unknown fields are rejected (422)."""

    model_config = ConfigDict(extra="forbid")

    # --- core vocabulary (tts_engine_common.CORE_FIELDS) -------------------
    text: str = Field(
        ...,
        min_length=1,
        description=(
            "Text to synthesize, e.g. 'Hello there'.  Expression markup is "
            "understood inside the text: a leading [emotion] tag "
            "(angry/contemplative/excited/joyful/mundane/nervous/sad/stern/"
            "surprised/tender), inline <event> tags (gasp/giggle/growl/gulp/"
            "laugh/pause/scoff/sigh/sob/um) and (((emphasis))) spans steer "
            "delivery (beta)."
        ),
    )
    audio_base64: str | None = Field(
        None,
        min_length=1,
        max_length=MAX_AUDIO_B64_LEN,
        description=(
            "Reference voice sample (5--30 s of a single speaker) as a "
            "base64 string, to clone that speaker instead of a built-in "
            "voice.  Any container soundfile can decode (WAV, MP3, OGG, "
            "FLAC, ...).  Mutually exclusive with `voice`; omit both to use "
            "the checkpoint's default voice."
        ),
    )
    reference_text: str | None = Field(
        None,
        description=(
            "Exact transcript of the reference clip.  Strongly recommended: "
            "when omitted the engine transcribes the clip with Whisper, "
            "which runs in English mode -- non-English references need this "
            "field.  Only accepted together with `audio_base64`."
        ),
    )
    language: str | None = Field(
        DEFAULT_LANGUAGE,
        description=(
            "Two-letter language code, e.g. 'en' or 'zh'.  Accepted for API "
            "consistency but not forwarded -- the engine has no language "
            "parameter; the *speaker* carries the language (built-in voices "
            "named after a language, e.g. 'German', or your own reference "
            "clip).  Omitted or empty defaults to 'en'."
        ),
    )
    seed: int | None = Field(
        None,
        ge=SEED_MIN,
        le=SEED_MAX,
        description=(
            "Random seed for reproducibility (the engine seeds its sampler "
            "with it).  If omitted, a random seed "
            f"in [{SEED_MIN}, {SEED_MAX}] is chosen and echoed in the "
            "response."
        ),
    )

    # --- engine-specific tuning (defaults mirror the engine's own defaults) --
    voice: str | None = Field(
        None,
        description=(
            "Built-in voice name from the loaded model (kitten-tts-2 offers "
            "115, e.g. 'Bruno' -- the checkpoint default -- 'Bella', or "
            "language-named voices such as 'German').  Mutually exclusive "
            "with `audio_base64`; unknown names are rejected (422).  Omit "
            "both to use the checkpoint's default voice."
        ),
    )
    preset: str | None = Field(
        None,
        description=(
            "Decoding preset from the loaded model (kitten-tts-2 offers "
            "'stable' (the default: temperature 0.8, top_p 0.8, top_k 50) "
            "and 'expressive' (temperature 0.9, top_p 0.9)).  Unknown names "
            "are rejected (422)."
        ),
    )
    normalize: bool = Field(
        True,
        description=(
            "Expand written forms ('$12.50' -> 'twelve dollars fifty cents') "
            "before synthesis (engine default).  The normalizer is "
            "English-tuned and mangles other languages -- disable it for "
            "non-English text."
        ),
    )
    max_new_tokens: int = Field(
        1000,
        ge=25,
        le=2000,
        description=(
            "Per-chunk token budget (engine default 1000; 25 tokens is about "
            "one second of audio).  Long text is sentence-chunked; each "
            "chunk gets its own budget."
        ),
    )
    temperature: float | None = Field(
        None,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature, overriding the preset (preset defaults: "
            "stable 0.8, expressive 0.9)."
        ),
    )
    top_p: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Nucleus (top-p) threshold, overriding the preset (defaults: 0.8/0.9).",
    )
    top_k: int | None = Field(
        None,
        ge=0,
        le=500,
        description="Top-k sampling cutoff, overriding the preset (default 50; 0 disables top-k).",
    )
    min_p: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Min-p sampling threshold, overriding the preset (default 0.0).",
    )

    @field_validator("text")
    @classmethod
    def _validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain non-whitespace characters")
        return v

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, v: object) -> str:
        # docs/02: null/empty means English.  'mode=before' means ``v`` is
        # the raw JSON value (pre-coercion): non-strings are rejected here
        # as ValueError (422), never downstream as AttributeError (500).
        return normalize_language(v)

    @field_validator("language")
    @classmethod
    def _check_language(cls, v: str) -> str:
        # docs/02: the API speaks two-letter codes.  The engine has no
        # language parameter -- the speaker carries the language -- so
        # there is no 'auto' sentinel either; accept any well-formed code
        # for API consistency (LuxTTS/VoxCPM no-support case).
        return validate_language_code(v)

    @model_validator(mode="after")
    def _reference_text_requires_audio(self) -> "SynthesisRequest":
        # The engine raises ValueError for a transcript without a clip; a
        # transcript with no clip is a client bug, not a speaker switch.
        # Reject at the boundary (422) rather than mapping to a 500.
        if self.reference_text is not None and self.audio_base64 is None:
            raise ValueError(
                "reference_text requires audio_base64 (the clip it transcribes)"
            )
        return self

    @model_validator(mode="after")
    def _voice_and_audio_exclusive(self) -> "SynthesisRequest":
        # The engine's _speaker(): "give either voice= or reference=, not
        # both" -- two speaker sources is ambiguous, reject (422) instead
        # of silently picking one.
        if self.audio_base64 is not None and self.voice is not None:
            raise ValueError("audio_base64 and voice are mutually exclusive; send one")
        return self


class SynthesisResponse(CoreSynthesisResponse):
    """The synthesis result (core fields from tts_engine_common, plus fid)."""

    fid: str = Field(..., description="Request ID (internal).")


class HealthResponse(CoreHealthResponse):
    """The /health core from tts_engine_common, with this engine's serverType.

    model/device pick up the import-time config; version carries the shared
    RELEASE_VERSION constant — overriding it here is how a server drifts.
    """

    serverType: Literal["KittenTTS"] = "KittenTTS"
    model: str = MODEL_NAME_OR_PATH
    device: str = DEVICE


# ---------------------------------------------------------------------------
# Capabilities (derived from SynthesisRequest — single source of truth)
# ---------------------------------------------------------------------------

CAPABILITIES = build_capabilities(
    SynthesisRequest,
    engine="kittentts",
    model=MODEL_NAME_OR_PATH,
    device=DEVICE,
    sample_rate=SAMPLE_RATE,
    watermarked=False,
    endpoint="/synthesize",
    reference_audio={
        "required": False,
        "formats": ["wav", "mp3", "ogg", "flac"],
        "min_duration_s": MIN_PROMPT_DURATION_S,
        "note": (
            "Optional: omit for a built-in `voice` (or the checkpoint's "
            "default voice).  For cloning, 5--30 s of a single speaker works "
            "well.  The vocoder conditions on the first 10 s of the clip and "
            "loops shorter clips internally; the speaker embedding and the "
            "in-context prompt see the whole clip.  Omitting `reference_text` "
            "makes the engine load Whisper and transcribe the clip in "
            "English mode -- supply it for non-English references."
        ),
    },
    languages=None,  # no fixed list; two-letter codes (docs/02), not forwarded
    overrides={
        "normalize": {"advanced": True},
        "max_new_tokens": {"step": 25, "advanced": True},
        "temperature": {"step": 0.05, "advanced": True},
        "top_p": {"step": 0.05, "advanced": True},
        "top_k": {"step": 1, "advanced": True},
        "min_p": {"step": 0.01, "advanced": True},
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded and CUDA cache cleared.")


app = FastAPI(
    title="KittenTTS 2 API",
    description=(
        "REST API around KittenTTS 2.  Send text plus either a built-in "
        "`voice` or a reference clip to clone, and get back speech.  "
        "Expression markup ([emotion] tags, <event> tags, (((emphasis)))) "
        "spans) rides inside the text.  Synthesis requests are serialized "
        "(single shared model).  Machine-readable parameter metadata at GET "
        "/capabilities."
    ),
    version=RELEASE_VERSION,
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
# Runtime — thin wrapper around the KittenTTS 2 model
# ---------------------------------------------------------------------------


@dataclass
class KittenTTSRuntime:
    """Holds the loaded model and its metadata for the lifetime of the server."""

    model: object
    sample_rate: int
    device: str


_runtime: KittenTTSRuntime | None = None

# The engine mutates shared state per call: the speaker embedding is parked
# on the language model (`model._pending_speaker_embedding`) for the duration
# of generate() and cleared afterwards, the reference/prompt caches are plain
# dicts, and the transformers generate() itself is not thread-safe.  Serialize
# synthesis; single-device throughput is the bottleneck anyway.
_synthesis_lock = threading.Lock()


def _get_runtime() -> KittenTTSRuntime:
    """Return the global runtime, loading the model once on first call."""
    global _runtime
    if _runtime is None:
        _check_device_available()
        logger.info(
            "Loading KittenTTS model '{}' on device '{}' (first run downloads "
            "from HuggingFace) ...",
            MODEL_NAME_OR_PATH,
            DEVICE_SETTING,
        )
        model = KittenTTS(
            MODEL_NAME_OR_PATH,
            # 'auto' -> None so the engine applies its own device pick.
            device=None if DEVICE_SETTING == "auto" else DEVICE_SETTING,
            # Empty env -> None so the repository's default applies.
            decoder=DECODER or None,
            weights=WEIGHTS or None,
        )
        # kittenml.KittenTTS dispatches on the repository's config.json: a
        # KITTEN2 repo yields a KittenTTS2 (the API this server implements);
        # an ONNX 0.8 repo yields a KittenTTSOnnx whose generate() takes
        # (text, voice, speed) with no cloning and different voice names.
        # Reject the legacy shape here rather than mis-map every request.
        if not hasattr(model, "decode_presets"):
            raise ValueError(
                f"'{MODEL_NAME_OR_PATH}' is not a KittenTTS 2 repository "
                "(no decode_presets) -- this server implements the "
                "KittenTTS 2 API and cannot drive the legacy 0.8 ONNX "
                "models.  Point KITTENTTS_MODEL at e.g. KittenML/kitten-tts-2."
            )
        # The loaded model's own rate (codec-backed property) and resolved
        # device: echo those rather than the import-time guesses, so an exotic
        # checkpoint stays self-consistent in responses.
        _runtime = KittenTTSRuntime(
            model=model,
            sample_rate=int(model.sample_rate),
            device=str(model.device),
        )
        logger.info(
            "Model loaded successfully. Sampling rate: {} Hz, device: {}, "
            "voices: {}, presets: {}",
            _runtime.sample_rate,
            _runtime.device,
            len(model.available_voices),
            sorted(model.decode_presets),
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, tags=["System"])
def root() -> str:
    """A friendly landing page so browser visitors don't get the auto-generated docs."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head><title>KittenTTS 2 API</title></head>
    <body>
        <h1>KittenTTS 2 REST API</h1>
        <p>Model: <code>{MODEL_NAME_OR_PATH}</code> on <code>{DEVICE}</code>.
        This server is a REST API, not a web server.</p>
        <p>Use a REST client like <strong>Postman</strong>, <strong>Insomnia</strong>,
        or <strong>curl</strong> to make requests (interactive docs at <a href="/docs">/docs</a>).</p>
        <ul>
            <li><code>GET /capabilities</code> &mdash; Machine-readable parameter metadata</li>
            <li><code>GET /health</code> &mdash; Check server status</li>
            <li><code>POST /synthesize</code> &mdash; Generate speech (built-in voice or cloned)</li>
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
    summary="Synthesize speech with a built-in voice or a cloned one",
)
def synthesize(req: SynthesisRequest) -> SynthesisResponse:
    """
    Synthesize audio using the provided text plus a built-in `voice`, a
    reference `audio_base64` clip, or neither (checkpoint default voice).

    The full parameter list is documented at GET /capabilities; the request
    schema mirrors it exactly (same model, no drift).
    """
    runtime = _get_runtime()

    # Voice and preset names live in the model repository, not in the engine
    # package, so they cannot be Literal enums on the request model -- check
    # them here and 422 rather than letting the engine's ValueError surface
    # as a 500.
    if req.voice is not None and req.voice not in runtime.model.available_voices:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown voice {req.voice!r}; the model offers "
                f"{len(runtime.model.available_voices)} voices (e.g. "
                f"{sorted(runtime.model.available_voices)[:5]}). "
                "See the model card for the full list."
            ),
        )
    if req.preset is not None and req.preset not in runtime.model.decode_presets:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown preset {req.preset!r}; this model offers "
                f"{sorted(runtime.model.decode_presets)}"
            ),
        )

    # Resolve randomised seed.
    seed = req.seed if req.seed is not None else random.randint(SEED_MIN, SEED_MAX)

    logger.info(
        "Synthesizing: seed={}, text_len={}, speaker={}, preset={}, "
        "normalize={}, max_new_tokens={}, lang={} (not forwarded)",
        seed,
        len(req.text),
        f"clone({len(req.reference_text or '')} chars transcript)"
        if req.audio_base64
        else f"voice:{req.voice}" if req.voice
        else "default-voice",
        req.preset or "<repo default>",
        req.normalize,
        req.max_new_tokens,
        req.language,
    )

    # The engine loads reference audio from a file path.  It *caches* the
    # derived prompt/embedding per path (KittenTTS2._reference_cache keyed by
    # (path, transcript), S3Codec keyed by path), so the clip is staged as a
    # content-hashed file that is deliberately *kept*: repeat requests with
    # the same clip reuse the cache.  Wipe the staging directory to reclaim
    # disk.
    reference_path: str | None = None
    if req.audio_base64 is not None:
        try:
            raw_audio = decode_base64(req.audio_base64)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")

        _check_reference_audio(raw_audio)

        reference_path = stage_audio(raw_audio, _TEMP_AUDIO_DIR)

    try:
        # Time the actual synthesis call.
        t0 = time.perf_counter()

        with _synthesis_lock:
            # The engine seeds its sampler itself via advanced={"seed": ...}
            # (torch.manual_seed, once per generation); seed_everything keeps
            # Python/NumPy deterministic too, matching the other servers.
            seed_everything(seed)
            wav = runtime.model.generate(
                text=req.text,
                voice=req.voice,
                reference=reference_path,
                reference_text=req.reference_text,
                preset=req.preset,  # None -> repository default preset
                max_new_tokens=req.max_new_tokens,
                normalize=req.normalize,
                # Sampling overrides: the engine ignores None values and keeps
                # the preset's own numbers, so pass them through as-is.
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                min_p=req.min_p,
                advanced={"seed": seed},
            )

        time_used = time.perf_counter() - t0

        # generate() returns a 1-D float32 numpy array at model.sample_rate.
        audio_array = np.asarray(wav).reshape(-1)
        sample_rate = runtime.sample_rate

        rtf = compute_rtf(time_used, len(audio_array), sample_rate)

        # Encode the output WAV to base64.
        audio_bytes = _numpy_to_wav_bytes(audio_array, sample_rate)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        audio_duration = len(audio_array) / sample_rate if sample_rate else 0.0
        logger.info(
            "Synthesis complete: {:.1f} s wall-clock, {:.1f} s audio, RTF={}",
            time_used,
            audio_duration,
            f"{rtf:.3f}" if rtf is not None else "n/a",
        )

        return SynthesisResponse(
            audio_base64=audio_b64,
            sample_rate=sample_rate,
            seed=seed,
            fid=str(uuid.uuid4()),
            time_used=time_used,
            rtf=rtf,
        )

    except Exception as exc:
        logger.error("Synthesis failed: {}", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMP_AUDIO_DIR = temp_audio_dir("kittentts_rest_api")


def seed_everything(seed: int) -> None:
    """Set the random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _check_reference_audio(raw_bytes: bytes) -> None:
    """Header-only decode to reject undecodable or too-short reference clips."""
    try:
        info = sf.info(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not decode reference audio: {exc}"
        )
    duration = info.frames / info.samplerate if info.samplerate else 0.0
    if duration < MIN_PROMPT_DURATION_S:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Reference audio is {duration:.2f} s long; at least "
                f"{MIN_PROMPT_DURATION_S:.0f} s is required for usable voice cloning."
            ),
        )


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


# ---------------------------------------------------------------------------
# Main (for running directly: python server_kittenTTS.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("KITTENTTS_HOST", "0.0.0.0")
    port = int(os.getenv("KITTENTTS_PORT", "7500"))
    logger.info("Starting KittenTTS REST API server on {}:{}", host, port)
    # Pass the app object directly instead of a module path string,
    # so this works regardless of how the file is invoked.
    uvicorn.run(app, host=host, port=port, log_level="info")
