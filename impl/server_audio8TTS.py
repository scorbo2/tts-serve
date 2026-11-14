"""
FastAPI REST server for Audio8 TTS zero-shot voice cloning
(github.com/Edge0-AI/Audio8_TTS, checkpoint Edge0/Audio8-TTS-Realistic-Air).

Audio8 is a compact 0.6B DualAR model (slow AR predicts one semantic token per
audio frame; a 4-layer fast AR fills in the frame's 10 codec codebooks).  The
codec runs at 44.1 kHz with 2048 samples per model frame, so one frame is
~21.5 ms of audio.  Clients send text plus an optional reference clip (base64)
and get back base64-encoded 44.1 kHz WAV.

Two synthesis modes, both through the same generate() call:

1. **Voice cloning** (reference clip + its exact transcript) -- the clip is
   encoded into the context and the model continues from it.  The transcript
   is not optional: the processor refuses a clip with no transcript
   ("reference_text is required when a reference voice is provided"), so this
   server requires the pair (422 otherwise) instead of letting the engine 500.
2. **No reference** -- the model's own default voice.  Supported by the engine
   (guides/inference.md shows it as a first-class mode), but the voice is not
   selectable: no preset, no voice design.

`seed` is meaningful (Python/NumPy/PyTorch RNGs plus a dedicated
torch.Generator for sampling, exactly like the engine CLI).  Measured on a
real GPU: a warm repeat of the same request is bit-identical, but the FIRST
request after startup differs from the later ones with the same seed (kernel
selection settles during the first run), so only warm repeats reproduce.

`language` is accepted but never forwarded: the engine has no language
parameter at all.  Realistic Air speaks 23 languages (including Thai and
Swedish) and detects which one from the input text, so capabilities advertise
`languages: null` (the docs/02 no-support case -- the VoxCPM/LuxTTS/Breeze
pattern).

The model's 2048 packed text+audio positions are a shared budget: the
reference clip costs ~21.5 positions per second of clip and the text costs one
per token.  Both are capped at the request boundary (`MAX_PROMPT_DURATION_S`,
`MAX_TEXT_CHARACTERS`) and the prompt is measured exactly after tokenization,
so "prompt too long" is an actionable 400 rather than an engine ValueError
surfacing as a 500.  What the prompt *leaves over* is measured too: a request
whose prompt eats all but a handful of positions is a 400 naming the numbers,
not a 200 carrying one frame of audio (measured: 1500 Devanagari characters plus
a 19.4 s clip left 151 frames, and the engine returned 0.0 s and `finished:
true`).  That floor is a bracket, not a cure -- the same script with 651 frames
left over still collapsed to 0.05 s on its own, which is why the docs push the
engine's own 150-character guidance.  The frame budgets are clamped with the same
arithmetic the engine applies, so the bad-case retry is never paid for twice when
both passes would land on the same ceiling.

Reference clips are decoded in memory (no temp files, no path-keyed cache).
Their duration is enforced on the samples that actually decode, not on the
container header, and an MP3 with no length table (Xing/Info/VBRI) is refused:
without one a decoder guesses the duration from the first frame's bitrate and
then stops at the guess, which would hand the model a silent fraction of the
clip while reporting nothing (measured: a 40.05 s reference arriving as 6.73 s).

Model weights are downloaded from HuggingFace (Edge0/Audio8-TTS-Realistic-Air)
on first start.  The checkpoint ships its own modeling/processor code, so the
loader runs it with trust_remote_code=True -- see impl/server_audio8TTS.md for
what that means.  Set HF_TOKEN if you mirror the checkpoint into a private repo.

Configuration (environment variables):
    AUDIO8_TTS_MODEL   HuggingFace id of a complete checkpoint (with its
                       bundled codec.pth) or a local directory holding one.
                       Default: Edge0/Audio8-TTS-Realistic-Air
    AUDIO8_TTS_DEVICE  Device to load on: 'auto' (CUDA when available, else
                       CPU), 'cpu', 'cuda', or 'cuda:<index>'.  Default: cuda.
                       The shared grammar (docs/01 D8) is checked at import
                       time; a device this box cannot serve (CUDA requested
                       without CUDA, or an index past the last GPU) fails at
                       model load -- before the checkpoint is fetched, never
                       after.  /health and /capabilities echo the *configured*
                       string rather than the resolved device, so 'auto'
                       reports as 'auto' even when it lands on CPU.
                       The 0.6B is a GPU model -- CPU runs (float32 only) but
                       measured RTF ~6.6 on a 24-core x86 box, i.e. six times
                       slower than real time.
    AUDIO8_TTS_DTYPE   'auto' (bfloat16 on CUDA, float32 on CPU), 'bfloat16',
                       'float16', or 'float32'.  Default: auto.  CPU requires
                       float32 (the engine's own rule); checked at load.
    AUDIO8_TTS_HOST    Bind host for `python server_audio8TTS.py`.
                       Default: 0.0.0.0
    AUDIO8_TTS_PORT    Bind port for `python server_audio8TTS.py`.
                       Default: 7500

The engine publishes no Python package: the checkpoint carries its own model
code, so the install is just the loader plus this server's dependencies:
    pip install "torch>=2.5" "torchaudio>=2.5" "transformers>=4.57,<5"
    pip install fastapi uvicorn loguru soundfile
    pip install ../tts-engine-common   # in-repo copy; or: pip install -e ../tts-engine-common

`torchaudio` is needed only when a reference clip has to be resampled (the
processor resamples anything that is not 44.1 kHz), which is most clips.

Usage:
    python server_audio8TTS.py
    # or: uvicorn server_audio8TTS:app --host 0.0.0.0 --port 7500
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
from typing import Any, Literal, NamedTuple

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Audio8 ships no package of its own -- the checkpoint's auto_map points at the
# modeling/processor code bundled with the weights, loaded through transformers
# with trust_remote_code=True.  A missing *transformers* is therefore the
# actionable "install the engine" case; anything else is a broken transitive
# dependency that a reinstall would not fix, so re-raise those untouched.
try:
    from transformers import AutoModel, AutoProcessor
except ImportError as exc:
    if getattr(exc, "name", None) not in ("transformers", "AutoModel", "AutoProcessor"):
        raise
    raise ImportError(
        "This server loads the Audio8 checkpoint through transformers "
        "(the model code ships inside the checkpoint). Install it:\n"
        '    pip install "torch>=2.5" "torchaudio>=2.5" "transformers>=4.57,<5"\n'
        "The `dtype=` argument used below needs transformers >= 4.57."
    ) from exc

from tts_engine_common import (
    DEFAULT_LANGUAGE,
    CoreHealthResponse,
    DeviceSpec,
    RELEASE_VERSION,
    CoreSynthesisResponse,
    build_capabilities,
    capabilities_endpoint,
    compute_rtf,
    decode_base64,
    normalize_language,
    parse_device,
    require_cuda_available,
    validate_language_code,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME_OR_PATH = os.getenv("AUDIO8_TTS_MODEL", "Edge0/Audio8-TTS-Realistic-Air")
DEVICE_ENV = "AUDIO8_TTS_DEVICE"
DEVICE_SETTING = os.getenv(DEVICE_ENV, "cuda")
DTYPE = os.getenv("AUDIO8_TTS_DTYPE", "auto").strip().lower()

# What *this engine* can serve, per the engine CLI's own --device grammar
# (audio8_tts_infer.py).  MPS is deliberately absent: the engine neither
# documents nor tests it.  The grammar rules themselves are shared (docs/01
# D8) -- this declaration is capability data, not another regex copy.
DEVICE_FAMILIES = ("auto", "cpu", "cuda")
DEVICE_ALLOW_INDEX = True
DEVICE_SPEC = parse_device(DEVICE_SETTING, env_var=DEVICE_ENV, families=DEVICE_FAMILIES)
# The canonical device string is what the grammar validated, not the raw env
# value: parse_device strips surrounding whitespace, and everything downstream
# (the loader, /health, /capabilities) must see exactly what was accepted.
DEVICE = DEVICE_SPEC.requested

_DTYPES = ("auto", "bfloat16", "float16", "float32")


def _validate_config() -> None:
    """Fail fast on bad configuration instead of partway through a model download."""
    if DTYPE not in _DTYPES:
        raise ValueError(f"AUDIO8_TTS_DTYPE must be one of {_DTYPES}, got {DTYPE!r}")


_validate_config()

# ---------------------------------------------------------------------------
# Engine constants (read off the checkpoint's own config.json, not assumed)
# ---------------------------------------------------------------------------

# ArkttsConfig.codec_sample_rate: the bundled codec's output rate.  Note this
# is NOT the 24 kHz of most tts-serve engines (nor VoxCPM's 48 kHz).
SAMPLE_RATE = 44100

# ArkttsCodec.frame_length: codec samples per model frame -> ~21.5 frames per
# second of audio.  A reference clip costs one context position per frame.
CODEC_FRAME_SIZE = 2048

# Derived, never hardcoded: how many model frames one second of audio costs.
# Used for every "~N seconds of audio" figure the server shows a client, so
# the prose cannot drift from the arithmetic.
FRAMES_PER_SECOND = SAMPLE_RATE / CODEC_FRAME_SIZE

# ArkttsConfig.max_seq_len: packed text+audio positions the model can attend
# to.  The reference clip, the transcript, the target text, and every generated
# frame all come out of this one budget.
MAX_CONTEXT_POSITIONS = 2048

# max_new_tokens is a frame budget, so anything past MAX_CONTEXT_POSITIONS is
# clamped by the engine (to whatever the prompt leaves room for).
MAX_NEW_TOKENS_LIMIT = MAX_CONTEXT_POSITIONS

SEED_MIN = 1
SEED_MAX = 1000

# Heuristic floor, not an engine limit (a 1 s clip *runs*): below ~2 s the
# speaker conditioning is too thin to hold a voice.  Same floor as the other
# tts-serve cloners.
MIN_PROMPT_DURATION_S = 2.0

# Ceiling, and this one IS arithmetic: 20 s of clip is ~430 of the 2048 context
# positions, which leaves room for MAX_TEXT_CHARACTERS of text (see below).
# The engine truncates nothing it can measure, so an over-long clip would
# silently starve the generation (or blow the prompt budget outright).  The
# engine's own demo references run 3--18 s.
MAX_PROMPT_DURATION_S = 20.0

# How far the duration a container *advertises* (sf.info) may disagree with
# what the stream actually decodes to before the discrepancy becomes an error.
# 0.5 s is far more than codec priming/padding ever accounts for (a frame is
# 46 ms) and far less than the guessing a header-less stream gets wrong by.
DECODED_DURATION_TOLERANCE_S = 0.5

# Bracketed by two live GPU runs of one request shape (a 19.4 s clip plus a
# 1500-character Devanagari script): with 151 frames left over the model emitted
# exactly one frame (0.0 s of audio, finished=true, RTF 60); with 623 left over it
# emitted 3.2 s of clean speech.  256 frames (~12 s of audio, a quarter of the
# default budget) sits between the degenerate answer and the shortest take that
# still sounded like speech, and stays far below what the vendor's own guidance
# endorses (each input under 150 characters -- a request that reaches this floor
# has already spent the context on itself).  A client that lands here cannot see
# the arithmetic, so the 400 spells it out rather than answering with a sneeze.
MIN_GENERATION_FRAMES = 256

# Worst case for the text half of the budget is one token per character (CJK,
# Devanagari); 1500 characters + 20 s of clip + prompt scaffolding stays inside
# MAX_CONTEXT_POSITIONS -- measured at 1897 for 1500 Devanagari characters plus a
# 19.4 s clip.  That is the cap's whole job: keep the schema from promising a
# prompt the context cannot hold.  It bounds *characters*, not tokens per
# character, so MIN_GENERATION_FRAMES is what catches the scripts that spend the
# context on themselves anyway.  The engine's guidance is far shorter -- keep each
# input under 150 characters for best prosody -- so this cap is a context guard,
# not a quality claim.
MAX_TEXT_CHARACTERS = 1500

# Sanity valve for the request payload (~85 s of 44.1 kHz audio).  Not a model
# limitation; MAX_PROMPT_DURATION_S does the real work.
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
        max_length=MAX_TEXT_CHARACTERS,
        description=(
            "Text to synthesize, e.g. 'Hello there'.  The model's 2048 "
            "position context is shared with the reference clip, hence the "
            "hard cap.  For best prosody keep it under ~150 characters and "
            "split longer passages into separate requests."
        ),
    )
    audio_base64: str | None = Field(
        None,
        min_length=1,
        max_length=MAX_AUDIO_B64_LEN,
        description=(
            "Reference voice clip (3--10 s works well) as a base64 string, "
            "for zero-shot cloning.  Must be sent together with "
            "`reference_text` -- the engine refuses a clip without its exact "
            "transcript.  Omit both to use the model's own default voice.  "
            "Any container soundfile can decode (WAV, MP3, OGG, FLAC, ...); "
            "any rate (resampled to 44.1 kHz server-side).  An MP3 must carry "
            "a length table (Xing/Info/VBRI): without one a decoder can only "
            "guess the duration and stops at the guess, so such a clip is "
            "refused (400) instead of being used in pieces."
        ),
    )
    reference_text: str | None = Field(
        None,
        description=(
            "Exact transcript of the reference clip.  Required whenever "
            "`audio_base64` is present, and rejected without it -- the "
            "pairing is a client contract, not a mode switch."
        ),
    )
    language: str | None = Field(
        DEFAULT_LANGUAGE,
        description=(
            "Two-letter language code, e.g. 'en' or 'th'.  Accepted for API "
            "consistency but not forwarded -- the engine has no language "
            "parameter and detects the language from the input text "
            "(Realistic Air covers 23 languages).  "
            "Omitted or empty defaults to 'en'."
        ),
    )
    seed: int | None = Field(
        None,
        ge=SEED_MIN,
        le=SEED_MAX,
        description=(
            "Random seed for reproducibility.  If omitted, a random seed "
            f"in [{SEED_MIN}, {SEED_MAX}] is chosen and echoed in the response."
        ),
    )

    # --- engine-specific tuning (defaults mirror the engine CLI's own) -----
    max_new_tokens: int = Field(
        1024,
        ge=1,
        le=MAX_NEW_TOKENS_LIMIT,
        description=(
            "Generation budget in audio frames (one frame = "
            f"{CODEC_FRAME_SIZE} codec samples, ~{FRAMES_PER_SECOND:.1f} per "
            "second of audio), so 1024 is ~47 s.  The model clamps this to the "
            f"{MAX_CONTEXT_POSITIONS}-position context minus the prompt, so a "
            "long text/clip shortens it.  A prompt that leaves under "
            f"{MIN_GENERATION_FRAMES} frames over is refused (400) rather "
            "than answered with a single frame of audio."
        ),
    )
    retry_max_new_tokens: int = Field(
        2000,
        ge=1,
        le=MAX_NEW_TOKENS_LIMIT,
        description=(
            "If the first pass hits its frame budget without emitting an end "
            "of speech token, retry once with this larger budget (the engine "
            "CLI's own default).  Setting it equal to `max_new_tokens` "
            "disables the retry; anything smaller is rejected (422).  Both "
            f"budgets are clamped to what the prompt leaves over out of "
            f"{MAX_CONTEXT_POSITIONS} positions, so a wide prompt disables the "
            "retry on its own -- a second pass could not add a frame."
        ),
    )
    temperature: float = Field(
        0.8,
        gt=0.0,
        le=2.0,
        description=(
            "Sampling temperature (engine CLI default 0.8; the checkpoint's "
            "generation_config suggests 0.7).  The engine only enforces that "
            "it is positive."
        ),
    )
    top_p: float = Field(
        0.95,
        gt=0.0,
        le=1.0,
        description="Nucleus (top-p) sampling threshold (engine CLI default).",
    )
    top_k: int = Field(
        50,
        ge=1,
        le=4096,
        description=(
            "Top-k candidate count (engine CLI default).  Must be positive -- "
            "the engine's own filter rejects 0 -- and values above the 4096 "
            "entry semantic codebook are a no-op, so the schema stops there."
        ),
    )
    greedy: bool = Field(
        False,
        description=(
            "Argmax decoding instead of sampling.  Deterministic regardless "
            "of seed, and noticeably flatter in prosody."
        ),
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
        # docs/02 no-support case (VoxCPM/LuxTTS precedent): the engine takes
        # no language parameter, so there is nothing to forward and no 'auto'
        # sentinel -- accept any well-formed code, reject garbage at the
        # boundary rather than letting a client believe it steered anything.
        return validate_language_code(v)

    @model_validator(mode="after")
    def _reference_clip_and_transcript_pair(self) -> "SynthesisRequest":
        # The engine's processor refuses a clip without a transcript, and
        # silently ignores a transcript without a clip.  Both are client bugs;
        # say so at the boundary (422) instead of forwarding half a pair to a
        # ValueError that arrives as a 500.
        if (self.audio_base64 is None) != (self.reference_text is None):
            raise ValueError(
                "audio_base64 and reference_text must be provided together "
                "(reference_text is the exact transcript of the reference clip)"
            )
        if self.reference_text is not None and not self.reference_text.strip():
            raise ValueError("reference_text must contain non-whitespace characters")
        if self.retry_max_new_tokens < self.max_new_tokens:
            raise ValueError(
                "retry_max_new_tokens must be >= max_new_tokens "
                "(set them equal to disable the retry)"
            )
        return self


class SynthesisResponse(CoreSynthesisResponse):
    """The synthesis result (core fields from tts_engine_common, plus extras)."""

    fid: str = Field(..., description="Request ID (internal).")
    finished: bool = Field(
        ...,
        description=(
            "Whether the model emitted its end-of-speech token.  False means "
            "it ran out of frame budget instead: the output may be truncated. "
            "Raise `max_new_tokens` (or shorten the text) and retry."
        ),
    )


class HealthResponse(CoreHealthResponse):
    """The /health core from tts_engine_common, with this engine's serverType.

    model/device pick up the import-time config; version carries the shared
    RELEASE_VERSION constant — overriding it here is how a server drifts.
    """

    serverType: Literal["Audio8 TTS"] = "Audio8 TTS"
    model: str = MODEL_NAME_OR_PATH
    device: str = DEVICE


# ---------------------------------------------------------------------------
# Capabilities (derived from SynthesisRequest -- single source of truth)
# ---------------------------------------------------------------------------

CAPABILITIES = build_capabilities(
    SynthesisRequest,
    engine="audio8-tts",
    model=MODEL_NAME_OR_PATH,
    device=DEVICE,
    sample_rate=SAMPLE_RATE,
    watermarked=False,
    endpoint="/synthesize",
    reference_audio={
        "required": False,
        "formats": ["wav", "mp3", "ogg", "flac"],
        "min_duration_s": MIN_PROMPT_DURATION_S,
        "max_duration_s": MAX_PROMPT_DURATION_S,
        "note": (
            "Optional: omit it (and `reference_text`) to use the model's own "
            "default voice -- there is no preset or voice-design mode.  For "
            "cloning, `reference_text` (the clip's exact transcript) is "
            "mandatory: the processor refuses a clip without it, so half a "
            "pair is a 422.  3--10 s of clean, single-speaker audio works "
            "best.  Nothing is truncated: each second of clip occupies "
            f"~{FRAMES_PER_SECOND:.1f} of the model's "
            f"{MAX_CONTEXT_POSITIONS} context positions, which is why clips "
            "over 20 s are refused outright -- and why a clip whose container "
            "length cannot be measured (an MP3 with no Xing/Info/VBRI table) "
            "is refused too, rather than decoded down to whatever a bitrate "
            "guess happens to land on."
        ),
    },
    languages=None,  # no fixed list; two-letter codes (docs/02), not forwarded
    overrides={
        "max_new_tokens": {"step": 32, "advanced": True},
        "retry_max_new_tokens": {"step": 32, "advanced": True},
        "temperature": {"step": 0.05},
        "top_p": {"step": 0.01, "advanced": True},
        "top_k": {"step": 1, "advanced": True},
        "greedy": {"advanced": True},
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
        del _runtime.processor
        _runtime = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded and CUDA cache cleared.")


app = FastAPI(
    title="Audio8 TTS API",
    description=(
        "REST API around Audio8-TTS-Realistic-Air (0.6B DualAR, 44.1 kHz).  "
        "Send text plus an optional reference clip and its exact transcript "
        "and get back cloned speech.  The model speaks 23 languages and "
        "detects which one from the text.  Synthesis requests are serialized "
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
# Runtime -- thin wrapper around the Audio8 (arktts) model
# ---------------------------------------------------------------------------


@dataclass
class Audio8Runtime:
    """Holds the loaded model/processor and their metadata for the server's life."""

    model: Any
    processor: Any
    sample_rate: int
    max_context_positions: int
    device: str


_runtime: Audio8Runtime | None = None

# generate() rebuilds every layer's static KV cache in place on each call
# (ArkttsModel._setup_generation_caches), decode_audio lazily attaches the
# shared codec, and both run under @torch.inference_mode(), which is
# thread-affine.  Serialize synthesis; single-GPU throughput is the bottleneck
# anyway.
_synthesis_lock = threading.Lock()


def _check_device_available(spec: DeviceSpec | None = None) -> None:
    """Raise unless this box can serve ``spec`` (default: the configured one).

    Runs before any checkpoint access: torch accepts ``cuda:99`` happily and
    ``is_available()`` only says *some* GPU exists, so without this the first
    complaint is an AcceleratorError at ``.to(device)`` -- i.e. after the
    checkpoint has already been downloaded and loaded, which is the exact
    failure this server promises to catch up front.  The wording is shared
    across servers (docs/01 D8); the probes stay here because the shared
    package must not import torch.
    """
    require_cuda_available(
        spec or DEVICE_SPEC,
        is_available=torch.cuda.is_available,
        device_count=torch.cuda.device_count,
        env_var=DEVICE_ENV,
    )


def _resolve_device(value: str) -> torch.device:
    """Map the configured device onto a torch device (the engine CLI's rules)."""
    spec = parse_device(value, env_var=DEVICE_ENV, families=DEVICE_FAMILIES)
    if spec.family == "auto":
        # 'auto' is a request to *choose*, not a request for CUDA, so it must
        # reach CPU when there is no GPU -- exactly the engine CLI's resolve_device.
        # Treating it as plain 'cuda' here would crash a CPU-only box at startup
        # while AUDIO8_TTS_DTYPE=auto was still promising float32 on CPU.
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = spec.requested

    device = torch.device(resolved)
    # The guard needs the *resolved* shape: 'auto' has no ordinal to check and
    # must not be reported as a CUDA request, while 'cuda:9' must be.
    _check_device_available(DeviceSpec(device.type, device.index, spec.requested))
    return device


def _resolve_dtype(value: str, device: torch.device) -> Any:
    """Map the configured dtype onto a torch dtype (the engine CLI's rules)."""
    if value == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cpu" and value != "float32":
        raise RuntimeError(
            f"AUDIO8_TTS_DTYPE={value} cannot be used on CPU: CPU inference "
            "requires float32"
        )
    return getattr(torch, value)


def _get_runtime() -> Audio8Runtime:
    """Return the global runtime, loading model + processor once on first call."""
    global _runtime
    if _runtime is None:
        logger.info(
            "Loading Audio8 checkpoint '{}' on device '{}' (dtype '{}'; first "
            "run downloads from HuggingFace) ...",
            MODEL_NAME_OR_PATH,
            DEVICE,
            DTYPE,
        )
        device = _resolve_device(DEVICE)
        dtype = _resolve_dtype(DTYPE, device)
        # trust_remote_code is not optional: the checkpoint's auto_map points at
        # its own ArkttsProcessor/ArkttsModel (the engine publishes no package).
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME_OR_PATH, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            MODEL_NAME_OR_PATH, trust_remote_code=True, dtype=dtype
        ).eval().to(device)
        # Read the rates/budgets off the loaded checkpoint rather than trusting
        # the module constants, so a future checkpoint self-corrects in the
        # response (the capabilities document is derived at import time).
        _runtime = Audio8Runtime(
            model=model,
            processor=processor,
            sample_rate=int(model.config.codec_sample_rate),
            max_context_positions=int(model.config.max_seq_len),
            device=str(device),
        )
        logger.info(
            "Model loaded successfully. Sampling rate: {} Hz, context: {} "
            "positions, device: {}, dtype: {}",
            _runtime.sample_rate,
            _runtime.max_context_positions,
            _runtime.device,
            dtype,
        )
    return _runtime


# ---------------------------------------------------------------------------
# Global exception handler -- catches anything that slips past endpoint handlers
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
    <head><title>Audio8 TTS API</title></head>
    <body>
        <h1>Audio8 TTS Voice Cloning REST API</h1>
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
    summary="Synthesize speech (zero-shot clone, or the model's default voice)",
)
def synthesize(req: SynthesisRequest) -> SynthesisResponse:
    """
    Synthesize audio from text, optionally cloning a reference voice.

    The full parameter list is documented at GET /capabilities; the request
    schema mirrors it exactly (same model, no drift).
    """
    runtime = _get_runtime()

    # Resolve randomised parameters.
    seed = req.seed if req.seed is not None else random.randint(SEED_MIN, SEED_MAX)

    logger.info(
        "Synthesizing: seed={}, text_len={}, mode={}, temperature={:.2f}, "
        "top_p={:.2f}, top_k={}, greedy={}, requested budget={} (retry {}); "
        "the effective ceiling depends on the prompt and is logged below, "
        "lang={} (not forwarded)",
        seed,
        len(req.text),
        "cloning" if req.audio_base64 else "default-voice",
        req.temperature,
        req.top_p,
        req.top_k,
        req.greedy,
        req.max_new_tokens,
        req.retry_max_new_tokens,
        req.language,
    )

    # Decode and sanity-check the reference audio before touching the model.
    reference: tuple[np.ndarray, int] | None = None
    if req.audio_base64 is not None:
        try:
            raw_audio = decode_base64(req.audio_base64)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")
        advertised_duration = _check_reference_audio(raw_audio)
        reference = _decode_reference_audio(raw_audio, advertised_duration)

    try:
        # Time the actual synthesis call.
        t0 = time.perf_counter()

        with _synthesis_lock:
            seed_everything(seed)
            inputs = _prepare_inputs(
                runtime, req.text, req.reference_text, reference
            )

            # The engine raises ValueError ("Prompt length N must be smaller
            # than 2048") once the prompt outgrows the context -- a client
            # mistake that would otherwise surface as a 500.  Measure it here,
            # exactly, from the tokenized prompt the engine is about to see.
            prompt = _prompt_position_count(inputs)
            free_positions = runtime.max_context_positions - prompt.total
            budget, retry_budget = _frame_budgets(req, free_positions)

            if free_positions <= 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Prompt needs {prompt.total} of the model's "
                        f"{runtime.max_context_positions} text+audio positions "
                        f"({prompt.describe()}). Nothing is left to generate."
                    ),
                )
            if free_positions < MIN_GENERATION_FRAMES:
                # The prompt fits, but it fits so tightly that the "answer" can
                # only be a fraction of a second.  The engine accepts this and
                # returns it as a finished take: measured on a real GPU, a 19.4 s
                # clip plus a 1500-character Devanagari script left 151 frames
                # and produced exactly one frame of audio (finished=true, RTF 60).
                # Refuse with the numbers the client cannot compute itself.
                #
                # This is a bracket, not a cure: the same script with 651 frames
                # left over still collapsed to 0.05 s on its own (a model-side
                # bail no server-side arithmetic can predict), which is why the
                # docs push the vendor's 150-character guidance rather than
                # promising that a 200 means a usable take.
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Prompt needs {prompt.total} of the model's "
                        f"{runtime.max_context_positions} text+audio positions "
                        f"({prompt.describe()}), leaving {free_positions} "
                        f"frames = {free_positions / FRAMES_PER_SECOND:.1f} s "
                        f"of audio to speak {len(req.text)} characters. Shorten "
                        "the text or the reference clip."
                    ),
                )
            if budget < req.max_new_tokens:
                logger.info(
                    "Prompt leaves {} positions over, so the requested {}-frame "
                    "budget is clamped to {}; the retry ceiling is {}.",
                    free_positions,
                    req.max_new_tokens,
                    budget,
                    retry_budget,
                )

            # A dedicated generator (not the global RNG) is what the engine CLI
            # uses; it also lets the retry below continue the same RNG stream
            # instead of replaying the first pass.
            generator = torch.Generator(device=torch.device(runtime.device))
            generator.manual_seed(seed)

            result = _generate(runtime, inputs, generator, req, budget)
            if not bool(result.finished[0]) and retry_budget > budget:
                # The first pass filled its frame budget without ever emitting
                # end-of-speech, so the audio is almost certainly cut off.  One
                # retry with a larger budget, exactly as the engine CLI does --
                # the retry's output replaces the truncated first pass.
                logger.warning(
                    "Generation hit its {} frame budget without an end-of-speech "
                    "token; retrying with {}.",
                    budget,
                    retry_budget,
                )
                result = _generate(runtime, inputs, generator, req, retry_budget)
            elif not bool(result.finished[0]):
                # The generation never said it was done, but the engine would
                # clamp a larger request back to the ceiling we just hit, so a
                # second pass cannot add a frame -- it would only re-roll the
                # same length and double the client's wait.
                logger.info(
                    "Generation ended without an end-of-speech token but the "
                    "{}-position prompt leaves only {} frames: retry skipped "
                    "(both budgets clamp to {}). Raise max_new_tokens only by "
                    "shortening the text or the clip.",
                    prompt.total,
                    free_positions,
                    budget,
                )

            waveform = _first_waveform(runtime, result)

        time_used = time.perf_counter() - t0

        audio_array = waveform
        sample_rate = runtime.sample_rate

        rtf = compute_rtf(time_used, len(audio_array), sample_rate)

        # Encode the output WAV to base64.
        audio_bytes = _numpy_to_wav_bytes(audio_array, sample_rate)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        audio_duration = len(audio_array) / sample_rate if sample_rate else 0.0
        if audio_duration == 0.0:
            logger.warning(
                "Synthesis produced no audio frames (finished={}); returning "
                "silence.",
                bool(result.finished[0]),
            )
        logger.info(
            "Synthesis complete: {:.1f} s wall-clock, {:.1f} s audio, "
            "RTF={}, frames={}, finished={}",
            time_used,
            audio_duration,
            f"{rtf:.3f}" if rtf is not None else "n/a",
            int(result.code_lengths[0]),
            bool(result.finished[0]),
        )

        return SynthesisResponse(
            audio_base64=audio_b64,
            sample_rate=sample_rate,
            seed=seed,
            fid=str(uuid.uuid4()),
            time_used=time_used,
            rtf=rtf,
            finished=bool(result.finished[0]),
        )

    # Pre-flight rejections (bad prompt budget) must survive the catch-all.
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Synthesis failed: {}", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Set the random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# MPEG length tables live in the first frame (Xing/Info) or the first frame of
# the first Gop (VBRI); 64 KiB past any ID3v2 tag covers every encoder in the
# wild.  A stray match in compressed data only restores the old (wrong) behavior
# of trusting a guessed length, so the scan is deliberately loose.
_MPEG_LENGTH_TAGS = (b"Xing", b"Info", b"VBRI")
_MPEG_TAG_SCAN_BYTES = 65536


def _id3v2_tag_end(raw_bytes: bytes) -> int:
    """Where an ID3v2 tag ends (0 if there is none) so the scan skips its cover art."""
    if raw_bytes[:3] != b"ID3" or len(raw_bytes) < 10:
        return 0
    size = (
        ((raw_bytes[6] & 0x7F) << 21)
        | ((raw_bytes[7] & 0x7F) << 14)
        | ((raw_bytes[8] & 0x7F) << 7)
        | (raw_bytes[9] & 0x7F)
    )
    return min(10 + size, len(raw_bytes))


def _looks_like_mpeg_audio(raw_bytes: bytes) -> bool:
    """ID3-tagged or a bare MPEG frame sync (11 set bits)."""
    if raw_bytes[:3] == b"ID3":
        return True
    return len(raw_bytes) >= 2 and raw_bytes[0] == 0xFF and (raw_bytes[1] & 0xE0) == 0xE0


def _reject_unmeasurable_mpeg_length(raw_bytes: bytes) -> None:
    """Refuse MPEG/MP3 clips whose length the decoder can only guess.

    libsndfile measures an MPEG stream from its length table (Xing / Info /
    VBRI).  With no table it *estimates* from the first frame's bitrate and then
    stops reading at that estimate -- so a VBR clip without one arrives as a
    fraction of itself.  Measured on this box: a 40.05 s reference came through
    as 6.73 s, sailed through the 2--20 s window (a window enforced against a
    number nobody measured), and the model was conditioned on the first sixth of
    the clip while being told nothing had happened.

    This server truncates nothing.  A container that truncates itself is the
    client's to fix, and the fix costs them one ffmpeg invocation.
    """
    if not _looks_like_mpeg_audio(raw_bytes):
        return
    window = raw_bytes[_id3v2_tag_end(raw_bytes):][: _MPEG_TAG_SCAN_BYTES]
    if any(tag in window for tag in _MPEG_LENGTH_TAGS):
        return
    raise HTTPException(
        status_code=400,
        detail=(
            "MP3 reference has no length table (Xing/Info/VBRI), so its "
            "duration can only be guessed from the first frame's bitrate -- "
            "and the decoder stops at that guess, silently discarding the rest "
            "of the clip. Re-encode the reference as WAV/FLAC/OGG, or re-write "
            "the MP3 with a length table (ffmpeg's libmp3lame does unless "
            "asked not to)."
        ),
    )


def _check_prompt_duration(duration: float, measured: str) -> None:
    """Enforce the clip window on one measured duration."""
    if duration < MIN_PROMPT_DURATION_S:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Reference audio is {duration:.2f} s long ({measured}); at "
                f"least {MIN_PROMPT_DURATION_S:.0f} s is required for usable "
                "voice cloning."
            ),
        )
    if duration > MAX_PROMPT_DURATION_S:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Reference audio is {duration:.2f} s long ({measured}); at "
                f"most {MAX_PROMPT_DURATION_S:.0f} s is accepted.  Nothing is "
                "truncated server-side, and every second of clip consumes "
                f"~{FRAMES_PER_SECOND:.1f} of the model's "
                f"{MAX_CONTEXT_POSITIONS} context positions that the text and "
                "the generated speech need."
            ),
        )


def _check_reference_audio(raw_bytes: bytes) -> float:
    """Cheap header pass: reject unmeasurable, undecodable, or out-of-window clips.

    Returns the duration the container *advertises*.  That number is an
    advertisement, not a measurement -- the decode re-checks it against the
    samples that actually exist (see _check_decoded_duration).
    """
    _reject_unmeasurable_mpeg_length(raw_bytes)
    try:
        info = sf.info(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not decode reference audio: {exc}"
        )
    advertised = info.frames / info.samplerate if info.samplerate else 0.0
    _check_prompt_duration(advertised, measured="per its container header")
    return advertised


def _check_decoded_duration(decoded: float, advertised: float) -> None:
    """Gate again on what the stream really decoded to.

    A container header is a claim: truncated streams, padded encodes and
    byte-exact copies can all advertise a duration the audio does not have.
    The 2--20 s window exists because of how many context positions a clip
    costs, so it is enforced on the duration that reaches the model first;
    the header disagreement is reported after it, because "your container and
    your audio disagree" is the diagnosis but "this clip is too long for the
    model's context" is the thing the client can act on.
    """
    _check_prompt_duration(decoded, measured="as decoded")
    if abs(decoded - advertised) > DECODED_DURATION_TOLERANCE_S:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Reference audio decoded to {decoded:.2f} s, but its container "
                f"header advertises {advertised:.2f} s: the container's length "
                "metadata does not match its audio stream, so the clip the "
                "model would be conditioned on is not the clip that was sent. "
                "Re-encode the reference as WAV."
            ),
        )


def _read_reference_samples(raw_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode to mono float32 samples plus their source rate (needs real NumPy).

    Mono mixdown happens here rather than inside the processor so what we hand
    over is unambiguous.
    """
    try:
        samples, source_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not decode reference audio: {exc}"
        )
    if samples.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Reference audio contains no samples")
    mono = samples.mean(axis=1)
    if not np.isfinite(mono).all():
        raise HTTPException(
            status_code=400, detail="Reference audio contains NaN or infinite samples"
        )
    return mono, int(source_rate)


def _decode_reference_audio(
    raw_bytes: bytes, advertised_duration_s: float
) -> tuple[np.ndarray, int]:
    """Decode the clip, then enforce the window on the samples that exist.

    Unlike most tts-serve engines, this one does not want a *file path*: the
    processor takes (samples, rate) directly, so there is no staging directory,
    no content-hashing, and nothing to clean up.
    """
    samples, source_rate = _read_reference_samples(raw_bytes)
    decoded = samples.shape[0] / source_rate if source_rate else 0.0
    _check_decoded_duration(decoded, advertised_duration_s)
    return samples, source_rate


def _prepare_inputs(
    runtime: Audio8Runtime,
    text: str,
    reference_text: str | None,
    reference: tuple[np.ndarray, int] | None,
) -> dict[str, Any]:
    """Tokenize the prompt (and shape the reference clip) into model inputs.

    The processor returns prefix/suffix token rows plus the clip's sample
    count; the model encodes the clip itself (ArkttsModel._prepare_prompt ->
    encode_audio), which is why nothing here touches the codec.
    """
    processor_kwargs: dict[str, Any] = {"text": text, "return_tensors": "pt"}
    if reference is not None:
        samples, source_rate = reference
        # Batch of one: the processor takes per-item lists for these fields,
        # and reference_audio is accepted as an in-memory (samples, rate) pair.
        processor_kwargs["reference_audio"] = [(samples, source_rate)]
        processor_kwargs["reference_text"] = [reference_text]
    inputs = runtime.processor(**processor_kwargs)
    device = torch.device(runtime.device)
    return {name: value.to(device) for name, value in inputs.items()}


class PromptPositions(NamedTuple):
    """The three ways a prompt spends the model's packed context positions."""

    tokens: int
    clip_frames: int

    @property
    def total(self) -> int:
        return self.tokens + self.clip_frames

    def describe(self) -> str:
        """A client-readable breakdown, so the 400 can say *what* to shorten."""
        if not self.clip_frames:
            return f"{self.tokens} text tokens, no reference clip"
        return (
            f"{self.tokens} text tokens + {self.clip_frames} reference-clip "
            f"frames ({self.clip_frames / FRAMES_PER_SECOND:.1f} s of clip at "
            f"{FRAMES_PER_SECOND:.1f} frames/second)"
        )


def _prompt_position_count(inputs: dict[str, Any]) -> PromptPositions:
    """How many of the model's packed positions the prompt occupies.

    Mirrors what ArkttsModel._prepare_prompt builds: unpadded prefix tokens,
    unpadded suffix tokens, and one frame per CODEC_FRAME_SIZE samples of
    reference clip.  Reads the processor's own masks/lengths rather than
    re-deriving them, so it cannot drift from the engine's arithmetic.
    """
    tokens = int(inputs["prefix_attention_mask"].sum()) + int(
        inputs["suffix_attention_mask"].sum()
    )
    clip_frames = 0
    clip_lengths = inputs.get("reference_audio_lengths")
    if clip_lengths is not None:
        samples = int(clip_lengths[0])
        clip_frames = (samples + CODEC_FRAME_SIZE - 1) // CODEC_FRAME_SIZE
    return PromptPositions(tokens=tokens, clip_frames=clip_frames)


def _frame_budgets(req: SynthesisRequest, free_positions: int) -> tuple[int, int]:
    """(first-pass, retry) frame budgets, after the clamp the engine applies.

    ArkttsModel.generate() silently caps max_new_tokens at
    ``max_seq_len - prompt_width`` (modeling_arktts.py).  Comparing the *request*
    fields against each other therefore lies about whether a retry can help: with
    a prompt wider than ``MAX_CONTEXT_POSITIONS - max_new_tokens``, the retry
    asks for more and is handed back the identical ceiling, paying for a second
    full generation pass that cannot produce one extra frame.  Do the same
    arithmetic here so the retry only runs when it can actually lengthen the
    audio.
    """
    return (
        min(req.max_new_tokens, free_positions),
        min(req.retry_max_new_tokens, free_positions),
    )


def _generate(
    runtime: Audio8Runtime,
    inputs: dict[str, Any],
    generator: Any,
    req: SynthesisRequest,
    max_new_tokens: int,
) -> Any:
    """One generate() pass. Sampling arguments mirror the engine CLI's call."""
    return runtime.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        do_sample=not req.greedy,
        generator=generator,
        return_dict_in_generate=True,
    )


def _first_waveform(runtime: Audio8Runtime, result: Any) -> np.ndarray:
    """Decode the generated codec frames to a waveform and trim the padding.

    decode_audio pads its batch; code_lengths is the engine's own count of
    frames that were actually emitted (frames after end-of-speech are -1).
    The waveform lives on the model's device, so it comes back to CPU as
    float32 numpy here -- np.asarray() on a CUDA tensor raises.
    """
    waveforms, waveform_lengths = runtime.model.decode_audio(result.codes)
    trimmed = waveforms[0, : int(waveform_lengths[0])]
    return trimmed.float().cpu().numpy()


def _numpy_to_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    """Convert a numpy audio array to WAV-encoded bytes (PCM_16)."""
    buffer = io.BytesIO()
    # NaN/inf survive np.clip (np.clip(nan) is still nan), and libsndfile turns
    # them into full-scale PCM samples -- measured on this box: nan -> -32768,
    # +inf -> +32767.  A single NaN anywhere in the codec output would otherwise
    # ship as loud clicks the client blames on the model.
    #
    # The clamp stays for the same reason: libsndfile 0.14 happens to saturate
    # out-of-range floats rather than wrap them (measured: 1.5 -> 32767), but
    # that is a property of whichever libsndfile is installed, not of this
    # server.  Don't make the client's audio depend on it.
    sanitized = np.nan_to_num(audio_array, nan=0.0, posinf=1.0, neginf=-1.0)
    sf.write(
        buffer,
        np.clip(sanitized, -1.0, 1.0),
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Main (for running directly: python server_audio8TTS.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("AUDIO8_TTS_HOST", "0.0.0.0")
    port = int(os.getenv("AUDIO8_TTS_PORT", "7500"))
    logger.info("Starting Audio8 TTS REST API server on {}:{}", host, port)
    # Pass the app object directly instead of a module path string,
    # so this works regardless of how the file is invoked.
    uvicorn.run(app, host=host, port=port, log_level="info")
