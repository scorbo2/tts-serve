"""
FastAPI REST server for faster-qwen3-tts voice cloning.

faster-qwen3-tts is a CUDA-graphs speed fork of Qwen3-TTS (several times
faster inference on NVIDIA GPUs).  This server exposes the engine's
*advanced* (ICL — in-context learning) cloning mode only: the full reference
audio is kept in the model's context, so the exact transcript of that audio
is **required** in the request.  The engine's "simple" x-vector-only mode
(no transcript, reduced quality) is deliberately not exposed.

Loads the model once on startup, then exposes a single POST endpoint for
synthesis.  Clients send text, a reference audio sample (base64), and the
sample's transcript; the server returns the generated audio as
base64-encoded 24 kHz WAV.

Capabilities: GET /capabilities returns a machine-readable description of
every request parameter, derived from the Pydantic request model so it can
never drift from what the server actually validates (see the
tts-engine-common README).

The engine requires an NVIDIA GPU: its CUDA-graph backend rejects non-CUDA
devices at load time, and PyTorch >= 2.5.1 is required for reliable graph
capture (older builds fail capture with "operation not permitted when
stream is capturing").

Model weights are downloaded from HuggingFace on first start unless a local
path is given.  The language list below is the Base checkpoint's (10
languages + auto); the engine itself validates against its own config.

Note: ``pip install faster-qwen3-tts`` pulls ``qwen-tts-hf``, a temporary
compatibility build of qwen-tts with Transformers 5 support.  It provides
the same ``qwen_tts`` Python package as upstream; do not install both
distributions in the same environment.

Configuration (environment variables):
    FASTER_QWEN3TTS_MODEL     HuggingFace id or local path.
                              Default: Qwen/Qwen3-TTS-12Hz-1.7B-Base
    FASTER_QWEN3TTS_DEVICE    CUDA device to load the model on: 'cuda' or
                              'cuda:N' for a specific GPU.  Must be a CUDA
                              device (the engine has no CPU/MPS backend).
                              Default: cuda
    FASTER_QWEN3TTS_HOST      Bind host for `python server_fasterQwen3TTS.py`.
                              Default: 0.0.0.0
    FASTER_QWEN3TTS_PORT      Bind port for `python server_fasterQwen3TTS.py`.
                              Default: 8000

Extra dependencies beyond the faster-qwen3-tts package:
    pip install fastapi uvicorn loguru soundfile
    pip install ../tts-engine-common # in-repo copy; or: pip install -e ../tts-engine-common

Usage:
    python server_fasterQwen3TTS.py
    # or: uvicorn server_fasterQwen3TTS:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import random
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator
from faster_qwen3_tts import FasterQwen3TTS
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

MODEL_NAME_OR_PATH = os.getenv("FASTER_QWEN3TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
DEVICE = os.getenv("FASTER_QWEN3TTS_DEVICE", "cuda")

# The 12 Hz codec is tokens/s, not the waveform rate: generated audio is
# 24 kHz (the engine infers it from the model and echoes it back in every
# result, which is what the response uses).
SAMPLE_RATE = 24000

SEED_MIN = 1
SEED_MAX = 1000

# Heuristic lower bound: below this the speaker embedding / reference codes
# degrade to near-garbage (the README's "3-second rapid voice clone" is the
# sweet spot).
MIN_REF_DURATION_S = 2.0

# Sanity valve for the request payload (~5 min of 24 kHz audio).
MAX_AUDIO_B64_LEN = 10_000_000

# Base checkpoint language support (the engine lower-cases and validates
# against its own config at generate time, so this table is a UI/validation
# convenience, not the final arbiter).
#
# The API contract is two-letter codes (docs/02-language-handling.md); the
# engine wants lowercase *names*, so LANGUAGE_CODE_TO_NAME is this server's
# internal mapping table and everything else is derived from it.
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


def _validate_config() -> None:
    """Fail fast on bad configuration instead of partway through a model download."""
    # The CUDA-graph backend hard-rejects non-CUDA devices at load time; the
    # prefix (rather than an exact 'cuda') also allows multi-GPU selection
    # like 'cuda:1'.
    if not DEVICE.startswith("cuda"):
        raise ValueError(
            f"FASTER_QWEN3TTS_DEVICE must be a CUDA device ('cuda' or "
            f"'cuda:N'); faster-qwen3-tts has no CPU/MPS backend, got {DEVICE!r}"
        )


_validate_config()

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

# Dynamic Literal over the Base checkpoint's language codes (+ the engine's
# 'auto' auto-detection sentinel) so the request schema and the
# /capabilities enum share one source.
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
            "runs ICL (advanced) mode, which conditions on the reference "
            "audio and its transcript together."
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
            "Random seed for reproducibility.  If omitted, a random seed "
            f"in [{SEED_MIN}, {SEED_MAX}] is chosen and echoed in the response."
        ),
    )

    # --- engine-specific tuning (None = engine default) ---------------------
    temperature: float | None = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.  Omit for the engine default.",
    )
    top_p: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold.  Omit for the engine default.",
    )
    repetition_penalty: float | None = Field(
        None,
        ge=1.0,
        le=2.0,
        description="Penalty applied to repeated tokens.  Omit for the engine default.",
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
        # ICL mode feeds the transcript straight into the context; a blank
        # one would clone with an empty prompt and the engine would
        # mis-condition, so reject it loudly.
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


class HealthResponse(BaseModel):
    """Health / readiness check."""

    status: Literal["ok"] = "ok"
    serverType: Literal["faster-qwen3-tts"] = "faster-qwen3-tts"
    model: str = MODEL_NAME_OR_PATH
    device: str = DEVICE


# ---------------------------------------------------------------------------
# Capabilities (derived from SynthesisRequest — single source of truth)
# ---------------------------------------------------------------------------

CAPABILITIES = build_capabilities(
    SynthesisRequest,
    engine="faster-qwen3-tts",
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
            "runs ICL (advanced) mode, which conditions on the reference "
            "audio and transcript together.  The engine appends 0.5 s of "
            "silence to the reference before encoding so its final phoneme "
            "does not bleed into the start of the output."
        ),
    },
    languages=sorted(LANGUAGE_CODES),
    overrides={
        "temperature": {"step": 0.05},
        "top_p": {"step": 0.01, "advanced": True},
        "repetition_penalty": {"step": 0.05, "advanced": True},
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
    title="Faster Qwen3-TTS Voice Cloning API",
    description=(
        "REST API around faster-qwen3-tts (CUDA-graphs Qwen3-TTS fork).  "
        "Send text + a reference audio sample + its transcript and get back "
        "cloned speech in ICL (advanced) mode.  Machine-readable parameter "
        "metadata at GET /capabilities."
    ),
    version="0.2.0",
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
# Runtime — thin wrapper around the faster-qwen3-tts model
# ---------------------------------------------------------------------------


@dataclass
class FasterQwen3TTSRuntime:
    """Holds the loaded model and its metadata for the lifetime of the server."""

    model: FasterQwen3TTS
    device: str


_runtime: FasterQwen3TTSRuntime | None = None

# The wrapper replays captured CUDA graphs (static KV buffers are not
# re-entrant) and keeps a shared voice-prompt cache, so concurrent
# generate_voice_clone() calls would stomp on each other's graph state.
# Serialize synthesis; single-GPU throughput is the bottleneck anyway.
_synthesis_lock = threading.Lock()


def _get_runtime() -> FasterQwen3TTSRuntime:
    """Return the global runtime, loading the model once on first call.

    The model is loaded in bfloat16 with SDPA attention (the engine's own
    defaults) and the CUDA graphs are captured up front, so the first
    request doesn't pay the capture cost.
    """
    global _runtime
    if _runtime is None:
        logger.info(
            "Loading faster-qwen3-tts model '%s' on device '%s' ...",
            MODEL_NAME_OR_PATH,
            DEVICE,
        )
        model = FasterQwen3TTS.from_pretrained(
            MODEL_NAME_OR_PATH,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        model.warmup()
        _runtime = FasterQwen3TTSRuntime(model=model, device=DEVICE)
        logger.info("Model loaded successfully.")
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
    <head><title>Faster Qwen3-TTS REST API</title></head>
    <body>
        <h1>Faster Qwen3-TTS Voice Cloning REST API</h1>
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

    faster-qwen3-tts performs zero-shot voice cloning in ICL (advanced) mode:
    a short reference audio clip and its exact transcript are kept in the
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
        "Synthesizing: seed={}, text_len={}, ref_text_len={}, lang={} (engine: {})",
        seed,
        len(req.text),
        len(req.reference_text),
        req.language,
        engine_language,
    )

    # Decode the reference audio in memory; the engine reads it back from a
    # temp file (see _write_temp_audio).
    try:
        raw_audio = decode_base64(req.audio_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")

    _check_reference_audio(raw_audio)

    prompt_audio_path = _write_temp_audio(raw_audio)

    # Only forward sampling params the client actually set, so the engine's
    # own defaults apply otherwise.
    sampling = {
        "temperature": req.temperature,
        "top_p": req.top_p,
        "repetition_penalty": req.repetition_penalty,
    }
    sampling = {k: v for k, v in sampling.items() if v is not None}

    try:
        t0 = time.perf_counter()

        with _synthesis_lock:
            torch.manual_seed(seed)
            wavs, sr = runtime.model.generate_voice_clone(
                text=req.text,
                language=engine_language,  # e.g. 'en' -> 'english'; 'auto' passes through
                ref_audio=prompt_audio_path,
                ref_text=req.reference_text,
                xvec_only=False,  # ICL (advanced) mode is all this server offers
                **sampling,
            )

        time_used = time.perf_counter() - t0

        # wavs[0] is the generated waveform (1-D numpy array, float32).
        wav = wavs[0]

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

    except Exception as exc:
        logger.error("Synthesis failed: {}", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # Clean up the temporary prompt audio file.
        _cleanup_temp(prompt_audio_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMP_AUDIO_DIR = Path(tempfile.gettempdir()) / "faster_qwen3_tts_rest_api"
_TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


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


def _write_temp_audio(raw_bytes: bytes) -> str:
    """
    Write raw audio bytes to a temporary file.

    The engine's ICL path reads the reference from a file path via
    soundfile.  The filename is the SHA-256 of the audio content rather than
    a UUID because the engine caches the encoded voice prompt per
    (path, transcript): content-addressed names let repeat requests with the
    same reference clip hit that cache instead of re-encoding the audio.
    The .wav extension is cosmetic — the loader sniffs the header, so
    MP3/OGG/FLAC bytes work fine.
    """
    digest = hashlib.sha256(raw_bytes).hexdigest()
    path = _TEMP_AUDIO_DIR / f"{digest}.wav"
    path.write_bytes(raw_bytes)
    logger.debug("Wrote temporary reference audio: {}", path)
    return str(path)


def _cleanup_temp(path: str) -> None:
    """Remove a temporary audio file if it exists."""
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


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
# Main (for running directly: python server_fasterQwen3TTS.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("FASTER_QWEN3TTS_HOST", "0.0.0.0")
    port = int(os.getenv("FASTER_QWEN3TTS_PORT", "8000"))
    logger.info("Starting faster-qwen3-tts REST API server on %s:%d", host, port)
    # Pass the app object directly instead of a module path string,
    # so this works regardless of how the file is invoked.
    uvicorn.run(app, host=host, port=port, log_level="info")
