"""
FastAPI REST server for LuxTTS voice cloning (github.com/ysharma3501/LuxTTS).

LuxTTS is a lightweight ZipVoice-distilled voice-cloning model that runs at
>150x realtime on GPU and faster than realtime on CPU.  Clients send text
plus a reference audio sample (base64); the server returns the generated
audio as base64-encoded 48 kHz WAV (the dual-head vocoder always outputs
48 kHz — unlike the 24 kHz of most tts-serve engines).

Unlike most tts-serve engines, LuxTTS has no transcript-of-the-reference
field: the engine *always* transcribes the reference clip with Whisper
(openai/whisper-base on GPU, whisper-tiny on CPU) and conditions on that
transcription plus the audio features.  The `reference_text` core field is
deliberately absent from this server, and every request pays the ASR
inference cost (the ASR model itself loads once at startup).

Likewise there is no language parameter in the engine API: its tokenizer
auto-detects English and Chinese per text segment (other scripts are
dropped).  The `language` field is accepted for API consistency (docs/02)
but is not forwarded to the engine.

`seed` is meaningful: the flow-matching solver draws its initial noise from
the PyTorch RNG, so the same seed and inputs reproduce the same *initial
noise*.  Reproducibility of the final audio, though, is device-dependent:
on CPU the whole graph is deterministic and identical inputs + seed yield
bit-identical output, but on CUDA some kernels are non-deterministic, so
repeated runs can differ by ~1e-7 in sample values.  Do not promise
bit-exact reproducibility for GPU generations.

Model weights are downloaded from HuggingFace (YatharthS/LuxTTS) on first
start.  Set HF_TOKEN in the environment if your checkpoint needs it.

Configuration (environment variables):
    LUX_TTS_MODEL        The model source.  Pass the default HuggingFace id
                         (YatharthS/LuxTTS) to have it downloaded, or a local
                         path to an already-extracted model directory.
                         Default: YatharthS/LuxTTS
                         Note: on the GPU path the engine treats any
                         non-default value as a local path (it only downloads
                         the default id), and on the CPU path the engine
                         ignores this variable entirely and always downloads
                         the default model — that is an upstream LuxTTS bug,
                         not a tts-serve one.
    LUX_TTS_DEVICE       Device to load the model on.  One of: cuda, mps, cpu.
                         Default: cuda.  Note: if 'cuda' is requested but
                         unavailable, the engine silently falls back to MPS,
                         then CPU.
    LUX_TTS_THREADS      CPU ONNX thread count (CPU device only).
                         Default: 4
    LUX_TTS_HOST         Bind host for `python server_luxTTS.py`.
                         Default: 0.0.0.0
    LUX_TTS_PORT         Bind port for `python server_luxTTS.py`.
                         Default: 7500

Extra dependencies beyond the LuxTTS repository:
    # LuxTTS is a git repo, not a PyPI package:
    git clone https://github.com/ysharma3501/LuxTTS.git
    pip install ./LuxTTS
    pip install fastapi uvicorn loguru soundfile
    pip install ../tts-engine-common # in-repo copy; or: pip install -e ../tts-engine-common

Usage:
    python server_luxTTS.py
    # or: uvicorn server_luxTTS:app --host 0.0.0.0 --port 7500
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
from pydantic import BaseModel, ConfigDict, Field, field_validator

from zipvoice.luxvoice import LuxTTS
from tts_engine_common import (
    DEFAULT_LANGUAGE,
    CoreSynthesisResponse,
    build_capabilities,
    capabilities_endpoint,
    cleanup_temp,
    compute_rtf,
    decode_base64,
    normalize_language,
    temp_audio_dir,
    validate_language_code,
    write_temp_audio,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_DEVICES = ("cuda", "mps", "cpu")

MODEL_NAME_OR_PATH = os.getenv("LUX_TTS_MODEL", "YatharthS/LuxTTS")
DEVICE = os.getenv("LUX_TTS_DEVICE", "cuda")
THREADS = int(os.getenv("LUX_TTS_THREADS", "4"))


def _validate_config() -> None:
    """Fail fast on bad configuration instead of partway through a model download."""
    if DEVICE not in SUPPORTED_DEVICES:
        raise ValueError(
            f"LUX_TTS_DEVICE must be one of {SUPPORTED_DEVICES}, got {DEVICE!r}"
        )
    if THREADS < 1:
        raise ValueError(
            f"LUX_TTS_THREADS must be a positive integer, got {THREADS}"
        )


_validate_config()

# The dual-head vocoder (48 kHz head merged with a 24 kHz head upsampled to
# 48 kHz) always outputs 48 kHz audio; return_smooth only selects which head
# provides the low end, never the rate.  Confirmed against the engine's
# own output, not assumed.
SAMPLE_RATE = 48000

SEED_MIN = 1
SEED_MAX = 1000

# The engine README's explicit lower bound for usable voice cloning.
MIN_PROMPT_DURATION_S = 3.0

# Sanity valve for the request payload (~80 s of 48 kHz audio).
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
        description="Text to synthesize, e.g. 'Hello there'.",
    )
    audio_base64: str = Field(
        ...,
        min_length=1,
        max_length=MAX_AUDIO_B64_LEN,
        description=(
            "Reference voice sample (roughly 10 s works well) as a base64 "
            "string.  Any container soundfile can decode (WAV, MP3, OGG, "
            "FLAC, ...).  The engine transcribes it with Whisper for "
            "conditioning (there is no transcript field); only its first "
            "prompt_duration seconds are used."
        ),
    )
    language: str | None = Field(
        DEFAULT_LANGUAGE,
        description=(
            "Two-letter language code, e.g. 'en' or 'zh'.  Accepted for API "
            "consistency but not forwarded — the engine has no language "
            "parameter; its tokenizer auto-detects English and Chinese per "
            "text segment (other scripts are dropped).  "
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

    # --- engine-specific tuning (defaults mirror the model's own defaults) --
    num_steps: int = Field(
        4,
        ge=1,
        le=32,
        description=(
            "Flow-matching sampling steps.  The model is distilled for 4 "
            "(README: 3-4 is best for efficiency); higher is slower with "
            "diminishing quality."
        ),
    )
    guidance_scale: float = Field(
        3.0,
        ge=0.0,
        le=10.0,
        description="Classifier-free guidance scale.",
    )
    t_shift: float = Field(
        0.5,
        gt=0.0,
        le=3.0,
        description=(
            "Flow-matching time shift (README: higher can sound better but "
            "with worse word error rate)."
        ),
    )
    speed: float = Field(
        1.0,
        ge=0.1,
        le=3.0,
        description="Speaking-speed control (1.0 = natural; lower = slower).",
    )
    return_smooth: bool = Field(
        False,
        description=(
            "Use the 24 kHz vocoder head (upsampled to 48 kHz) instead of "
            "the full-band 48 kHz head.  Try this if you hear metallic "
            "artifacts (README)."
        ),
    )
    prompt_duration: float = Field(
        5.0,
        ge=1.0,
        le=1000.0,
        description=(
            "Seconds of the reference clip to use for conditioning "
            "(README: lower can speed up inference; 1000 if you find "
            "artifacts)."
        ),
    )
    prompt_rms: float = Field(
        0.001,
        ge=0.0,
        le=1.0,
        description=(
            "Target RMS for the reference clip (README: higher makes it "
            "sound louder, ~0.01 recommended; engine default 0.001)."
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
        # docs/02: the API speaks two-letter codes.  There is no 'auto'
        # sentinel: the engine always auto-detects and offers no way to
        # influence it, so exposing 'auto' would be a no-op lie.
        return validate_language_code(v)


class SynthesisResponse(CoreSynthesisResponse):
    """The synthesis result (core fields from tts_engine_common, plus fid)."""

    fid: str = Field(..., description="Request ID (internal).")


class HealthResponse(BaseModel):
    """Health / readiness check."""

    status: Literal["ok"] = "ok"
    serverType: Literal["LuxTTS"] = "LuxTTS"
    model: str = MODEL_NAME_OR_PATH
    device: str = DEVICE


# ---------------------------------------------------------------------------
# Capabilities (derived from SynthesisRequest — single source of truth)
# ---------------------------------------------------------------------------

CAPABILITIES = build_capabilities(
    SynthesisRequest,
    engine="lux-tts",
    model=MODEL_NAME_OR_PATH,
    device=DEVICE,
    sample_rate=SAMPLE_RATE,
    watermarked=False,
    endpoint="/synthesize",
    reference_audio={
        "required": True,
        "formats": ["wav", "mp3", "ogg", "flac"],
        "min_duration_s": MIN_PROMPT_DURATION_S,
        "note": (
            "The engine transcribes the reference clip with Whisper "
            "(openai/whisper-base on GPU, whisper-tiny on CPU) — there is "
            "no transcript field, and every request pays the ASR cost.  "
            "Only the first prompt_duration seconds are used; roughly 10 s "
            "of clean speech clones best.  The first request after startup "
            "also pays a one-time librosa initialisation (~10 s)."
        ),
    },
    languages=None,  # no fixed list; two-letter codes (docs/02), not forwarded
    overrides={
        "num_steps": {"step": 1},
        "guidance_scale": {"step": 0.1},
        "t_shift": {"step": 0.05},
        "speed": {"step": 0.05},
        "prompt_duration": {"step": 1},
        "prompt_rms": {"step": 0.001, "advanced": True},
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
    title="LuxTTS Voice Cloning API",
    description=(
        "REST API around LuxTTS.  Send text + a reference audio sample and "
        "get back cloned speech.  Synthesis requests are serialized (single "
        "shared model).  Machine-readable parameter metadata at "
        "GET /capabilities."
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
# Runtime — thin wrapper around the LuxTTS model
# ---------------------------------------------------------------------------


@dataclass
class LuxTTSRuntime:
    """Holds the loaded model and its metadata for the lifetime of the server."""

    model: LuxTTS
    sample_rate: int
    device: str


_runtime: LuxTTSRuntime | None = None

# generate_speech() mutates the shared vocoder's `return_48k` flag in place
# on every call, and the Whisper pipeline plus the model itself carry
# per-call state — concurrent requests would stomp on each other.
# Serialize synthesis; single-device throughput is the bottleneck anyway.
_synthesis_lock = threading.Lock()


def _get_runtime() -> LuxTTSRuntime:
    """Return the global runtime, loading the model once on first call."""
    global _runtime
    if _runtime is None:
        logger.info(
            "Loading LuxTTS model '%s' on device '%s' (first run downloads "
            "from HuggingFace) ...",
            MODEL_NAME_OR_PATH,
            DEVICE,
        )
        model = LuxTTS(MODEL_NAME_OR_PATH, device=DEVICE, threads=THREADS)
        # model.device is the engine's *resolved* device: it silently
        # falls back cuda -> mps -> cpu when the requested device is
        # unavailable, so echo that rather than the configured value.
        _runtime = LuxTTSRuntime(
            model=model,
            sample_rate=SAMPLE_RATE,
            device=str(model.device),
        )
        logger.info(
            "Model loaded successfully. Sampling rate: %d Hz, device: %s",
            _runtime.sample_rate,
            _runtime.device,
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
    <head><title>LuxTTS REST API</title></head>
    <body>
        <h1>LuxTTS Voice Cloning REST API</h1>
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
    summary="Synthesize speech from text + reference audio",
)
def synthesize(req: SynthesisRequest) -> SynthesisResponse:
    """
    Synthesize audio using the provided text and reference audio sample.

    The full parameter list is documented at GET /capabilities; the request
    schema mirrors it exactly (same model, no drift).
    """
    runtime = _get_runtime()

    # Resolve randomised seed.
    seed = req.seed if req.seed is not None else random.randint(SEED_MIN, SEED_MAX)

    logger.info(
        "Synthesizing: seed={}, text_len={}, steps={}, guidance={:.1f}, "
        "t_shift={:.2f}, speed={:.2f}, smooth={}, prompt_dur={:.0f}s, "
        "prompt_rms={:.3f}, lang={} (not forwarded)",
        seed,
        len(req.text),
        req.num_steps,
        req.guidance_scale,
        req.t_shift,
        req.speed,
        req.return_smooth,
        req.prompt_duration,
        req.prompt_rms,
        req.language,
    )

    # Decode and sanity-check the reference audio before touching the model.
    try:
        raw_audio = decode_base64(req.audio_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")

    _check_reference_audio(raw_audio)

    # The engine loads prompt audio from a file path (librosa).  One-shot
    # engine — no path-keyed cache — so a UUID temp file cleaned up per
    # request is fine (no content-hashing needed).
    prompt_audio_path = write_temp_audio(raw_audio, _TEMP_AUDIO_DIR)

    try:
        # Time the actual synthesis call (ASR transcription included — it
        # is part of the engine's per-request cost, not setup).
        t0 = time.perf_counter()

        with _synthesis_lock:
            seed_everything(seed)
            encoded_prompt = runtime.model.encode_prompt(
                prompt_audio_path,
                duration=req.prompt_duration,
                rms=req.prompt_rms,
            )
            # The engine's `language` value is not forwarded: it has no
            # language parameter (docs/02 no-support case).
            wav = runtime.model.generate_speech(
                req.text,
                encoded_prompt,
                num_steps=req.num_steps,
                guidance_scale=req.guidance_scale,
                t_shift=req.t_shift,
                speed=req.speed,
                return_smooth=req.return_smooth,
            )

        time_used = time.perf_counter() - t0

        # generate_speech() returns a (1, N) float tensor in [-1, 1] on CPU.
        audio_array = wav.detach().cpu().numpy().reshape(-1)
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
    finally:
        # Clean up the temporary reference audio file.
        cleanup_temp(prompt_audio_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMP_AUDIO_DIR = temp_audio_dir("lux_tts_rest_api")


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
# Main (for running directly: python server_luxTTS.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("LUX_TTS_HOST", "0.0.0.0")
    port = int(os.getenv("LUX_TTS_PORT", "7500"))
    logger.info("Starting LuxTTS REST API server on %s:%d", host, port)
    # Pass the app object directly instead of a module path string,
    # so this works regardless of how the file is invoked.
    uvicorn.run(app, host=host, port=port, log_level="info")
