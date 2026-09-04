"""
FastAPI REST server for OmniVoice voice cloning.

Loads the model once on startup, then exposes a single POST endpoint for
synthesis.  Clients send text plus a reference audio sample (base64); the
transcript of that sample is optional (the engine auto-transcribes it with
Whisper when omitted).  The server returns the generated audio as
base64-encoded 24 kHz WAV.

Capabilities: GET /capabilities returns a machine-readable description of
every request parameter, derived from the Pydantic request model so it can
never drift from what the server actually validates (see the
tts-engine-common README).

Model weights are downloaded from HuggingFace (k2-fsa/OmniVoice) on first
start unless a local path is given.

Configuration (environment variables):
    OMNIVOICE_MODEL      HuggingFace id or local path.
                         Default: k2-fsa/OmniVoice
    OMNIVOICE_DEVICE     Device to load the model on.  One of: cuda, mps, cpu.
                         Default: cuda
    OMNIVOICE_HOST       Bind host for `python server_omnivoice.py`.
                         Default: 0.0.0.0
    OMNIVOICE_PORT       Bind port for `python server_omnivoice.py`.
                         Default: 8000

Extra dependencies beyond the omnivoice package:
    pip install fastapi uvicorn loguru soundfile
    pip install tts-engine-common    # or: pip install -e ../tts-engine-common

Usage:
    python server_omnivoice.py
    # or: uvicorn server_omnivoice:app --host 0.0.0.0 --port 8000
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
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.audio import load_audio_bytes
from tts_engine_common import (
    CoreSynthesisResponse,
    build_capabilities,
    capabilities_endpoint,
    compute_rtf,
    decode_base64,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_DEVICES = ("cuda", "mps", "cpu")

MODEL_NAME_OR_PATH = os.getenv("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
DEVICE = os.getenv("OMNIVOICE_DEVICE", "cuda")

# Output sample rate is set by the audio tokenizer's feature extractor
# (higgs-audio-v2, 24 kHz); model.sampling_rate is derived from it at load
# time and echoed in every response.
SAMPLE_RATE = 24000

SEED_MIN = 1
SEED_MAX = 1000

# Heuristic lower bound: below this the speaker conditioning (and any
# auto-transcription) degrade to near-garbage.
MIN_REF_DURATION_S = 2.0

# The engine trims references >20 s at the largest silence gap; we document
# (and reject absurd uploads with) the same figure.
MAX_REF_DURATION_S = 20.0

# Sanity valve for the request payload (~5 min of 24 kHz audio).
MAX_AUDIO_B64_LEN = 10_000_000

DEFAULT_PRECISION = "bfloat16"


def _validate_config() -> None:
    """Fail fast on bad configuration instead of partway through a model download."""
    if DEVICE not in SUPPORTED_DEVICES:
        raise ValueError(
            f"OMNIVOICE_DEVICE must be one of {SUPPORTED_DEVICES}, got {DEVICE!r}"
        )


_validate_config()

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
            "Reference voice sample (3-10 s recommended) as a base64 string. "
            "Any container soundfile can decode (WAV, MP3, OGG, FLAC, ...). "
            "Clips over 20 s are trimmed at the largest silence gap."
        ),
    )
    reference_text: str | None = Field(
        None,
        description=(
            "Exact transcript of the reference audio.  If omitted, the "
            "reference is auto-transcribed with Whisper ASR (the first such "
            "request pays a one-time ASR model load, which can be slow)."
        ),
    )
    language: str | None = Field(
        None,
        description=(
            "Language code (e.g. 'en') or name (e.g. 'English').  Omit for "
            "language-agnostic mode."
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

    # --- engine-specific tuning (defaults mirror the engine's own defaults) -
    num_steps: int = Field(
        32,
        ge=4,
        le=128,
        description=(
            "Flow-matching sampling steps.  32 is the engine default; 16 is a "
            "reasonable fast/quality trade-off."
        ),
    )
    guidance_scale: float = Field(
        2.0,
        ge=0.0,
        le=10.0,
        description="Classifier-free guidance scale.",
    )
    denoise: bool = Field(
        True,
        description="Prepend the denoise token (recommended when the reference clip has background noise).",
    )

    @field_validator("text")
    @classmethod
    def _validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain non-whitespace characters")
        return v


class SynthesisResponse(CoreSynthesisResponse):
    """The synthesis result (core fields from tts_engine_common, plus extras)."""

    fid: str = Field(..., description="Request ID (internal).")
    num_steps: int = Field(..., description="Flow-matching sampling steps used.")


class HealthResponse(BaseModel):
    """Health / readiness check."""

    status: Literal["ok"] = "ok"
    serverType: Literal["OmniVoice"] = "OmniVoice"
    model: str = MODEL_NAME_OR_PATH
    device: str = DEVICE


# ---------------------------------------------------------------------------
# Capabilities (derived from SynthesisRequest — single source of truth)
# ---------------------------------------------------------------------------

CAPABILITIES = build_capabilities(
    SynthesisRequest,
    engine="omnivoice",
    model=MODEL_NAME_OR_PATH,
    device=DEVICE,
    sample_rate=SAMPLE_RATE,
    watermarked=False,
    endpoint="/synthesize",
    reference_audio={
        "required": True,
        "formats": ["wav", "mp3", "ogg", "flac"],
        "min_duration_s": MIN_REF_DURATION_S,
        "max_duration_s": MAX_REF_DURATION_S,
        "note": (
            "3-10 s recommended.  Clips over 20 s are trimmed at the largest "
            "silence gap and cloning quality degrades.  If reference_text is "
            "omitted, the clip is auto-transcribed with Whisper."
        ),
    },
    languages=None,  # 646 languages; code or name, free-form
    overrides={
        "num_steps": {"step": 4},
        "guidance_scale": {"step": 0.1},
        "denoise": {"advanced": True},
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
    title="OmniVoice Cloning API",
    description=(
        "REST API around OmniVoice.  Send text + a reference audio sample and "
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
# Runtime — thin wrapper around the OmniVoice model
# ---------------------------------------------------------------------------


@dataclass
class OmniVoiceRuntime:
    """Holds the loaded model and its metadata for the lifetime of the server."""

    model: OmniVoice
    sample_rate: int
    device: str


_runtime: OmniVoiceRuntime | None = None

# Synthesis is serialized for two reasons: the model's lazy Whisper-ASR load
# is not race-safe, and single-GPU throughput is the bottleneck anyway.
_synthesis_lock = threading.Lock()


def _get_dtype(precision: str) -> torch.dtype:
    """Map a precision string to a torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if precision not in dtype_map:
        raise ValueError(
            f"Unsupported precision: {precision}. Choose from {list(dtype_map.keys())}"
        )
    return dtype_map[precision]


def _get_runtime() -> OmniVoiceRuntime:
    """Return the global runtime, loading the model once on first call."""
    global _runtime
    if _runtime is None:
        logger.info(
            "Loading OmniVoice model '%s' on device '%s' (precision=%s) ...",
            MODEL_NAME_OR_PATH,
            DEVICE,
            DEFAULT_PRECISION,
        )
        model = OmniVoice.from_pretrained(
            MODEL_NAME_OR_PATH,
            device_map=DEVICE,
            torch_dtype=_get_dtype(DEFAULT_PRECISION),
        )
        _runtime = OmniVoiceRuntime(
            model=model,
            sample_rate=model.sampling_rate,
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
    <head><title>OmniVoice REST API</title></head>
    <body>
        <h1>OmniVoice Cloning REST API</h1>
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

    # Resolve randomised parameters.
    seed = req.seed if req.seed is not None else random.randint(SEED_MIN, SEED_MAX)

    logger.info(
        "Synthesizing: seed={}, steps={}, text_len={}, guidance={:.1f}, "
        "denoise={}, lang={}",
        seed,
        req.num_steps,
        len(req.text),
        req.guidance_scale,
        req.denoise,
        req.language,
    )

    # Decode and sanity-check the reference audio before touching the model.
    try:
        raw_audio = decode_base64(req.audio_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")

    _check_reference_audio(raw_audio)

    # In-memory (waveform, sr) tuple — the engine resamples if needed, so no
    # temporary file is required (unlike engines that insist on paths).
    try:
        ref_wav = load_audio_bytes(raw_audio, runtime.sample_rate)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not decode reference audio: {exc}"
        )

    try:
        # Time the actual synthesis call.
        t0 = time.perf_counter()

        with _synthesis_lock:
            seed_everything(seed)
            audios = runtime.model.generate(
                text=req.text,
                language=req.language,
                ref_audio=(ref_wav, runtime.sample_rate),
                ref_text=req.reference_text,
                generation_config=OmniVoiceGenerationConfig(
                    num_step=req.num_steps,
                    guidance_scale=req.guidance_scale,
                    denoise=req.denoise,
                ),
            )

        time_used = time.perf_counter() - t0

        # generate() returns list[np.ndarray] — one 1-D array per input text.
        audio_array = audios[0]
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
            num_steps=req.num_steps,
        )

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
# Main (for running directly: python server_omnivoice.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("OMNIVOICE_HOST", "0.0.0.0")
    port = int(os.getenv("OMNIVOICE_PORT", "8000"))
    logger.info("Starting OmniVoice REST API server on %s:%d", host, port)
    # Pass the app object directly instead of a module path string,
    # so this works regardless of how the file is invoked.
    uvicorn.run(app, host=host, port=port, log_level="info")
