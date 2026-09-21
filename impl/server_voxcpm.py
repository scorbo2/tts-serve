"""
FastAPI REST server for VoxCPM2 voice cloning and voice design.

Loads the model once on startup, then exposes a single POST endpoint for
synthesis.  Clients send text plus an optional reference audio sample (base64);
the server returns the generated audio as base64-encoded 48 kHz WAV.

VoxCPM2 supports three modes:

1. **Voice Design** — no reference audio needed; describe the desired voice
   in parentheses at the start of ``text`` (e.g.
   ``"(A young woman, gentle voice)Hello there."``).
2. **Controllable Cloning** — provide reference audio only; the model clones
   the timbre.
3. **Ultimate Cloning** — provide reference audio *and* its exact transcript;
   the model treats the reference as a spoken prefix and continues from it,
   faithfully reproducing every vocal detail.

The engine auto-detects language from text content (30 languages supported
internally), so the ``language`` field is accepted but not forwarded.

Capabilities: GET /capabilities returns a machine-readable description of
every request parameter, derived from the Pydantic request model so it can
never drift from what the server actually validates (see the
tts-engine-common README).

Model weights are downloaded from HuggingFace (openbmb/VoxCPM2) on first
start.  Set HF_TOKEN in the environment if your checkpoint needs it.

Configuration (environment variables):
    VOXCPM_MODEL      HuggingFace id or local path.
                      Default: openbmb/VoxCPM2
    VOXCPM_DEVICE     Device to load the model on.  One of: auto, cuda, mps,
                      cpu.  Default: cuda
    VOXCPM_HOST       Bind host for `python server_voxcpm.py`.
                      Default: 0.0.0.0
    VOXCPM_PORT       Bind port for `python server_voxcpm.py`.
                      Default: 7500

Extra dependencies beyond the voxcpm package:
    pip install fastapi uvicorn loguru soundfile
    pip install ../tts-engine-common # in-repo copy; or: pip install -e ../tts-engine-common

Usage:
    python server_voxcpm.py
    # or: uvicorn server_voxcpm:app --host 0.0.0.0 --port 7500
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
from voxcpm import VoxCPM
from voxcpm.model.utils import resolve_runtime_device
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

MODEL_NAME_OR_PATH = os.getenv("VOXCPM_MODEL", "openbmb/VoxCPM2")
DEVICE = os.getenv("VOXCPM_DEVICE", "cuda")

# VoxCPM2 outputs 48 kHz studio-quality audio via AudioVAE V2's asymmetric
# encode/decode design (16 kHz input → 48 kHz output).
SAMPLE_RATE = 48000

SEED_MIN = 1
SEED_MAX = 1000

# Heuristic lower bound: below this the voice cloning degrades to near-garbage.
MIN_PROMPT_DURATION_S = 2.0

# Sanity valve for the request payload (~3.5 min of 48 kHz audio).
MAX_AUDIO_B64_LEN = 10_000_000


def _validate_config() -> None:
    """Fail fast on bad configuration instead of partway through a model download."""
    # VoxCPM's resolve_runtime_device validates at load time, but we check
    # early here so the server fails before any HF download.
    explicit = DEVICE.strip().lower()
    valid = ("auto", "cpu", "mps", "cuda")
    if explicit not in valid and not explicit.startswith("cuda:"):
        raise ValueError(
            f"VOXCPM_DEVICE must be one of {valid} or cuda:N, got {DEVICE!r}"
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
    audio_base64: str | None = Field(
        None,
        min_length=1,
        max_length=MAX_AUDIO_B64_LEN,
        description=(
            "Reference voice sample as a base64 string (optional).  "
            "Any container soundfile can decode (WAV, MP3, OGG, FLAC, ...).  "
            "~3 s is enough for high-quality cloning.  "
            "Omit for voice design mode (describe the voice in parentheses "
            "at the start of text, e.g. \"(A young woman, gentle voice) "
            "Hello there.\")."
        ),
    )
    reference_text: str | None = Field(
        None,
        description=(
            "Exact transcript of the reference audio (optional).  "
            "When provided alongside audio_base64, enables ultimate cloning "
            "mode: the model treats the reference as a spoken prefix and "
            "continues from it, faithfully reproducing every vocal detail.  "
            "Without reference_text, only controllable cloning (timbre copy) "
            "is used."
        ),
    )
    language: str | None = Field(
        DEFAULT_LANGUAGE,
        description=(
            "Two-letter language code, e.g. 'en', 'zh' "
            "(accepted but not forwarded — VoxCPM2 auto-detects language "
            "from text content across 30 languages).  "
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
    cfg_value: float | None = Field(
        None,
        ge=0.0,
        le=10.0,
        description=(
            "Classifier-free guidance scale.  Higher values increase "
            "expressiveness and adherence to style cues.  "
            "Omit for the engine default (2.0)."
        ),
    )
    inference_timesteps: int | None = Field(
        None,
        ge=1,
        le=50,
        description=(
            "Number of diffusion inference steps.  More steps = higher "
            "quality but slower.  Omit for the engine default (10)."
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
    def _validate_language(cls, v: str) -> str:
        return validate_language_code(v)


class SynthesisResponse(CoreSynthesisResponse):
    """The synthesis result (core fields from tts_engine_common, plus fid)."""

    fid: str = Field(..., description="Request ID (internal).")


class HealthResponse(BaseModel):
    """Health / readiness check."""

    status: Literal["ok"] = "ok"
    serverType: Literal["VoxCPM"] = "VoxCPM"
    model: str = MODEL_NAME_OR_PATH
    device: str = DEVICE

# ---------------------------------------------------------------------------
# Capabilities (derived from SynthesisRequest — single source of truth)
# ---------------------------------------------------------------------------

CAPABILITIES = build_capabilities(
    SynthesisRequest,
    engine="voxcpm",
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
            "Optional: omit for voice design mode (describe the voice in "
            "parentheses at the start of text).  ~3 s is enough for "
            "high-quality cloning.  Provide reference_text for ultimate "
            "cloning (audio continuation mode)."
        ),
    },
    languages=None,  # engine auto-detects language from text (30 langs)
    overrides={
        "cfg_value": {"step": 0.5},
        "inference_timesteps": {"step": 1, "advanced": True},
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
    title="VoxCPM Voice Cloning API",
    description=(
        "REST API around VoxCPM2.  Send text + an optional reference audio "
        "sample and get back studio-quality 48 kHz speech.  Supports voice "
        "design (no reference), controllable cloning, and ultimate cloning "
        "(with reference transcript).  Machine-readable parameter metadata "
        "at GET /capabilities."
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
# Runtime — thin wrapper around the VoxCPM model
# ---------------------------------------------------------------------------


@dataclass
class VoxCPMRuntime:
    """Holds the loaded model and its metadata for the lifetime of the server."""

    model: VoxCPM
    sample_rate: int
    device: str


_runtime: VoxCPMRuntime | None = None

# VoxCPM's KV cache and prompt cache are shared mutable state between calls,
# so concurrent requests would stomp on each other's generation state.
_synthesis_lock = threading.Lock()


def _get_runtime() -> VoxCPMRuntime:
    """Return the global runtime, loading the model once on first call."""
    global _runtime
    if _runtime is None:
        logger.info(
            "Loading VoxCPM model '%s' on device '%s' ...",
            MODEL_NAME_OR_PATH,
            DEVICE,
        )
        resolved_device = resolve_runtime_device(DEVICE, "cuda")
        model = VoxCPM.from_pretrained(
            hf_model_id=MODEL_NAME_OR_PATH,
            load_denoiser=False,  # denoiser adds overhead, not needed for synthesis
            optimize=True,
            device=resolved_device,
        )
        _runtime = VoxCPMRuntime(
            model=model,
            sample_rate=SAMPLE_RATE,
            device=resolved_device,
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
    <head><title>VoxCPM REST API</title></head>
    <body>
        <h1>VoxCPM Voice Cloning REST API</h1>
        <p>Model: <code>{MODEL_NAME_OR_PATH}</code> on <code>{DEVICE}</code>.
        This server is a REST API, not a web server.</p>
        <p>Use a REST client like <strong>Postman</strong>, <strong>Insomnia</strong>,
        or <strong>curl</strong> to make requests (interactive docs at <a href="/docs">/docs</a>).</p>
        <ul>
            <li><code>GET /capabilities</code> &mdash; Machine-readable parameter metadata</li>
            <li><code>GET /health</code> &mdash; Check server status</li>
            <li><code>POST /synthesize</code> &mdash; Generate speech</li>
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
    summary="Synthesize speech from text + optional reference audio",
)
def synthesize(req: SynthesisRequest) -> SynthesisResponse:
    """
    Synthesize audio using the provided text and optional reference audio.

    Three modes are supported:
    - **Voice Design**: no reference audio; describe the voice in parentheses
      at the start of ``text`` (e.g. ``"(A young woman, gentle voice)Hello."``).
    - **Controllable Cloning**: reference audio only (timbre copy).
    - **Ultimate Cloning**: reference audio + transcript (audio continuation).

    The full parameter list is documented at GET /capabilities; the request
    schema mirrors it exactly (same model, no drift).
    """
    runtime = _get_runtime()

    # Resolve randomised seed for reproducibility.
    seed = req.seed if req.seed is not None else random.randint(SEED_MIN, SEED_MAX)

    logger.info(
        "Synthesizing: seed=%d, text_len=%d, has_ref=%s, has_ref_text=%s, "
        "cfg_value=%s, timesteps=%s",
        seed,
        len(req.text),
        req.audio_base64 is not None,
        req.reference_text is not None,
        req.cfg_value,
        req.inference_timesteps,
    )

    # Decode and sanity-check the reference audio before touching the model.
    prompt_audio_path: str | None = None
    ref_audio_path: str | None = None

    if req.audio_base64 is not None:
        try:
            raw_audio = decode_base64(req.audio_base64)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")

        _check_reference_audio(raw_audio)

        # Stage to a temp file; the engine requires a file path.
        # UUID naming, deleted after synthesis (one-shot engine).
        prompt_audio_path = write_temp_audio(raw_audio, _TEMP_AUDIO_DIR)
        ref_audio_path = prompt_audio_path

    try:
        t0 = time.perf_counter()

        with _synthesis_lock:
            # Build kwargs for the engine's generate() method.
            generate_kwargs: dict = {
                "text": req.text,
                "seed": seed,
            }
            if req.cfg_value is not None:
                generate_kwargs["cfg_value"] = req.cfg_value
            if req.inference_timesteps is not None:
                generate_kwargs["inference_timesteps"] = req.inference_timesteps

            # Ultimate cloning: reference audio + transcript (audio continuation)
            if req.reference_text is not None:
                generate_kwargs["prompt_wav_path"] = ref_audio_path
                generate_kwargs["prompt_text"] = req.reference_text
                generate_kwargs["reference_wav_path"] = ref_audio_path
            elif req.audio_base64 is not None:
                # Controllable cloning: reference audio only
                generate_kwargs["reference_wav_path"] = ref_audio_path
            # else: voice design mode — no reference audio at all

            wav = runtime.model.generate(**generate_kwargs)

        time_used = time.perf_counter() - t0

        # wav is a 1-D numpy array (float32) on CPU.
        sample_rate = runtime.sample_rate

        rtf = compute_rtf(time_used, len(wav), sample_rate)

        # Encode the output WAV to base64.
        audio_bytes = _numpy_to_wav_bytes(wav, sample_rate)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        audio_duration = len(wav) / sample_rate if sample_rate else 0.0
        logger.info(
            "Synthesis complete: %.1f s wall-clock, %.1f s audio, RTF=%s",
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
        # Clean up the temporary reference audio file(s) — guard against None
        # for voice-design mode where no reference audio is provided.
        if prompt_audio_path is not None:
            cleanup_temp(prompt_audio_path)
        if ref_audio_path is not None and ref_audio_path != prompt_audio_path:
            cleanup_temp(ref_audio_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMP_AUDIO_DIR = temp_audio_dir("voxcpm_rest_api")


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
# Main (for running directly: python server_voxcpm.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("VOXCPM_HOST", "0.0.0.0")
    port = int(os.getenv("VOXCPM_PORT", "7500"))
    logger.info("Starting VoxCPM REST API server on %s:%d", host, port)
    # Pass the app object directly instead of a module path string,
    # so this works regardless of how the file is invoked.
    uvicorn.run(app, host=host, port=port, log_level="info")
