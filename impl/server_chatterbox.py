"""
FastAPI REST server for Chatterbox Multilingual voice cloning.

Loads the model once on startup, then exposes a single POST endpoint for
synthesis.  Clients send text plus a reference audio sample (base64); the
server returns the generated audio as base64-encoded 24 kHz WAV.  All output
is PerTh-watermarked by the library itself.

Unlike most tts-serve engines, Chatterbox conditions purely on the reference
audio -- there is no transcript-of-the-reference field in the request schema,
so the `reference_text` core field is deliberately absent from this server.

Capabilities: GET /capabilities returns a machine-readable description of
every request parameter, derived from the Pydantic request model so it can
never drift from what the server actually validates (see the
tts-engine-common README).

Model weights are downloaded from HuggingFace (ResembleAI/chatterbox) on
first start.  Set HF_TOKEN in the environment if your checkpoint needs it.

Configuration (environment variables):
    CHATTERBOX_DEVICE    Device to load the model on.  One of: cuda, mps, cpu.
                         Default: cuda
    CHATTERBOX_T3_MODEL  Multilingual T3 checkpoint.  One of: v2, v3
                         (aliases t3_mtl23ls_v2 / t3_mtl23ls_v3), or a custom
                         .safetensors filename.  Default: v3 (latest release).
    CHATTERBOX_HOST      Bind host for `python server_chatterbox.py`.
                         Default: 0.0.0.0
    CHATTERBOX_PORT      Bind port for `python server_chatterbox.py`.
                         Default: 8000

Extra dependencies beyond the chatterbox-tts package:
    pip install fastapi uvicorn loguru soundfile
    pip install tts-engine-common    # or: pip install -e ../tts-engine-common

Usage:
    python server_chatterbox.py
    # or: uvicorn server_chatterbox:app --host 0.0.0.0 --port 8000
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

from chatterbox.mtl_tts import (
    MULTILINGUAL_T3_MODELS,
    S3GEN_SR,
    SUPPORTED_LANGUAGES,
    ChatterboxMultilingualTTS,
)
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

DEVICE = os.getenv("CHATTERBOX_DEVICE", "cuda")
T3_MODEL = os.getenv("CHATTERBOX_T3_MODEL", "v3")
MODEL_LABEL = f"chatterbox-multilingual-{T3_MODEL}"


def _validate_config() -> None:
    """Fail fast on bad configuration instead of partway through a model download."""
    if DEVICE not in SUPPORTED_DEVICES:
        raise ValueError(
            f"CHATTERBOX_DEVICE must be one of {SUPPORTED_DEVICES}, got {DEVICE!r}"
        )
    if T3_MODEL not in MULTILINGUAL_T3_MODELS and not T3_MODEL.endswith(".safetensors"):
        raise ValueError(
            f"CHATTERBOX_T3_MODEL must be one of {sorted(MULTILINGUAL_T3_MODELS)} "
            f"or a .safetensors filename, got {T3_MODEL!r}"
        )


_validate_config()

SEED_MIN = 1
SEED_MAX = 1000

# Heuristic lower bound: below this the speaker encoder / decoder conditioning
# degrade to near-garbage.  The model itself does not enforce a minimum.
MIN_PROMPT_DURATION_S = 2.0

# Sanity valve for the request payload (~5 min of 24 kHz audio).  Not a model
# limitation -- the reference clip is truncated to 10 s internally.
MAX_AUDIO_B64_LEN = 10_000_000

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

# Dynamic Literal built from the engine's own language table, so the request
# schema, the /capabilities enum, and engine-side validation share one source.
# (dict preserves insertion order, so the enum is stable across processes.)
Language = Literal[tuple(SUPPORTED_LANGUAGES)]


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
            "FLAC, ...).  The model uses only its first 10 s."
        ),
    )
    language: Language | None = Field(
        None,
        description=(
            "Language code, e.g. 'en', 'fr', 'zh' "
            f"(supported: {', '.join(SUPPORTED_LANGUAGES)}). "
            "Omit to skip the language tag."
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
    exaggeration: float = Field(
        0.5,
        ge=0.0,
        le=2.0,
        description=(
            "Expression/energy boost (README: ~0.5 general use, ~0.7+ for "
            "dramatic speech).  Higher values tend to speed up delivery."
        ),
    )
    cfg_weight: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Classifier-free guidance weight (README: ~0.5 general use, ~0.3 "
            "for fast-talking references or to reduce accent bleed from a "
            "foreign-language reference clip)."
        ),
    )
    temperature: float = Field(
        0.8,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the T3 language model.",
    )
    repetition_penalty: float = Field(
        1.2,
        ge=1.0,
        le=2.0,
        description="Penalty applied to repeated speech tokens.",
    )
    min_p: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="Min-p sampling threshold.",
    )
    top_p: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus (top-p) sampling threshold.",
    )

    @field_validator("text")
    @classmethod
    def _validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain non-whitespace characters")
        return v


class SynthesisResponse(CoreSynthesisResponse):
    """The synthesis result (core fields from tts_engine_common, plus fid)."""

    fid: str = Field(..., description="Request ID (internal).")


class HealthResponse(BaseModel):
    """Health / readiness check."""

    status: Literal["ok"] = "ok"
    serverType: Literal["Chatterbox"] = "Chatterbox"
    model: str = MODEL_LABEL
    device: str = DEVICE


# ---------------------------------------------------------------------------
# Capabilities (derived from SynthesisRequest — single source of truth)
# ---------------------------------------------------------------------------

CAPABILITIES = build_capabilities(
    SynthesisRequest,
    engine="chatterbox",
    model=MODEL_LABEL,
    device=DEVICE,
    sample_rate=S3GEN_SR,
    watermarked=True,  # the PerTh watermark is applied inside the library
    endpoint="/synthesize",
    reference_audio={
        "required": True,
        "formats": ["wav", "mp3", "ogg", "flac"],
        "min_duration_s": MIN_PROMPT_DURATION_S,
        "note": (
            "Only the first 10 s are used for speaker conditioning; longer "
            "clips are truncated."
        ),
    },
    languages=list(SUPPORTED_LANGUAGES),
    overrides={
        "exaggeration": {"step": 0.05},
        "cfg_weight": {"step": 0.05},
        "temperature": {"step": 0.05},
        "repetition_penalty": {"step": 0.05, "advanced": True},
        "min_p": {"step": 0.01, "advanced": True},
        "top_p": {"step": 0.01, "advanced": True},
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
    title="Chatterbox Cloning API",
    description=(
        "REST API around Chatterbox Multilingual.  Send text + a reference "
        "audio sample and get back cloned speech.  Synthesis requests are "
        "serialized (single shared model).  Machine-readable parameter "
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
# Runtime — thin wrapper around the Chatterbox model
# ---------------------------------------------------------------------------


@dataclass
class ChatterboxRuntime:
    """Holds the loaded model and its metadata for the lifetime of the server."""

    model: ChatterboxMultilingualTTS
    sample_rate: int
    device: str


_runtime: ChatterboxRuntime | None = None

# Chatterbox.generate() with a fresh audio_prompt_path rewrites the model's
# shared `conds` state in place, so concurrent requests would stomp on each
# other's speaker conditioning.  Serialize synthesis; single-GPU throughput
# is the bottleneck anyway.
_synthesis_lock = threading.Lock()


def _get_runtime() -> ChatterboxRuntime:
    """Return the global runtime, loading the model once on first call."""
    global _runtime
    if _runtime is None:
        logger.info(
            "Loading %s on device '%s' (first run downloads from HuggingFace) ...",
            MODEL_LABEL,
            DEVICE,
        )
        model = ChatterboxMultilingualTTS.from_pretrained(
            device=DEVICE,
            t3_model=T3_MODEL,
        )
        _runtime = ChatterboxRuntime(
            model=model,
            sample_rate=model.sr,
            device=DEVICE,
        )
        logger.info(
            "Model loaded successfully. Sampling rate: %d Hz, device: %s",
            _runtime.sample_rate,
            DEVICE,
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
    <head><title>Chatterbox REST API</title></head>
    <body>
        <h1>Chatterbox Cloning REST API</h1>
        <p>Model: <code>{MODEL_LABEL}</code> on <code>{DEVICE}</code>.
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
        "Synthesizing: seed={}, text_len={}, exaggeration={:.2f}, cfg={:.2f}, "
        "temp={:.2f}, rep_penalty={:.2f}, min_p={:.2f}, top_p={:.2f}, lang={}",
        seed,
        len(req.text),
        req.exaggeration,
        req.cfg_weight,
        req.temperature,
        req.repetition_penalty,
        req.min_p,
        req.top_p,
        req.language,
    )

    # Decode and sanity-check the reference audio before touching the model.
    try:
        raw_audio = decode_base64(req.audio_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")

    _check_reference_audio(raw_audio)

    prompt_audio_path = _write_temp_audio(raw_audio)

    try:
        # Time the actual synthesis call.
        t0 = time.perf_counter()

        with _synthesis_lock:
            seed_everything(seed)
            wav = runtime.model.generate(
                text=req.text,
                language_id=req.language,
                audio_prompt_path=prompt_audio_path,
                exaggeration=req.exaggeration,
                cfg_weight=req.cfg_weight,
                temperature=req.temperature,
                repetition_penalty=req.repetition_penalty,
                min_p=req.min_p,
                top_p=req.top_p,
            )

        time_used = time.perf_counter() - t0

        # generate() returns a (1, N) float tensor.
        audio_array = wav[0].detach().cpu().numpy()
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
        _cleanup_temp(prompt_audio_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMP_AUDIO_DIR = Path("/tmp/chatterbox_rest_api")
_TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


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


def _write_temp_audio(raw_bytes: bytes) -> str:
    """
    Write raw audio bytes to a temporary file.

    The model expects a file path (librosa loads it from disk).  The
    .wav extension is cosmetic -- soundfile sniffs the container from the
    header, so MP3/OGG/FLAC bytes work fine.
    """
    path = _TEMP_AUDIO_DIR / f"{uuid.uuid4().hex}.wav"
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
# Main (for running directly: python server_chatterbox.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("CHATTERBOX_HOST", "0.0.0.0")
    port = int(os.getenv("CHATTERBOX_PORT", "8000"))
    logger.info("Starting Chatterbox REST API server on %s:%d", host, port)
    # Pass the app object directly instead of a module path string,
    # so this works regardless of how the file is invoked.
    uvicorn.run(app, host=host, port=port, log_level="info")
