"""
FastAPI REST server for dots.tts voice cloning.

Loads the model once on startup, then exposes a single POST endpoint for
synthesis.  Clients send text plus a reference audio sample (base64); the
transcript of that sample is optional.  The server returns the generated
audio as base64-encoded 48 kHz WAV (dots.tts' AudioVAE vocoder is 48 kHz —
unlike the 24 kHz of the other tts-serve engines).

Capabilities: GET /capabilities returns a machine-readable description of
every request parameter, derived from the Pydantic request model so it can
never drift from what the server actually validates (see the
tts-engine-common README).

The runtime picks the device itself (CUDA if available, else CPU); there is
no device configuration for this engine.

Model weights are downloaded from HuggingFace (rednote-hilab/dots.tts-soar)
on first start unless a local path is given.

Configuration (environment variables):
    DOTS_TTS_MODEL      HuggingFace id or local path.
                        Default: rednote-hilab/dots.tts-soar
    DOTS_TTS_HOST       Bind host for `python server_dotsTTS.py`.
                        Default: 0.0.0.0
    DOTS_TTS_PORT       Bind port for `python server_dotsTTS.py`.
                        Default: 8000

Extra dependencies beyond the dots.tts package:
    pip install fastapi uvicorn loguru soundfile
    pip install tts-engine-common    # or: pip install -e ../tts-engine-common

Usage:
    python server_dotsTTS.py
    # or: uvicorn server_dotsTTS:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import io
import math
import os
import random
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dots_tts.runtime import DotsTtsRuntime
from dots_tts.utils.util import seed_everything
from tts_engine_common import (
    DEFAULT_LANGUAGE,
    CoreSynthesisResponse,
    build_capabilities,
    capabilities_endpoint,
    decode_base64,
    normalize_language,
    validate_language_code,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME_OR_PATH = os.getenv("DOTS_TTS_MODEL", "rednote-hilab/dots.tts-soar")

# The runtime auto-selects CUDA when available (see DotsTtsRuntime.__init__);
# mirror its decision for the capabilities document.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The 48 kHz AudioVAE vocoder; the runtime reads it from the model config at
# load time and returns it in every result.
SAMPLE_RATE = 48000

SEED_MIN = 1
SEED_MAX = 1000

# Heuristic lower bound for usable speaker conditioning.
MIN_REF_DURATION_S = 2.0

# Sanity valve for the request payload (~2.5 min of 48 kHz audio).
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
            "FLAC, ...)."
        ),
    )
    reference_text: str | None = Field(
        None,
        description=(
            "Exact transcript of the reference audio.  Optional — audio-only "
            "cloning works, but the transcript improves conditioning."
        ),
    )
    language: str | None = Field(
        DEFAULT_LANGUAGE,
        description=(
            "Two-letter language code, e.g. 'en' or 'zh', or 'auto' for "
            "auto-detection.  Omitted or empty defaults to 'en'."
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
        10,
        ge=1,
        le=64,
        description="Flow-matching sampling steps.",
    )
    guidance_scale: float = Field(
        1.2,
        ge=0.0,
        le=5.0,
        description="Classifier-free guidance scale.",
    )
    speaker_scale: float = Field(
        1.5,
        ge=0.0,
        le=3.0,
        description="Scale applied to the reference speaker embedding.",
    )
    ode_method: Literal["euler", "midpoint", "rk4"] = Field(
        "euler",
        description="ODE / flow-matching solver method.",
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
        # docs/02: the API speaks two-letter codes; 'auto' is the
        # auto-detection sentinel (mapped to the engine's 'auto_detect'
        # before the model call).  The contract check itself lives in the
        # shared helper — allow_auto=True keeps 'auto' legal here.
        return validate_language_code(v, allow_auto=True)


class SynthesisResponse(CoreSynthesisResponse):
    """The synthesis result (core fields from tts_engine_common, plus extras)."""

    fid: str = Field(..., description="Request ID (internal).")
    num_steps: int = Field(..., description="Flow-matching sampling steps used.")


class HealthResponse(BaseModel):
    """Health / readiness check."""

    status: Literal["ok"] = "ok"
    serverType: Literal["dots.tts"] = "dots.tts"
    model: str = MODEL_NAME_OR_PATH
    device: str = DEVICE


# ---------------------------------------------------------------------------
# Capabilities (derived from SynthesisRequest — single source of truth)
# ---------------------------------------------------------------------------

CAPABILITIES = build_capabilities(
    SynthesisRequest,
    engine="dots.tts",
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
            "Roughly 10 s of clean, low-noise audio clones best.  The "
            "transcript (reference_text) is optional but improves conditioning."
        ),
    },
    languages=None,  # no fixed list; two-letter codes + 'auto' (docs/02)
    overrides={
        "num_steps": {"step": 1},
        "guidance_scale": {"step": 0.1},
        "speaker_scale": {"step": 0.1},
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
    title="dots.tts Voice Cloning API",
    description=(
        "REST API around dots.tts.  Send text + a reference audio sample and "
        "get back cloned speech.  Machine-readable parameter metadata at "
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
# Runtime
# ---------------------------------------------------------------------------


_runtime: DotsTtsRuntime | None = None


def _get_runtime() -> DotsTtsRuntime:
    """Return the global runtime, loading the model once on first call."""
    global _runtime
    if _runtime is None:
        logger.info(
            "Loading dots.tts model '%s' (precision=%s) ...",
            MODEL_NAME_OR_PATH,
            "bfloat16",
        )
        _runtime = DotsTtsRuntime.from_pretrained(
            MODEL_NAME_OR_PATH,
            precision="bfloat16",
        )
        logger.info(
            "Model loaded successfully. Sample rate: %d Hz, device: %s",
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
    <head><title>dots.tts REST API</title></head>
    <body>
        <h1>dots.tts Voice Cloning REST API</h1>
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
        "Synthesizing: seed={}, steps={}, text_len={}, guidance={:.1f}, "
        "speaker_scale={:.1f}, ode={}, lang={}",
        seed,
        req.num_steps,
        len(req.text),
        req.guidance_scale,
        req.speaker_scale,
        req.ode_method,
        req.language,
    )

    # Set the random seed for reproducibility (the engine's own helper also
    # pins cuDNN determinism, which the flow-matching solver cares about).
    seed_everything(seed)

    # Decode the reference audio.
    try:
        raw_audio = decode_base64(req.audio_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")

    _check_reference_audio(raw_audio)

    # The runtime insists on a file path for prompt audio, hence the temp
    # file (the .wav extension is cosmetic — the loader sniffs the header).
    prompt_audio_path = _write_temp_audio(raw_audio)

    try:
        result = runtime.generate(
            text=req.text,
            prompt_audio_path=prompt_audio_path,
            prompt_text=req.reference_text,
            ode_method=req.ode_method,
            num_steps=req.num_steps,
            guidance_scale=req.guidance_scale,
            speaker_scale=req.speaker_scale,
            language=_to_engine_language(req.language),
        )

        # The runtime computes time_used and rtf itself; rtf can be inf for
        # degenerate output, which the response model maps to null.
        audio_bytes = _tensor_to_wav_bytes(result["audio"], result["sample_rate"])
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        rtf = result["rtf"]
        logger.info(
            "Synthesis complete: {:.1f} s wall-clock, rtf={}",
            result["time_used"],
            f"{rtf:.3f}" if isinstance(rtf, (int, float)) and math.isfinite(rtf) else "n/a",
        )

        return SynthesisResponse(
            audio_base64=audio_b64,
            sample_rate=result["sample_rate"],
            seed=seed,
            fid=result["fid"],
            time_used=result["time_used"],
            rtf=result["rtf"],
            num_steps=req.num_steps,
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

_TEMP_AUDIO_DIR = Path("/tmp/dots_tts_rest_api")
_TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _to_engine_language(language: str) -> str:
    """Map the API-level language value to the form the dots.tts runtime wants.

    The runtime takes uppercase codes ('EN') or the 'auto_detect' sentinel;
    the API contract (docs/02) is lowercase two-letter codes or 'auto'.
    """
    if language == "auto":
        return "auto_detect"
    return language.upper()


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

    The runtime expects a file path for prompt audio.  The .wav extension is
    cosmetic — the loader sniffs the container from the header, so
    MP3/OGG/FLAC bytes work fine.
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


def _tensor_to_wav_bytes(audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
    """Convert the runtime's (1, T) float tensor to WAV-encoded bytes (PCM_16)."""
    buffer = io.BytesIO()
    # Clip to [-1, 1]: the PCM_16 conversion wraps out-of-range floats instead
    # of clamping them, which would produce crackling artifacts.
    sf.write(
        buffer,
        torch.clip(audio_tensor.float().cpu().squeeze(), -1.0, 1.0).numpy(),
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Main (for running directly: python server_dotsTTS.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("DOTS_TTS_HOST", "0.0.0.0")
    port = int(os.getenv("DOTS_TTS_PORT", "8000"))
    logger.info("Starting dots.tts REST API server on %s:%d", host, port)
    # Pass the app object directly instead of a module path string,
    # so this works regardless of how the file is invoked.
    uvicorn.run(app, host=host, port=port, log_level="info")
