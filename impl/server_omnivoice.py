"""
FastAPI REST server for OmniVoice voice cloning.

Loads the model once on startup, then exposes a single POST endpoint for
synthesis.  Clients send text, a reference audio sample (base64), and the
transcript of that sample.  The server returns the generated audio as
base64-encoded WAV.

Usage:
    python server.py
    # or: uvicorn server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import io
import math
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

# No sys.path manipulation needed: 'uv sync' installs omnivoice as an
# editable package, so 'from omnivoice import ...' resolves directly.

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from loguru import logger  # noqa: E402
from pydantic import BaseModel, Field, field_validator  # noqa: E402
from typing_extensions import Literal  # noqa: E402

from omnivoice import OmniVoice, OmniVoiceGenerationConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# You can specify the huggingface id of the model: "k2-fsa/OmniVoice"
# That will pull it from your local cache, or download it if you don't have it.
#
# Alternatively, you can download it to some local directory,
# and specify the path: "/home/user/my-models/OmniVoice/"
MODEL_NAME_OR_PATH = "k2-fsa/OmniVoice"

DEFAULT_PRECISION = "bfloat16"
DEFAULT_GUIDANCE_SCALE = 1.2
DEFAULT_SPEAKER_SCALE = 1.5
DEFAULT_ODE_METHOD = "euler"
SEED_MIN = 1
SEED_MAX = 1000
NUM_STEPS_MIN = 10
NUM_STEPS_MAX = 20

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class SynthesisRequest(BaseModel):
    """A single synthesis request."""

    text: str = Field(..., description="Text to synthesize, e.g. 'Hello there'")
    audio_base64: str = Field(
        ...,
        description=(
            "Reference audio sample (~10 s WAV) encoded as a base64 string. "
            "The server will decode this to a temporary WAV file before synthesis."
        ),
    )
    prompt_text: str = Field(
        ...,
        description=(
            "Exact transcript of the reference audio, e.g. "
            "'Hello, my name is Alice, nice to meet you'."
        ),
    )

    # Optional overrides (defaults are sensible for voice cloning)
    seed: int | None = Field(
        None,
        ge=SEED_MIN,
        le=SEED_MAX,
        description=(
            "Random seed for reproducibility.  If omitted, a random seed "
            f"in [{SEED_MIN}, {SEED_MAX}] is chosen."
        ),
    )
    num_steps: int | None = Field(
        None,
        ge=NUM_STEPS_MIN,
        le=NUM_STEPS_MAX,
        description=(
            "Number of flow-matching sampling steps.  If omitted, a random "
            f"integer in [{NUM_STEPS_MIN}, {NUM_STEPS_MAX}] is chosen."
        ),
    )
    guidance_scale: float = Field(
        DEFAULT_GUIDANCE_SCALE,
        description="Classifier-free guidance scale.",
    )
    speaker_scale: float = Field(
        DEFAULT_SPEAKER_SCALE,
        description="Scale applied to the reference speaker embedding.",
    )
    ode_method: Literal["euler", "midpoint", "rk4"] = Field(
        DEFAULT_ODE_METHOD,
        description="ODE / flow-matching solver method.",
    )
    language: str | None = Field(
        None,
        description="Language tag (e.g. 'EN', 'ZH').  None = auto.",
    )


class SynthesisResponse(BaseModel):
    """The synthesis result."""

    audio_base64: str = Field(
        ...,
        description="Generated WAV audio encoded as a base64 string.",
    )
    sample_rate: int = Field(..., description="Audio sample rate (Hz).")
    seed: int = Field(..., description="Seed used for this generation.")
    num_steps: int = Field(..., description="Number of steps used.")
    fid: str = Field(..., description="Request ID (internal).")
    time_used: float = Field(
        ..., description="Wall-clock generation time in seconds."
    )
    rtf: float | None = Field(
        ...,
        description=(
            "Real-time factor (time / audio duration).  "
            "null when audio duration is zero."
        ),
    )

    @field_validator("rtf", mode="before")
    @classmethod
    def _sanitize_rtf(cls, v: float | None) -> float | None:
        """Replace inf / nan with None so JSON serialization never fails."""
        if v is None:
            return None
        if math.isinf(v) or math.isnan(v):
            return None
        return v

    @field_validator("time_used", mode="before")
    @classmethod
    def _sanitize_time_used(cls, v: float | None) -> float | None:
        """Guard against inf / nan in time_used as well."""
        if v is None:
            return None
        if math.isinf(v) or math.isnan(v):
            return 0.0
        return v


class HealthResponse(BaseModel):
    """Health / readiness check."""

    status: Literal["ok"] = "ok"
    serverType: Literal["OmniVoice"] = "OmniVoice"
    model: str = MODEL_NAME_OR_PATH


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OmniVoice Cloning API",
    description=(
        "REST API around OmniVoice.  Send text + a reference audio sample "
        "and get back cloned speech."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Runtime — thin wrapper around the OmniVoice model
# ---------------------------------------------------------------------------


@dataclass
class OmniVoiceRuntime:
    """Holds the loaded model and its metadata for the lifetime of the server."""

    model: OmniVoice
    sample_rate: int


# ---------------------------------------------------------------------------
# Lazy model loading — loaded on first request (or explicitly via startup).
# ---------------------------------------------------------------------------

_runtime: OmniVoiceRuntime | None = None


def _get_dtype(precision: str) -> torch.dtype:
    """Map a precision string to a torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if precision not in dtype_map:
        raise ValueError(f"Unsupported precision: {precision}. Choose from {list(dtype_map.keys())}")
    return dtype_map[precision]


def _get_runtime() -> OmniVoiceRuntime:
    """Return the global runtime, loading the model once on first call."""
    global _runtime
    if _runtime is None:
        logger.info(
            "Loading OmniVoice model '%s' (precision=%s) ...",
            MODEL_NAME_OR_PATH,
            DEFAULT_PRECISION,
        )
        dtype = _get_dtype(DEFAULT_PRECISION)
        model = OmniVoice.from_pretrained(
            MODEL_NAME_OR_PATH,
            device_map="cuda",
            torch_dtype=dtype,
        )
        _runtime = OmniVoiceRuntime(
            model=model,
            sample_rate=model.sampling_rate,
        )
        logger.info(
            "Model loaded successfully. Sampling rate: %d Hz, device: %s",
            _runtime.sample_rate,
            model.device,
        )
    return _runtime


# ---------------------------------------------------------------------------
# Global exception handler — catches anything that slips past endpoint handlers
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def _unhandled_exception(
    _request, exc: Exception
) -> JSONResponse:
    """Return a meaningful 500 instead of FastAPI's blank ``detail: ''``."""
    logger.error("Unhandled exception: {}", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )


# ---------------------------------------------------------------------------
# Startup / shutdown hooks
# ---------------------------------------------------------------------------


@app.on_event("startup")
def startup() -> None:
    """Pre-load the model so the first request is fast."""
    _get_runtime()


@app.on_event("shutdown")
def shutdown() -> None:
    """Clean up GPU resources."""
    global _runtime
    if _runtime is not None:
        # Release CUDA memory.
        del _runtime.model
        _runtime = None
    torch.cuda.empty_cache()
    logger.info("Model unloaded and CUDA cache cleared.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, tags=["System"])
def root() -> str:
    """A friendly landing page so browser visitors don't get the auto-generated docs."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>OmniVoice REST API</title></head>
    <body>
        <h1>OmniVoice Cloning REST API</h1>
        <p>This server is configured and running. It is a REST API, not a web server.</p>
        <p>Use a REST client like <strong>Postman</strong>, <strong>Insomnia</strong>,
        or <strong>curl</strong> to make requests.</p>
        <ul>
            <li><code>GET /health</code> &mdash; Check server status</li>
            <li><code>POST /synthesize</code> &mdash; Generate cloned speech</li>
        </ul>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Check whether the server is alive and the model is loaded."""
    model_loaded = _runtime is not None
    if not model_loaded:
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

    Parameters
    ----------
    req : SynthesisRequest
        - **text**: The text to speak.
        - **audio_base64**: Base64-encoded WAV reference audio.
        - **prompt_text**: Transcript of the reference audio.
        - **seed** (optional): Random seed (1–1000).  Random if omitted.
        - **num_steps** (optional): Sampling steps (10–20).  Random if omitted.
        - **guidance_scale**, **speaker_scale**, **ode_method**, **language**:
          Optional tuning parameters.

    Returns
    -------
    SynthesisResponse
        Base64-encoded WAV audio, sample rate, seed, steps, timing metrics.
    """
    runtime = _get_runtime()

    # Resolve randomised parameters.
    seed = req.seed if req.seed is not None else random.randint(SEED_MIN, SEED_MAX)
    num_steps = (
        req.num_steps
        if req.num_steps is not None
        else random.randint(NUM_STEPS_MIN, NUM_STEPS_MAX)
    )

    logger.info(
        "Synthesizing: seed={}, steps={}, text_len={}, guidance={:.1f}, "
        "speaker_scale={:.1f}, ode={}, lang={}",
        seed,
        num_steps,
        len(req.text),
        req.guidance_scale,
        req.speaker_scale,
        req.ode_method,
        req.language,
    )

    # Set the random seed for reproducibility.
    seed_everything(seed)

    # Decode the base64 audio and write to a temporary file.
    try:
        raw_audio = base64.b64decode(req.audio_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")

    prompt_audio_path = _write_temp_wav(raw_audio)

    try:
        # Time the actual synthesis call.
        t0 = time.perf_counter()

        # Build generation config from request parameters.
        # OmniVoice's generate() accepts **kwargs that get forwarded to
        # OmniVoiceGenerationConfig.from_dict(), so we pass num_step and
        # guidance_scale directly.
        gen_kwargs = {
            "num_step": num_steps,
            "guidance_scale": req.guidance_scale,
        }

        # OmniVoice uses natural language names like "English", not codes like "EN".
        # If the client sent a short code, try to map it.
        language = req.language
        if language is not None:
            lang_map = {"en": "English", "zh": "Chinese", "ja": "Japanese",
                         "ko": "Korean", "fr": "French", "de": "German",
                         "es": "Spanish", "it": "Italian", "pt": "Portuguese",
                         "ru": "Russian", "ar": "Arabic", "hi": "Hindi"}
            language = lang_map.get(language.lower(), language)

        # generate() returns list[np.ndarray] — one 1-D array per input text.
        audios = runtime.model.generate(
            text=req.text,
            ref_audio=prompt_audio_path,
            ref_text=req.prompt_text,
            language=language,
            generation_config=OmniVoiceGenerationConfig.from_dict(gen_kwargs),
        )

        time_used = time.perf_counter() - t0

        # Extract the first (and only) audio array.
        audio_array = audios[0]
        sample_rate = runtime.sample_rate

        # Compute audio duration and RTF.
        audio_duration = len(audio_array) / sample_rate if sample_rate else 0.0
        rtf = time_used / audio_duration if audio_duration > 0 else None

        # Encode the output WAV to base64.
        audio_bytes = _numpy_to_wav_bytes(audio_array, sample_rate)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        logger.info(
            "Synthesis complete: %.1f s wall-clock, %.1f s audio, RTF=%.3f",
            time_used, audio_duration, rtf if rtf else 0,
        )

        return SynthesisResponse(
            audio_base64=audio_b64,
            sample_rate=sample_rate,
            seed=seed,
            num_steps=num_steps,
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

_TEMP_AUDIO_DIR = Path("/tmp/omnivoice_rest_api")
_TEMP_AUDIO_DIR.mkdir(exist_ok=True)


def seed_everything(seed: int) -> None:
    """Set the random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_temp_wav(raw_bytes: bytes) -> str:
    """
    Write raw audio bytes to a temporary WAV file.

    The input is expected to be a valid WAV file (just bytes).  We write it
    directly to disk because the runtime expects a file path.
    """
    path = _TEMP_AUDIO_DIR / f"{uuid.uuid4().hex}.wav"
    path.write_bytes(raw_bytes)
    logger.debug("Wrote temporary prompt audio: {}", path)
    return str(path)


def _cleanup_temp(path: str) -> None:
    """Remove a temporary audio file if it exists."""
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _numpy_to_wav_bytes(audio_array, sample_rate: int) -> bytes:
    """
    Convert a numpy audio array to WAV-encoded bytes.

    OmniVoice.generate() returns np.ndarray at 24 kHz, so this is the
    fast path that avoids an unnecessary torch round-trip.
    """
    buffer = io.BytesIO()
    sf.write(
        buffer,
        audio_array,
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Main (for running directly: python -m apps.rest_api.server)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting OmniVoice REST API server on 0.0.0.0:8000")
    # Pass the app object directly instead of a module path string,
    # so this works regardless of how the file is invoked.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
