"""
FastAPI REST server for Kyutai Pocket TTS.

Loads the model once on startup, then exposes a single POST endpoint for
synthesis.  Pocket TTS is a ~100M-parameter CPU-first model (no GPU needed,
GPU is not officially supported), so this is the lightweight option in
tts-serve: a few times faster than real time on a handful of CPU cores.

The voice comes from one of three places, in this order:
  1. ``audio_base64`` -- a reference clip, cloned zero-shot;
  2. ``voice``        -- one of the engine's built-in preset voices;
  3. neither          -- the loaded language's default preset voice.
``audio_base64`` and ``voice`` are mutually exclusive (422 if both are given).

Capabilities: GET /capabilities returns a machine-readable description of
every request parameter, derived from the Pydantic request model so it can
never drift from what the server actually validates (see the
tts-engine-common README).

Model weights are downloaded from HuggingFace (kyutai/pocket-tts) on first
start.  Voice cloning needs the gated weights: accept the terms at
https://huggingface.co/kyutai/pocket-tts and log in locally (`hf auth login`).
Without them the server still starts but only the preset voices work, and a
request carrying ``audio_base64`` is rejected with a 400.

Configuration (environment variables):
    POCKET_TTS_MODEL     Engine model/config name, passed to
                         ``TTSModel.load_model(language=...)``: a language
                         ("english", "french", "german", "portuguese",
                         "italian", "spanish", "dutch"), optionally with a
                         "_24l" suffix, or an "english_*" release.  One
                         instance serves exactly one language.
                         Default: english
    POCKET_TTS_QUANTIZE  1 to apply dynamic int8 quantization (about 27%
                         faster on x86 CPUs, no measurable quality loss;
                         install ``pocket-tts[quantize]`` for best speed).
                         Default: 0
    POCKET_TTS_HOST      Bind host for `python server_pocketTTS.py`.
                         Default: 0.0.0.0
    POCKET_TTS_PORT      Bind port for `python server_pocketTTS.py`.
                         Default: 7500

Install the engine (CPU-only PyTorch build -- avoids several GB of CUDA wheels):
    pip install pocket-tts --extra-index-url https://download.pytorch.org/whl/cpu

Extra dependencies beyond the Pocket TTS package:
    pip install fastapi uvicorn loguru soundfile
    pip install ../tts-engine-common # in-repo copy; or: pip install -e ../tts-engine-common

Usage:
    python server_pocketTTS.py
    # or: uvicorn server_pocketTTS:app --host 0.0.0.0 --port 7500
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import random
import threading
import time
import uuid
from collections import OrderedDict
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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pocket_tts import TTSModel
from pocket_tts.default_parameters import (
    DEFAULT_LANGUAGE as ENGINE_DEFAULT_MODEL,
    get_default_voice_for_language,
)
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

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

# The engine's language names -> the two-letter codes of docs/02.  A model
# name is "<language>" or "<language>_<variant>" (e.g. "french_24l",
# "english_2026-09"); the part before the first underscore picks the code.
LANGUAGE_NAME_TO_CODE = {
    "english": "en",
    "french": "fr",
    "german": "de",
    "portuguese": "pt",
    "italian": "it",
    "spanish": "es",
    "dutch": "nl",
}

MODEL_NAME = os.getenv("POCKET_TTS_MODEL", ENGINE_DEFAULT_MODEL)
QUANTIZE = os.getenv("POCKET_TTS_QUANTIZE", "0") == "1"
# Pocket TTS is CPU-only here (GPU is not officially supported by the engine;
# see "Running on GPU" in https://github.com/kyutai-labs/pocket-tts).
DEVICE = "cpu"


def _validate_config() -> None:
    """Fail fast on bad configuration instead of partway through a model download."""
    language_name = MODEL_NAME.split("_", 1)[0]
    if language_name not in LANGUAGE_NAME_TO_CODE:
        raise ValueError(
            f"POCKET_TTS_MODEL must start with one of {sorted(LANGUAGE_NAME_TO_CODE)}, "
            f"got {MODEL_NAME!r}"
        )
    if os.getenv("POCKET_TTS_QUANTIZE", "0") not in ("0", "1"):
        raise ValueError(
            f"POCKET_TTS_QUANTIZE must be 0 or 1, got {os.getenv('POCKET_TTS_QUANTIZE')!r}"
        )


_validate_config()

# The one language this process speaks (the model is fixed at load time).
LANGUAGE_NAME = MODEL_NAME.split("_", 1)[0]
LANGUAGE_CODE = LANGUAGE_NAME_TO_CODE[LANGUAGE_NAME]
MODEL_LABEL = f"pocket-tts-{MODEL_NAME}"

# All shipped configs use a 24 kHz Mimi codec; the response reports the loaded
# model's own rate.
SAMPLE_RATE = 24000

SEED_MIN = 1
SEED_MAX = 1000

# Heuristic lower bound: the engine has no stated minimum, but a clip shorter
# than this carries too little speaker information to clone.
MIN_PROMPT_DURATION_S = 1.0

# The engine truncates prompts to 30 s (``truncate=True``).
MAX_PROMPT_DURATION_S = 30.0

# Sanity valve for the request payload (~5 min of 24 kHz audio).  Not a model
# limitation -- the reference clip is truncated to 30 s internally.
MAX_AUDIO_B64_LEN = 10_000_000

# Conditioned voice states kept for reuse (LRU).  Encoding a reference clip
# is the expensive per-request step; repeat requests with the same clip or
# preset voice skip it.
VOICE_STATE_CACHE_SIZE = 8

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

# Built from the engine's own preset table so the request schema and the
# /capabilities enum share one source.  (Some presets belong to a specific
# language -- e.g. "daan" is Dutch -- and the engine rejects mismatches.)
PresetVoice = Literal[tuple(_ORIGINS_OF_PREDEFINED_VOICES)]


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
            "Reference voice sample (roughly 5--15 s of clean speech works "
            "well) as a base64 string, cloned zero-shot.  Any container "
            "soundfile can decode (WAV, MP3, OGG, FLAC, ...).  The engine "
            "uses at most the first 30 s.  Omit to use `voice` or the "
            "default preset voice.  Mutually exclusive with `voice`."
        ),
    )
    language: str | None = Field(
        DEFAULT_LANGUAGE,
        description=(
            "Two-letter language code, e.g. 'en'.  Accepted for API "
            "consistency but not forwarded: a Pocket TTS server speaks the "
            f"single language of its loaded model ('{LANGUAGE_CODE}' here), "
            "whatever is requested.  Omitted or empty defaults to 'en'."
        ),
    )
    seed: int | None = Field(
        None,
        ge=SEED_MIN,
        le=SEED_MAX,
        description=(
            "Random seed for reproducibility.  If omitted, a random seed "
            f"in [{SEED_MIN}, {SEED_MAX}] is chosen and echoed in the response.  "
            "The voice itself is fixed by the clip or preset; the seed only "
            "varies delivery, so the same seed gives the same audio."
        ),
    )

    # --- engine-specific ---------------------------------------------------
    voice: PresetVoice | None = Field(
        None,
        description=(
            "Built-in preset voice name.  Omit (together with "
            "`audio_base64`) to use the loaded language's default voice.  "
            "Presets are tied to the language of their recordings; the "
            "engine rejects a preset that does not match the loaded model.  "
            "Mutually exclusive with `audio_base64`."
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
        # Not forwarded to the engine (see the field description), but a
        # malformed code like 'x?' is still a client error.
        return validate_language_code(v)

    @model_validator(mode="after")
    def _voice_sources_exclusive(self) -> "SynthesisRequest":
        # Two voice sources is ambiguous: reject rather than silently picking
        # one and handing back a voice the client did not expect.
        if self.audio_base64 is not None and self.voice is not None:
            raise ValueError("audio_base64 and voice are mutually exclusive; send one")
        return self


class SynthesisResponse(CoreSynthesisResponse):
    """The synthesis result (core fields from tts_engine_common, plus fid)."""

    fid: str = Field(..., description="Request ID (internal).")


class HealthResponse(BaseModel):
    """Health / readiness check."""

    status: Literal["ok"] = "ok"
    serverType: Literal["PocketTTS"] = "PocketTTS"
    model: str = MODEL_LABEL
    device: str = DEVICE


# ---------------------------------------------------------------------------
# Capabilities (derived from SynthesisRequest — single source of truth)
# ---------------------------------------------------------------------------

CAPABILITIES = build_capabilities(
    SynthesisRequest,
    engine="pocket-tts",
    model=MODEL_LABEL,
    device=DEVICE,
    sample_rate=SAMPLE_RATE,
    watermarked=False,
    endpoint="/synthesize",
    reference_audio={
        "required": False,
        "formats": ["wav", "mp3", "ogg", "flac"],
        "min_duration_s": MIN_PROMPT_DURATION_S,
        "note": (
            "Optional: omit to use a built-in preset (`voice`, or the loaded "
            "language's default).  Zero-shot cloning needs the gated model "
            "weights; clips are truncated to 30 s.  The engine does not use "
            "a transcript of the clip, so there is no `reference_text`."
        ),
    },
    languages=[LANGUAGE_CODE],  # one model = one language per server instance
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
    _voice_states.clear()
    logger.info("Model unloaded.")


app = FastAPI(
    title="Pocket TTS API",
    description=(
        "REST API around Kyutai Pocket TTS, a small CPU-first model.  Send "
        "text, optionally with a reference audio sample or a preset voice "
        "name, and get back speech.  Synthesis requests are serialized "
        "(single shared model).  Machine-readable parameter metadata at "
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
# Runtime — thin wrapper around the Pocket TTS model
# ---------------------------------------------------------------------------


@dataclass
class PocketRuntime:
    """Holds the loaded model and its metadata for the lifetime of the server."""

    model: TTSModel
    sample_rate: int
    device: str
    has_voice_cloning: bool


_runtime: PocketRuntime | None = None

# ``generate_audio`` is documented as not thread-safe, and the sampler draws
# from the process-global torch RNG (seeded per request).  Serialize synthesis.
_synthesis_lock = threading.Lock()

# LRU of conditioned voice states, keyed by clip hash / preset name.  Only
# touched under ``_synthesis_lock``.  States are reused with ``copy_state=True``
# so a request never mutates a cached entry.
_voice_states: OrderedDict[str, object] = OrderedDict()


def _get_runtime() -> PocketRuntime:
    """Return the global runtime, loading the model once on first call."""
    global _runtime
    if _runtime is None:
        logger.info(
            "Loading {} on cpu (quantize={}; first run downloads from HuggingFace) ...",
            MODEL_LABEL,
            QUANTIZE,
        )
        model = TTSModel.load_model(language=MODEL_NAME, quantize=QUANTIZE)
        _runtime = PocketRuntime(
            model=model,
            sample_rate=model.sample_rate,
            device=DEVICE,
            has_voice_cloning=getattr(model, "has_voice_cloning", True),
        )
        logger.info(
            "Model loaded successfully. Sampling rate: {} Hz, voice cloning: {}",
            _runtime.sample_rate,
            _runtime.has_voice_cloning,
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
    <head><title>Pocket TTS REST API</title></head>
    <body>
        <h1>Pocket TTS REST API</h1>
        <p>Model: <code>{MODEL_LABEL}</code> on <code>{DEVICE}</code>.
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
    summary="Synthesize speech from text (+ optional reference audio or preset voice)",
)
def synthesize(req: SynthesisRequest) -> SynthesisResponse:
    """
    Synthesize audio using the provided text and, optionally, a reference
    audio sample or a preset voice.

    The full parameter list is documented at GET /capabilities; the request
    schema mirrors it exactly (same model, no drift).
    """
    runtime = _get_runtime()

    seed = req.seed if req.seed is not None else random.randint(SEED_MIN, SEED_MAX)

    logger.info(
        "Synthesizing: seed={}, text_len={}, voice={}, cloning={}, lang={}",
        seed,
        len(req.text),
        req.voice,
        req.audio_base64 is not None,
        req.language,
    )

    # Decode and sanity-check the reference audio before touching the model.
    raw_audio: bytes | None = None
    if req.audio_base64 is not None:
        try:
            raw_audio = decode_base64(req.audio_base64)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")
        _check_reference_audio(raw_audio)
        if not runtime.has_voice_cloning:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Voice cloning is unavailable: the gated model weights are "
                    "not installed.  Accept the terms at "
                    "https://huggingface.co/kyutai/pocket-tts and run `hf auth "
                    "login`, or use a preset `voice`."
                ),
            )

    try:
        t0 = time.perf_counter()

        with _synthesis_lock:
            seed_everything(seed)
            voice_state = _voice_state_for(runtime, raw_audio, req.voice)
            wav = runtime.model.generate_audio(voice_state, req.text, copy_state=True)

        time_used = time.perf_counter() - t0

        # generate_audio() returns mono float audio; flatten any [1, N] shape.
        audio_array = wav.detach().cpu().numpy().reshape(-1)
        sample_rate = runtime.sample_rate

        rtf = compute_rtf(time_used, len(audio_array), sample_rate)

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

_TEMP_AUDIO_DIR = temp_audio_dir("pocket_tts_rest_api")


def seed_everything(seed: int) -> None:
    """Set the random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _voice_state_for(runtime: PocketRuntime, raw_audio: bytes | None, voice: str | None):
    """Conditioned voice state for this request, from cache when possible.

    Must be called with ``_synthesis_lock`` held (the cache and the engine's
    state builder are not thread-safe).
    """
    if raw_audio is not None:
        key = "clip:" + hashlib.sha256(raw_audio).hexdigest()
    else:
        key = "voice:" + (voice or get_default_voice_for_language(LANGUAGE_NAME))

    cached = _voice_states.get(key)
    if cached is not None:
        _voice_states.move_to_end(key)
        return cached

    if raw_audio is not None:
        # The engine wants a file path for a clip; one-shot staging, since the
        # conditioned state (not the file) is what we cache.
        prompt_path = write_temp_audio(raw_audio, _TEMP_AUDIO_DIR)
        try:
            state = runtime.model.get_state_for_audio_prompt(Path(prompt_path), truncate=True)
        finally:
            cleanup_temp(prompt_path)
    else:
        state = runtime.model.get_state_for_audio_prompt(key.removeprefix("voice:"))

    _voice_states[key] = state
    while len(_voice_states) > VOICE_STATE_CACHE_SIZE:
        _voice_states.popitem(last=False)
    return state


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
# Main (for running directly: python server_pocketTTS.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("POCKET_TTS_HOST", "0.0.0.0")
    port = int(os.getenv("POCKET_TTS_PORT", "7500"))
    logger.info("Starting Pocket TTS REST API server on {}:{}", host, port)
    # Pass the app object directly instead of a module path string,
    # so this works regardless of how the file is invoked.
    uvicorn.run(app, host=host, port=port, log_level="info")
