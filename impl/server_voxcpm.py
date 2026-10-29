"""
FastAPI REST server for VoxCPM2 voice cloning and voice design
(github.com/OpenBMB/VoxCPM).

VoxCPM2 is a 2 billion parameter tokenizer-free TTS model that runs on CUDA,
MPS, or CPU.  Clients send text plus an optional reference audio sample (base64);
the server returns the generated audio as base64-encoded 48 kHz WAV
(VoxCPM2's AudioVAE V2 outputs 48 kHz directly — unlike the 24 kHz of most
tts-serve engines).

Three synthesis modes, all via the same `generate()` call:

1. **Voice design** (no reference audio) -- create a voice from a
   natural-language description placed in parentheses at the start of the
   text, e.g. `"(A young woman, gentle voice)Hello there!"`.
2. **Controllable cloning** (reference audio only) -- clone the timbre of
   the reference clip; style control still via a `(control)` prefix in text.
3. **Ultimate cloning** (reference audio + exact transcript) -- reproduce
   every vocal nuance; pass the same reference clip to both the prompt audio
   and its transcript for maximum similarity.

`seed` is meaningful: VoxCPM seeds PyTorch's RNG before each generation, so
the same seed and inputs are reproducible.  On CPU the output is
bit-identical; on CUDA some GPU kernels are non-deterministic, so repeated
runs can differ slightly (similar to LuxTTS).  Do not promise bit-exact
reproducibility for GPU generations.

Model weights are downloaded from HuggingFace (`openbmb/VoxCPM2`) on first
start.  Set HF_TOKEN in the environment if your checkpoint needs it.

Configuration (environment variables):
    VOXCPM_MODEL       The model source.  Pass the default HuggingFace id
                       (openbmb/VoxCPM2) to have it downloaded, or a local
                       path to an already-extracted model directory.
                       Default: openbmb/VoxCPM2
    VOXCPM_DEVICE      Device to load the model on: 'auto' (engine picks --
                        CUDA preferred, then MPS, then CPU), 'cuda',
                        'cuda:<index>', 'mps', or 'cpu'.  Default: cuda.  The
                        grammar is checked at import time; an explicit device
                        that is unavailable fails at model load.
    VOXCPM_MPS_DTYPE   Override dtype for MPS only.  The engine forces
                       float32 on MPS by default because bfloat16/float16
                       cause numerical drift that breaks the diffusion loop.
                       One of: float32, bfloat16, float16.  Leave unset in
                       production; only override to test future engine
                       improvements.
    VOXCPM_HOST        Bind host for `python server_voxcpm.py`.
                       Default: 0.0.0.0
    VOXCPM_PORT        Bind port for `python server_voxcpm.py`.
                       Default: 7500

Extra dependencies beyond the VoxCPM repository:
    pip install voxcpm fastapi uvicorn loguru soundfile
    pip install ../tts-engine-common   # in-repo copy; or: pip install -e ../tts-engine-common

Usage:
    python server_voxcpm.py
    # or: uvicorn server_voxcpm:app --host 0.0.0.0 --port 7500
"""

from __future__ import annotations

import base64
import io
import inspect
import os
import random
import re
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

from voxcpm import VoxCPM

# The `seed` parameter was added to ``VoxCPM._generate`` in a later VoxCPM
# release.  Older installs (e.g. 2.0.0 / 1.5.0 on PyPI) raise
# "unexpected keyword argument 'seed'" if we pass it, so introspect the
# installed engine and only forward ``seed`` when the signature supports it.
try:
    _GENERATE_ACCEPTS_SEED = (
        "seed" in inspect.signature(VoxCPM._generate).parameters
    )
except AttributeError:
    # Stub (test machines) or an engine version that doesn't expose _generate —
    # seed is applied via torch.manual_seed in seed_everything() anyway.
    _GENERATE_ACCEPTS_SEED = False

from tts_engine_common import (
    DEFAULT_LANGUAGE,
    CoreHealthResponse,
    RELEASE_VERSION,
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

# MPS dtype override: the engine reads VOXCPM_MPS_DTYPE internally via
# pick_runtime_dtype() -- we validate it here only to fail fast at startup
# rather than partway through a model download.
_MPS_DTYPE = os.getenv("VOXCPM_MPS_DTYPE", "").strip().lower()
if _MPS_DTYPE:
    _VALID_MPS_DTYPES = ("float32", "bfloat16", "float16")
    if _MPS_DTYPE not in _VALID_MPS_DTYPES:
        raise ValueError(
            f"VOXCPM_MPS_DTYPE must be one of {_VALID_MPS_DTYPES}, got {_MPS_DTYPE!r}"
        )


# The engine's device grammar (see voxcpm's resolve_runtime_device):
# 'auto' (or unset) -> automatic selection, otherwise an explicit device,
# optionally indexed ('cuda:1').  We check the *grammar* here, at import
# time, so a typo fails before a model download starts; *availability* of an
# explicit device is the engine's job at load time (it raises a clear error).
_DEVICE_RE = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")


def _validate_config() -> None:
    """Fail fast on bad configuration instead of partway through a model download."""
    if not _DEVICE_RE.match(DEVICE):
        raise ValueError(
            "VOXCPM_DEVICE must be 'auto', 'cpu', 'mps', 'cuda', or "
            f"'cuda:<index>', got {DEVICE!r}"
        )


_validate_config()

# VoxCPM2's AudioVAE V2 outputs 48 kHz audio (asymmetric encode/decode with
# built-in super-resolution).  Confirmed against the engine's own
# `audio_vae.out_sample_rate`, not assumed.
SAMPLE_RATE = 48000

SEED_MIN = 1
SEED_MAX = 1000

# The reference clip minimum used by most tts-serve engines; VoxCPM loads
# prompt audio at 16 kHz via librosa and conditions on its latent patches --
# anything shorter than 2 s yields unstable cloning.
MIN_PROMPT_DURATION_S = 2.0

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
        description="Text to synthesize, e.g. 'Hello there'.  Control instructions "
                    "go in parentheses at the start, e.g. "
                    "'(A young woman, gentle voice)Welcome!'.",
    )
    audio_base64: str | None = Field(
        None,
        min_length=1,
        max_length=MAX_AUDIO_B64_LEN,
        description=(
            "Reference voice sample (roughly 10 s works well) as a base64 "
            "string, for controllable or ultimate cloning.  Omit entirely for "
            "voice design mode (a `(control instruction)` prefix in `text` "
            "steers the generated voice).  Any container soundfile can decode "
            "(WAV, MP3, OGG, FLAC, ...)."
        ),
    )
    reference_text: str | None = Field(
        None,
        description=(
            "Exact transcript of the reference clip.  Only accepted together "
            "with `audio_base64` (ultimate cloning / continuation mode); "
            "providing it without `audio_base64` is rejected (422) -- a "
            "transcript with no clip is a client error, not a mode switch."
        ),
    )
    language: str | None = Field(
        DEFAULT_LANGUAGE,
        description=(
            "Two-letter language code, e.g. 'en' or 'zh'.  Accepted for API "
            "consistency but not forwarded -- the engine has no language "
            "parameter; its text encoder auto-detects the language from the "
            "input text (30 languages supported).  "
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
    cfg_value: float = Field(
        2.0,
        ge=0.1,
        le=10.0,
        description=(
            "Classifier-free guidance scale (README: higher is more expressive "
            "but can sound unstable; the CLI enforces 0.1--10.0)."
        ),
    )
    inference_timesteps: int = Field(
        10,
        ge=1,
        le=100,
        description=(
            "Flow-matching sampling steps (README: recommended 4--30; higher "
            "is slower with diminishing quality)."
        ),
    )
    retry_badcase: bool = Field(
        True,
        description=(
            "Retry when the audio-to-text length ratio is out of range "
            "(README default).  Disabled automatically in streaming mode."
        ),
    )
    normalize: bool = Field(
        False,
        description=(
            "Run text normalization (wetext) before generation.  Disabled by "
            "default; enable for cleaner output with numbers/abbreviations."
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
        # docs/02: the API speaks two-letter codes.  The engine has no
        # language parameter and auto-detects from text, so there is no
        # 'auto' sentinel either -- accept any well-formed code for
        # API consistency (LuxTTS no-support case).
        return validate_language_code(v)

    @model_validator(mode="after")
    def _reference_text_requires_audio(self) -> "SynthesisRequest":
        # A transcript with no clip is a client bug, not a mode switch:
        # silently degrading to voice design would hand back audio the client
        # did not ask for and mask the bug.  Reject at the boundary (422)
        # rather than letting the engine's pairing check raise (500).
        if self.reference_text is not None and self.audio_base64 is None:
            raise ValueError(
                "reference_text requires audio_base64 (the clip it transcribes)"
            )
        return self


class SynthesisResponse(CoreSynthesisResponse):
    """The synthesis result (core fields from tts_engine_common, plus fid)."""

    fid: str = Field(..., description="Request ID (internal).")


class HealthResponse(CoreHealthResponse):
    """The /health core from tts_engine_common, with this engine's serverType.

    model/device pick up the import-time config; version carries the shared
    RELEASE_VERSION constant — overriding it here is how a server drifts.
    """

    serverType: Literal["VoxCPM"] = "VoxCPM"
    model: str = MODEL_NAME_OR_PATH
    device: str = DEVICE


# ---------------------------------------------------------------------------
# Capabilities (derived from SynthesisRequest -- single source of truth)
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
            "Optional: omit for voice design mode (a `(control instruction)` "
            "prefix in `text` steers the generated voice).  For controllable "
            "cloning, provide a roughly 3--10 s reference clip (the engine "
            "loads it at 16 kHz via librosa and conditions on its latent "
            "patches).  For ultimate cloning, also pass `reference_text` "
            "(the clip's exact transcript): the model treats the clip as a "
            "spoken prefix and continues from it, reproducing every vocal "
            "nuance."
        ),
    },
    languages=None,  # no fixed list; two-letter codes (docs/02), not forwarded
    overrides={
        "cfg_value": {"step": 0.1},
        "inference_timesteps": {"step": 1},
        "retry_badcase": {"advanced": True},
        "normalize": {"advanced": True},
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
    title="VoxCPM2 TTS API",
    description=(
        "REST API around VoxCPM2.  Send text plus an optional reference audio "
        "sample and get back cloned or designed speech.  Three modes: voice "
        "design (no reference), controllable cloning (reference only), and "
        "ultimate cloning (reference + transcript).  Synthesis requests are "
        "serialized (single shared model).  Machine-readable parameter "
        "metadata at GET /capabilities."
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
# Runtime -- thin wrapper around the VoxCPM model
# ---------------------------------------------------------------------------


@dataclass
class VoxCPMRuntime:
    """Holds the loaded model and its metadata for the lifetime of the server."""

    model: VoxCPM
    sample_rate: int
    device: str


_runtime: VoxCPMRuntime | None = None

# The engine mutates shared state per call: KV caches are filled in place,
# torch.compiled functions (forward_step, feat_encoder, feat_decoder) are
# not re-entrant, @torch.inference_mode() contexts are thread-affine, and
# `last_successful_seed` is a mutable instance attribute.  Serialize
# synthesis; single-device throughput is the bottleneck anyway.
_synthesis_lock = threading.Lock()


def _get_runtime() -> VoxCPMRuntime:
    """Return the global runtime, loading the model once on first call."""
    global _runtime
    if _runtime is None:
        logger.info(
            "Loading VoxCPM model '{}' on device '{}' (first run downloads "
            "from HuggingFace) ...",
            MODEL_NAME_OR_PATH,
            DEVICE,
        )
        model = VoxCPM.from_pretrained(
            MODEL_NAME_OR_PATH,
            # The ZipEnhancer denoiser is deliberately not exposed by this
            # server: it needs a separate ModelScope download and only helps
            # very noisy reference clips.  Pre-filtering noisy clips is the
            # client's job.
            load_denoiser=False,
            optimize=True,        # torch.compile on CUDA (default True in engine)
            device=DEVICE,
        )
        # model.tts_model.sample_rate is the engine's own output rate (48 kHz
        # for VoxCPM2 / AudioVAE V2); echo that rather than our constant so a
        # future variant would stay self-consistent.
        _runtime = VoxCPMRuntime(
            model=model,
            sample_rate=int(model.tts_model.sample_rate),
            device=str(model.tts_model.device),
        )
        logger.info(
            "Model loaded successfully. Sampling rate: {} Hz, device: {}",
            _runtime.sample_rate,
            _runtime.device,
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
    <head><title>VoxCPM2 TTS API</title></head>
    <body>
        <h1>VoxCPM2 Voice Cloning &amp; Design REST API</h1>
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
    summary="Synthesize speech (voice design, cloning, or continuation)",
)
def synthesize(req: SynthesisRequest) -> SynthesisResponse:
    """
    Synthesize audio using the provided text and optional reference audio.

    The full parameter list is documented at GET /capabilities; the request
    schema mirrors it exactly (same model, no drift).
    """
    runtime = _get_runtime()

    # Resolve randomised seed.
    seed = req.seed if req.seed is not None else random.randint(SEED_MIN, SEED_MAX)

    logger.info(
        "Synthesizing: seed={}, text_len={}, mode={}, cfg={:.1f}, steps={}, "
        "retry_badcase={}, normalize={}, lang={} (not forwarded)",
        seed,
        len(req.text),
        "ultimate-cloning" if (req.audio_base64 and req.reference_text)
        else "cloning" if req.audio_base64
        else "voice-design",
        req.cfg_value,
        req.inference_timesteps,
        req.retry_badcase,
        req.normalize,
        req.language,
    )

    # Map request fields onto the engine's generate() signature.
    prompt_audio_path: str | None = None
    reference_audio_path: str | None = None

    if req.audio_base64 is not None:
        # Decode and sanity-check the reference audio before touching the model.
        try:
            raw_audio = decode_base64(req.audio_base64)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")

        _check_reference_audio(raw_audio)

        # The engine loads prompt/reference audio from a file path (librosa).
        # One-shot engine -- no path-keyed cache -- so a UUID temp file
        # cleaned up per request is fine (no content-hashing needed).
        audio_path = write_temp_audio(raw_audio, _TEMP_AUDIO_DIR)

        if req.reference_text is not None:
            # Ultimate cloning: same clip for prompt audio + transcript.
            prompt_audio_path = audio_path
            reference_audio_path = audio_path
        else:
            # Controllable cloning: reference audio only.
            reference_audio_path = audio_path

    try:
        # Time the actual synthesis call.
        t0 = time.perf_counter()

        with _synthesis_lock:
            seed_everything(seed)
            # `_GENERATE_ACCEPTS_SEED` is decided at import time by introspecting
            # the installed engine, so this stays compatible with both older
            # VoxCPM releases (no `seed` param) and newer ones.
            generate_kwargs = {
                "text": req.text,
                "prompt_wav_path": prompt_audio_path,
                "prompt_text": req.reference_text,
                "reference_wav_path": reference_audio_path,
                "cfg_value": req.cfg_value,
                "inference_timesteps": req.inference_timesteps,
                "normalize": req.normalize,
                "retry_badcase": req.retry_badcase,
            }
            if _GENERATE_ACCEPTS_SEED:
                generate_kwargs["seed"] = seed
            wav = runtime.model.generate(**generate_kwargs)
            # Echo the seed actually used: retry_badcase can increment it
            # internally across retries, so the engine's tracked value is the
            # authoritative one (exactly how app.py reads it).  Read it under
            # the same lock -- a concurrent request's generate() would
            # overwrite it in the window between release and read otherwise.
            actual_seed = getattr(
                runtime.model.tts_model, "last_successful_seed", seed
            )

        time_used = time.perf_counter() - t0

        # generate() returns a 1-D float32 numpy array on CPU.
        audio_array = wav.reshape(-1)
        sample_rate = runtime.sample_rate

        rtf = compute_rtf(time_used, len(audio_array), sample_rate)

        # Encode the output WAV to base64.
        audio_bytes = _numpy_to_wav_bytes(audio_array, sample_rate)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        audio_duration = len(audio_array) / sample_rate if sample_rate else 0.0
        logger.info(
            "Synthesis complete: {:.1f} s wall-clock, {:.1f} s audio, RTF={}, "
            "seed={}",
            time_used,
            audio_duration,
            f"{rtf:.3f}" if rtf is not None else "n/a",
            actual_seed,
        )

        return SynthesisResponse(
            audio_base64=audio_b64,
            sample_rate=sample_rate,
            seed=actual_seed,
            fid=str(uuid.uuid4()),
            time_used=time_used,
            rtf=rtf,
        )

    except Exception as exc:
        logger.error("Synthesis failed: {}", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # Clean up the temporary reference audio file(s).
        if prompt_audio_path is not None:
            cleanup_temp(prompt_audio_path)
        if reference_audio_path is not None and reference_audio_path != prompt_audio_path:
            cleanup_temp(reference_audio_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMP_AUDIO_DIR = temp_audio_dir("voxcpm_rest_api")


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
