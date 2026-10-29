"""
FastAPI REST server for Ming-3.0-tts voice cloning (7.9B MoE checkpoint,
engine name "tiny"; github.com/inclusionAI/Ming-3.0-tts).

Ming-3.0-tts ships two checkpoints — a 0.5b dense model and this one,
7.9B-total / 1.3B-active MoE on Ling V3 — with separate Python stacks
(conflicting torch / x-transformers pins), so each has its own server:
the dense model is ``server_ming3TTS.py``.  Yes, the engine calls the 7.9B
MoE "tiny"; the name is the checkpoint's, not our editorial choice.

ICL (in-context) voice cloning only, via the engine's ``seed`` task
(Chinese / English) or ``multilingual`` task (German / Spanish / French /
Korean / Japanese — the task is selected from ``language``).  Both tasks
condition on the reference clip **and its exact transcript**: a missing
``reference_text`` is a 422, not a silent fallback (the engine has no
transcript-free clone mode for these tasks; its ``speaker_only``
conditioning only exists on the control tasks, which are deliberately not
exposed).  The engine's other modes (voice design, built-in IP voices,
podcast, emotion / speed / dialect control) are likewise out of scope:
this server is a voice-cloning server.

Inline paralinguistic tags work verbatim in ``text`` on the ``seed`` and
``paralinguistic`` prompt layouts: [Breathing], [Cough], [Sigh],
[Laughter], [Surprise-ah], [Surprise-oh], [Dissatisfaction-hnn], [Uhm],
[Confirmation-en], [Question-ei], [Question-ah], [Question-en],
[Question-oh], [Question-huh].

The engine loads the reference clip per request (no path-keyed prompt
cache), so reference audio is staged in a UUID-named temp file and deleted
when the request finishes.

``seed`` is meaningful and is passed straight to the engine, which seeds
the whole pipeline itself at the start of each synthesis; the response
echoes the seed that was used.  On CUDA, reproducibility is near-exact,
not bit-exact (non-deterministic kernels).

44.1 kHz output: the engine *enforces* sample_rate=44100 as part of its
release-checkpoint contract (it refuses to load anything else), so the
capabilities constant cannot drift on an official checkpoint.

The MoE runtime is heavily CUDA-oriented (its own pins: Python 3.10.15,
PyTorch 2.7.1+cu118, fla-core).  The vendor recommends an H100/H800-class
GPU; ~16 GiB of weights in bf16.  Use a dedicated virtualenv — its
requirements-tiny.txt conflicts with the 0.5b requirements.txt.

Configuration (environment variables):
    MING3_TTS_MOE_MODEL    Local checkpoint directory or HuggingFace id.
                           Default: inclusionAI/Ming-3.0-tts-tiny
                           (the "tiny" checkpoint is the 7.9B MoE.)
    MING3_TTS_MOE_DEVICE   Torch device to load the model on: 'cuda',
                           'cuda:N', 'cpu', or 'mps'.  Only CUDA is tested
                           (the engine's runtime env is CUDA-oriented).
                           Default: cuda
    MING3_TTS_MOE_HOST     Bind host for `python server_ming3TTS_moe.py`.
                           Default: 0.0.0.0
    MING3_TTS_MOE_PORT     Bind port for `python server_ming3TTS_moe.py`.
                           Default: 7500

Extra dependencies beyond the Ming-3.0-tts repository:
    git clone https://github.com/inclusionAI/Ming-3.0-tts.git
    cd Ming-3.0-tts
    conda create -n ming-tiny python=3.10.15 -y && conda activate ming-tiny
    pip install -r requirements-tiny.txt
    pip install fastapi uvicorn loguru soundfile
    pip install ../tts-engine-common # in-repo copy; or: pip install -e ../tts-engine-common

The server must be able to import the engine's ``ming_tiny`` package.
Running `python /path/to/server_ming3TTS_moe.py` puts the *script's*
directory on sys.path, not your working directory, so the Ming-3.0-tts
repo root must be added explicitly:

    cd Ming-3.0-tts
    PYTHONPATH=. MING3_TTS_MOE_MODEL=/path/to/Ming-3.0-tts-tiny \
        python ../tts-serve/impl/server_ming3TTS_moe.py

Usage:
    python server_ming3TTS_moe.py
    # or: uvicorn server_ming3TTS_moe:app --host 0.0.0.0 --port 7500
"""

from __future__ import annotations

import base64
import io
import math
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
from pydantic import BaseModel, ConfigDict, Field, field_validator

# The engine ships as a plain git repo (no PyPI package, no setup.py), so
# ``ming_tiny`` (and its transitive fla-core / transformers pins) is
# importable only when the repo root is on PYTHONPATH.  Give the most
# common deployment failure mode a message people can actually act on.
try:
    from ming_tiny import MingTinyTTS
    from ming_tiny.task_protocol import (
        FORMAL_CFG,
        FORMAL_FLOW_STEPS,
        FORMAL_SIGMA,
        FORMAL_TEMPERATURE,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"The Ming-3.0-tts engine module {exc.name!r} is not importable.  "
        "The Ming-3.0-tts repo root must be on PYTHONPATH (requirements-"
        "tiny.txt installed in a Python 3.10 env) -- e.g. from the "
        "Ming-3.0-tts repo root:  PYTHONPATH=. python /path/to/tts-serve/"
        "impl/server_ming3TTS_moe.py"
    ) from exc

from tts_engine_common import (
    DEFAULT_LANGUAGE,
    CoreHealthResponse,
    RELEASE_VERSION,
    CoreSynthesisResponse,
    build_capabilities,
    capabilities_endpoint,
    canonical_language_codes,
    canonical_language_literal,
    cleanup_temp,
    compute_rtf,
    decode_base64,
    normalize_language,
    temp_audio_dir,
    write_temp_audio,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME_OR_PATH = os.getenv("MING3_TTS_MOE_MODEL", "inclusionAI/Ming-3.0-tts-tiny")
DEVICE = os.getenv("MING3_TTS_MOE_DEVICE", "cuda")

# The engine enforces 44100 as part of its release-contract validation
# (ming_tiny.task_protocol.validate_release_engine), so this constant is
# the checkpoint's own number, not an estimate.
SAMPLE_RATE = 44100

SEED_MIN = 1
SEED_MAX = 1000

# Heuristic lower bound: the CampPlus speaker embedding needs enough speech
# for usable frames, and the AudioVAE prompt continuation needs enough
# audio to seed the first decode step.
MIN_PROMPT_DURATION_S = 2.0

# Sanity valve for the request payload (~85 s of 44.1 kHz 16-bit mono).
MAX_AUDIO_B64_LEN = 10_000_000

# Flow-matching sampler defaults, imported from the engine itself (its
# "formal" evaluation settings — the numbers the published benchmarks ran
# with).  No magic numbers on this server.
DEFAULT_CFG_SCALE = FORMAL_CFG
DEFAULT_FLOW_STEPS = FORMAL_FLOW_STEPS
DEFAULT_SIGMA = FORMAL_SIGMA
DEFAULT_TEMPERATURE = FORMAL_TEMPERATURE

# The engine splits cloning by task, and each task has a fixed allowed
# language set (ming_tiny.task_protocol.TASK_SPECS): 'seed' handles
# Chinese/English, 'multilingual' the five languages added in Ming-3.0.
# The API speaks two-letter codes (docs/02); the server selects the task
# from the code and passes the code as ``target_language`` (the engine's
# normalize_target_language accepts codes).  No 'auto': both tasks require
# an explicit target language, so the engine offers no auto-detection
# sentinel to map onto.
SEED_LANGUAGE_CODES = ("zh", "en")
MULTILINGUAL_LANGUAGE_CODES = ("de", "es", "fr", "ja", "ko")
LANGUAGE_CODES = SEED_LANGUAGE_CODES + MULTILINGUAL_LANGUAGE_CODES

_DEVICE_PATTERN = re.compile(r"^(cuda(:\d+)?|cpu|mps)$")


def _validate_config() -> None:
    """Fail fast on bad configuration instead of partway through a model download."""
    # Grammar check only (VoxCPM-style): the engine checks availability at
    # load time; only CUDA is tested.
    if not _DEVICE_PATTERN.fullmatch(DEVICE):
        raise ValueError(
            "MING3_TTS_MOE_DEVICE must be 'cuda', 'cuda:N', 'cpu', or 'mps', "
            f"got {DEVICE!r}"
        )


_validate_config()


def _task_for_language(code: str) -> str:
    """Map a language code to the engine task that is allowed to speak it."""
    return "seed" if code in SEED_LANGUAGE_CODES else "multilingual"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

# Dynamic Literal over the codes the two cloning tasks cover, so the
# request schema and the /capabilities enum share one source.  Built via the
# shared helper to pin canonical sorted order (LANGUAGE_CODES keeps the
# task-grouped order for _task_for_language; the collision this dodges is
# documented in tts_engine_common.canonical_language_literal).
Language = canonical_language_literal(LANGUAGE_CODES)


class SynthesisRequest(BaseModel):
    """A single synthesis request. Unknown fields are rejected (422)."""

    model_config = ConfigDict(extra="forbid")

    # --- core vocabulary (tts_engine_common.CORE_FIELDS) -------------------
    text: str = Field(
        ...,
        min_length=1,
        description=(
            "Text to synthesize, e.g. 'Hello there'.  Inline paralinguistic "
            "tags are supported (Chinese or English): [Breathing], [Cough], "
            "[Sigh], [Laughter], [Surprise-ah], [Surprise-oh], "
            "[Dissatisfaction-hnn], [Uhm], [Confirmation-en], [Question-ei], "
            "[Question-ah], [Question-en], [Question-oh], [Question-huh]."
        ),
    )
    audio_base64: str = Field(
        ...,
        min_length=1,
        max_length=MAX_AUDIO_B64_LEN,
        description=(
            "Reference voice sample as a base64 string.  Any container "
            "soundfile can decode (WAV, MP3, OGG, FLAC, ...); any input "
            "sample rate (the engine resamples).  ~3-10 s of clean speech "
            "is the sweet spot."
        ),
    )
    reference_text: str = Field(
        ...,
        min_length=1,
        description=(
            "Exact transcript of the reference audio.  Required: cloning "
            "is in-context, conditioning on the clip and its transcript "
            "together, so the words must match the clip exactly (a wrong "
            "transcript degrades the clone)."
        ),
    )
    language: Language | None = Field(
        DEFAULT_LANGUAGE,
        description=(
            "Two-letter language code (supported: "
            f"{', '.join(sorted(LANGUAGE_CODES))}); selects the engine "
            "cloning task (zh/en: seed, de/es/fr/ja/ko: multilingual).  "
            "Omitted or empty defaults to 'en'.  Set it to the language "
            "of *text*."
        ),
    )
    seed: int | None = Field(
        None,
        ge=SEED_MIN,
        le=SEED_MAX,
        description=(
            "Random seed for reproducibility.  If omitted, a random seed "
            f"in [{SEED_MIN}, {SEED_MAX}] is chosen and echoed in the "
            "response."
        ),
    )

    # --- engine-specific tuning (defaults mirror the engine's formal ones) --
    cfg_scale: float = Field(
        DEFAULT_CFG_SCALE,
        gt=0.0,
        description=(
            "Classifier-free guidance scale.  Engine formal default "
            f"{DEFAULT_CFG_SCALE:.1f}.  Must be finite and > 0."
        ),
    )
    flow_steps: int = Field(
        DEFAULT_FLOW_STEPS,
        ge=1,
        description=(
            "Flow-matching solver steps.  Engine formal default "
            f"{DEFAULT_FLOW_STEPS}; more steps cost time, not quality "
            "past that."
        ),
    )
    sigma: float = Field(
        DEFAULT_SIGMA,
        ge=0.0,
        description=(
            "Flow-matching noise scale.  Engine formal default "
            f"{DEFAULT_SIGMA:.2f}.  Must be finite and >= 0."
        ),
    )
    temperature: float = Field(
        DEFAULT_TEMPERATURE,
        ge=0.0,
        description=(
            "Sampling temperature.  Engine formal default 0.0 (greedy).  "
            "Must be finite and >= 0."
        ),
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
        # A blank transcript would clone against an empty prompt; reject
        # it loudly at the boundary (the field is required, so null is
        # already a 422).
        if not v.strip():
            raise ValueError("reference_text must contain non-whitespace characters")
        return v

    @field_validator("cfg_scale", "sigma", "temperature")
    @classmethod
    def _check_finite(cls, v: float) -> float:
        # gt/ge bounds reject wrong signs but pass +inf; keep the engine
        # from ever seeing a non-finite sampler parameter.
        if not math.isfinite(v):
            raise ValueError("must be a finite number")
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
    """The synthesis result (core fields from tts_engine_common, plus extras)."""

    fid: str = Field(..., description="Request ID (internal).")
    stopped: bool = Field(
        ...,
        description=(
            "Whether the engine's stop detector fired before the decode "
            "budget ran out.  False means the output may be truncated: "
            "try a different seed or shorter text."
        ),
    )


class HealthResponse(CoreHealthResponse):
    """The /health core from tts_engine_common, with this engine's serverType.

    model/device pick up the import-time config; version carries the shared
    RELEASE_VERSION constant — overriding it here is how a server drifts.
    """

    serverType: Literal["ming3-tts-moe"] = "ming3-tts-moe"
    model: str = MODEL_NAME_OR_PATH
    device: str = DEVICE


# ---------------------------------------------------------------------------
# Capabilities (derived from SynthesisRequest — single source of truth)
# ---------------------------------------------------------------------------

CAPABILITIES = build_capabilities(
    SynthesisRequest,
    engine="ming3-tts-moe",
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
            "Cloning is in-context only: the exact transcript "
            "(reference_text) is required alongside the clip.  The engine "
            "re-encodes the reference per request (no prompt cache)."
        ),
    },
    languages=list(canonical_language_codes(LANGUAGE_CODES)),
    overrides={
        "cfg_scale": {"step": 0.1, "advanced": True},
        "flow_steps": {"step": 1, "advanced": True},
        "sigma": {"step": 0.05, "advanced": True},
        "temperature": {"step": 0.05, "advanced": True},
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
    title="Ming-3.0-tts (MoE) Voice Cloning API",
    description=(
        "REST API around Ming-3.0-tts tiny (7.9B MoE).  Send text + a "
        "reference audio sample + its exact transcript and get back cloned "
        "44.1 kHz speech.  Synthesis requests are serialized (single shared "
        "model).  Machine-readable parameter metadata at GET /capabilities."
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
# Runtime — thin wrapper around the Ming-3.0-tts MoE engine
# ---------------------------------------------------------------------------


@dataclass
class MingTinyTTSRuntime:
    """Holds the loaded model and its metadata for the lifetime of the server."""

    model: MingTinyTTS
    device: str
    sample_rate: int


_runtime: MingTinyTTSRuntime | None = None

# generate_speech() runs the AR decode loop and the flow-matching head
# against shared latent-history/KV buffers (not re-entrant), and the engine
# seeds process-wide RNG at the start of every synthesis.  Serialize
# synthesis; single-GPU throughput is the bottleneck anyway.
_synthesis_lock = threading.Lock()


def _get_runtime() -> MingTinyTTSRuntime:
    """Return the global runtime, loading the model once on first call.

    ``from_pretrained`` configures the release runtime environment (bf16,
    portable MoE loop, fp32 recurrent state, SDPA attention), pins
    transformers==4.46.3, and validates the checkpoint against the release
    contract — a wrong or half-downloaded checkpoint fails here at startup
    rather than mid-request.
    """
    global _runtime
    if _runtime is None:
        logger.info(
            "Loading Ming-3.0-tts MoE model '%s' on device '%s' ...",
            MODEL_NAME_OR_PATH,
            DEVICE,
        )
        model = MingTinyTTS.from_pretrained(MODEL_NAME_OR_PATH, device=DEVICE)
        _runtime = MingTinyTTSRuntime(
            model=model,
            device=DEVICE,
            # Echo the rate the engine enforces (44100 Hz) rather than a
            # hardcoded constant, so the response stays self-consistent.
            sample_rate=int(model.sample_rate),
        )
        logger.info(
            "Model loaded successfully. Sample rate: {} Hz, device: {}",
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
    <head><title>Ming-3.0-tts (MoE) REST API</title></head>
    <body>
        <h1>Ming-3.0-tts (MoE) Voice Cloning REST API</h1>
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
    Synthesize audio using the provided text, reference audio, and its
    exact transcript.

    In-context voice cloning: the engine selects its ``seed`` task for
    zh/en and ``multilingual`` for de/es/fr/ja/ko, then conditions on the
    reference clip and transcript together.  There is no transcript-free
    clone mode on these tasks, hence the 422.

    The full parameter list is documented at GET /capabilities; the request
    schema mirrors it exactly (same model, no drift).
    """
    runtime = _get_runtime()

    # Resolve randomised seed for reproducibility.
    seed = req.seed if req.seed is not None else random.randint(SEED_MIN, SEED_MAX)

    task = _task_for_language(req.language)

    logger.info(
        "Synthesizing: seed={}, task={}, text_len={}, ref_text_len={}, "
        "lang={}, cfg={:.1f}, flow_steps={}, sigma={:.2f}, temperature={:.2f}",
        seed,
        task,
        len(req.text),
        len(req.reference_text),
        req.language,
        req.cfg_scale,
        req.flow_steps,
        req.sigma,
        req.temperature,
    )

    # Decode and sanity-check the reference audio before touching the model.
    try:
        raw_audio = decode_base64(req.audio_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")

    _check_reference_audio(raw_audio)

    # The engine loads reference audio from a file path and re-encodes it
    # per request (no path-keyed cache), so a UUID temp file deleted per
    # request is fine.
    ref_audio_path = write_temp_audio(raw_audio, _TEMP_AUDIO_DIR)

    try:
        t0 = time.perf_counter()

        with _synthesis_lock:
            # The engine's own synthesize() seeds the whole pipeline from
            # the passed seed (no server-side re-seeding needed); the
            # output contract validation lives in MingTinyTTS.generate.
            output = runtime.model.generate(
                task=task,
                text=req.text,
                prompt_wav=ref_audio_path,
                prompt_text=req.reference_text,
                # docs/02: two-letter codes; the engine's
                # normalize_target_language maps codes to canonical names.
                target_language=req.language,
                seed=seed,
                cfg_scale=req.cfg_scale,
                flow_steps=req.flow_steps,
                sigma=req.sigma,
                temperature=req.temperature,
            )

        time_used = time.perf_counter() - t0

        # output.waveform is the generated audio ((1, N) or (N,), the
        # engine's own save_waveform normalizes both).
        wav = _tensor_to_numpy(output.waveform)

        sample_rate = runtime.sample_rate
        rtf = compute_rtf(time_used, len(wav), sample_rate)

        audio_bytes = _numpy_to_wav_bytes(wav, sample_rate)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        audio_duration = len(wav) / sample_rate if sample_rate else 0.0
        stopped = bool(output.stopped)
        logger.info(
            "Synthesis complete: {:.1f} s wall-clock, {:.1f} s audio, "
            "RTF={}, stopped={}",
            time_used,
            audio_duration,
            f"{rtf:.3f}" if rtf is not None else "n/a",
            stopped,
        )

        return SynthesisResponse(
            audio_base64=audio_b64,
            sample_rate=sample_rate,
            seed=seed,
            fid=str(uuid.uuid4()),
            time_used=time_used,
            rtf=rtf,
            stopped=stopped,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Synthesis failed: {}", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # Clean up the temporary reference audio file.
        cleanup_temp(ref_audio_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMP_AUDIO_DIR = temp_audio_dir("ming3_tts_moe_rest_api")


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


def _tensor_to_numpy(waveform) -> np.ndarray:
    """Convert the engine's waveform tensor to 1-D float32 numpy."""
    wav = waveform.detach().cpu().float()
    while wav.ndim > 1 and wav.shape[0] == 1:
        # Same batch-dim normalization the engine's own save_waveform does.
        wav = wav.squeeze(0)
    return wav.numpy()


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
# Main (for running directly: python server_ming3TTS_moe.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("MING3_TTS_MOE_HOST", "0.0.0.0")
    port = int(os.getenv("MING3_TTS_MOE_PORT", "7500"))
    logger.info("Starting Ming-3.0-tts (MoE) REST API server on %s:%d", host, port)
    # Pass the app object directly instead of a module path string,
    # so this works regardless of how the file is invoked.
    uvicorn.run(app, host=host, port=port, log_level="info")
