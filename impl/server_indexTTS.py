"""
FastAPI REST server for IndexTTS-2.5 voice cloning (index-tts).

IndexTTS-2.5 is a zero-shot multilingual voice-cloning model with
controllable emotion.  Clients send text plus a reference audio sample
(base64); the transcript of that sample is *not* used — the engine
conditions on the audio alone.  The server returns the generated audio as
base64-encoded 22.05 kHz WAV (the 22 kHz BigVGAN vocoder — the third
distinct rate in this repo, after 24 kHz and 48 kHz).

Emotion is steered through exactly one mechanism per request (the engine
mixes from a single source and would silently discard the others):
  * ``emotion_audio_base64`` — a second reference clip; the engine extracts
    its emotion embedding and blends it with the speaker's own emotion
    according to ``emotion_alpha`` (0.0 = speaker's emotion, 1.0 = the
    clip's emotion).
  * ``emotion_vector`` — an 8-component vector
    ``[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]``;
    ``emotion_alpha`` scales its strength.
  * ``emotion_text`` — free text describing the desired emotion; the server
    must be started with ``INDEXTTS_USE_QWEN_EMO=1`` (loads the QwenEmotion
    text-to-emotion model) or the request is rejected with 400.

Loads the model once on startup, then exposes a single POST endpoint for
synthesis.

Capabilities: GET /capabilities returns a machine-readable description of
every request parameter, derived from the Pydantic request model so it can
never drift from what the server actually validates (see the
tts-engine-common README).

Model weights (IndexTeam/IndexTTS-2.5) are downloaded from HuggingFace into
``INDEXTTS_MODEL_DIR`` on first start if that directory does not already
contain the checkpoint (``config.yaml``).  Auxiliary models (w2v-bert, etc.)
are fetched by the engine itself at load time.  Note: ``pip install indextts``
installs the *v2* inference stack; this server imports the v2.5 module
(``indextts.infer_v2_5``) and pins the matching checkpoint, so use the
index-tts repository (or a package build) that ships ``infer_v2_5.py``.

Configuration (environment variables):
    INDEXTTS_MODEL_DIR     Directory holding the IndexTTS-2.5 checkpoint.
                           Downloaded from HuggingFace if it lacks config.yaml.
                           Default: checkpoints
    INDEXTTS_DEVICE        Device to load the model on: 'cuda', 'cuda:N',
                           'cpu', 'mps' or 'xpu'.  Unset = the engine
                           auto-selects (CUDA, else XPU, else MPS, else CPU).
    INDEXTTS_USE_BF16      '1' to load with bfloat16 weights (lower VRAM
                           use; the engine itself disables it on CPU/MPS).
                           Default: off
    INDEXTTS_USE_QWEN_EMO  '1' to additionally load the QwenEmotion model,
                           enabling the ``emotion_text`` request parameter.
                           Default: off
    INDEXTTS_HOST          Bind host for `python server_indexTTS.py`.
                           Default: 0.0.0.0
    INDEXTTS_PORT          Bind port for `python server_indexTTS.py`.
                           Default: 7500

Extra dependencies beyond the index-tts package:
    pip install fastapi uvicorn loguru soundfile
    pip install ../tts-engine-common # in-repo copy; or: pip install -e ../tts-engine-common

Usage:
    python server_indexTTS.py
    # or: uvicorn server_indexTTS:app --host 0.0.0.0 --port 7500
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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from indextts.infer_v2_5 import IndexTTS2
from indextts.utils.model_download import snapshot_download
from tts_engine_common import (
    DEFAULT_LANGUAGE,
    CoreSynthesisResponse,
    build_capabilities,
    capabilities_endpoint,
    compute_rtf,
    decode_base64,
    normalize_language,
    stage_audio,
    temp_audio_dir,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_REPO_ID = "IndexTeam/IndexTTS-2.5"

MODEL_DIR = os.getenv("INDEXTTS_MODEL_DIR", "checkpoints").strip() or "checkpoints"
DEVICE = os.getenv("INDEXTTS_DEVICE", "").strip()
USE_BF16 = os.getenv("INDEXTTS_USE_BF16", "").strip().lower() in ("1", "true", "yes", "on")
USE_QWEN_EMO = os.getenv("INDEXTTS_USE_QWEN_EMO", "").strip().lower() in ("1", "true", "yes", "on")

# The engine's 22 kHz BigVGAN vocoder; infer() returns the authoritative
# rate in its result tuple, which is what the response echoes.
SAMPLE_RATE = 22050

SEED_MIN = 1
SEED_MAX = 1000

# Heuristic lower bound: below this the speaker embedding / reference codes
# degrade to near-garbage.  The engine hard-cuts references at 15 s.
MIN_REF_DURATION_S = 2.0

# Sanity valve for the request payload (~7 min of 22.05 kHz audio).
MAX_AUDIO_B64_LEN = 10_000_000

# The webui's language dropdown.  The engine's lang_to_token() would
# silently fall back to the 'common' vocabulary for anything else, so
# validate against the advertised set instead of letting it degrade.
LANGUAGE_CODES = ("ar", "en", "es", "ja", "zh")

# The 8 emotion-vector components, in the order the engine expects.
EMOTION_VECTOR_LABELS = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)


def _validate_config() -> None:
    """Fail fast on bad configuration instead of partway through a model download."""
    if MODEL_DIR in ("", "."):
        raise ValueError("INDEXTTS_MODEL_DIR must name a real directory, got empty/'.'")
    if DEVICE:
        family = DEVICE.split(":", 1)[0]
        if family not in ("cuda", "cpu", "mps", "xpu"):
            raise ValueError(
                f"INDEXTTS_DEVICE must be 'cuda', 'cuda:N', 'cpu', 'mps' or "
                f"'xpu'; leave it unset for auto-selection, got {DEVICE!r}"
            )


_validate_config()

# Capabilities/health report a device *family* at import time (the model is
# not loaded yet); the engine resolves the same auto-selection at load time.
DEVICE_REPORT = (
    DEVICE.split(":", 1)[0]
    if DEVICE
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

# Dynamic Literal over the engine's language codes so the request schema and
# the /capabilities enum share one source.
Language = Literal[tuple(LANGUAGE_CODES)]


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
            "Reference voice sample (roughly 5-10 s works well) as a base64 "
            "string.  Any container soundfile can decode (WAV, MP3, OGG, "
            "FLAC, ...).  Only the first 15 s are used — the engine hard-"
            "cuts longer clips.  No transcript is needed."
        ),
    )
    language: Language | None = Field(
        DEFAULT_LANGUAGE,
        description=(
            "Two-letter language code "
            f"(supported: {', '.join(LANGUAGE_CODES)}).  "
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

    # --- emotion control (at most one source per request) ------------------
    emotion_audio_base64: str | None = Field(
        None,
        min_length=1,
        max_length=MAX_AUDIO_B64_LEN,
        description=(
            "Optional second reference clip whose *emotion* to adopt, as a "
            "base64 string (the voice itself still comes from audio_base64). "
            "Use with emotion_alpha to blend; mutually exclusive with "
            "emotion_vector and emotion_text."
        ),
    )
    emotion_alpha: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Strength of the emotion guidance.  With emotion_audio_base64: "
            "0.0 keeps the speaker's own emotion, 1.0 fully adopts the "
            "emotion clip's.  With emotion_vector/emotion_text: scales the "
            "resulting emotion vector (1.0 = full strength)."
        ),
    )
    emotion_vector: list[float] | None = Field(
        None,
        min_length=8,
        max_length=8,
        description=(
            "Optional 8-component emotion vector "
            f"[{', '.join(EMOTION_VECTOR_LABELS)}], each in [0.0, 1.0].  "
            "Mutually exclusive with emotion_audio_base64 and emotion_text."
        ),
    )
    emotion_text: str | None = Field(
        None,
        min_length=1,
        description=(
            "Optional free-text description of the desired emotion (e.g. "
            "'excited and cheerful'), converted to an emotion vector by the "
            "QwenEmotion model.  Requires the server to be started with "
            "INDEXTTS_USE_QWEN_EMO=1, otherwise requests are rejected with "
            "400.  Mutually exclusive with the other emotion sources."
        ),
    )

    # --- engine-specific tuning (None = engine default) ---------------------
    duration_factor: float = Field(
        1.0,
        ge=0.1,
        le=3.0,
        description=(
            "Length control for the output relative to natural speaking "
            "length (1.0 = natural; >1 stretches, <1 compresses)."
        ),
    )
    temperature: float | None = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.  Omit for the engine default (0.8).",
    )
    top_p: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold.  Omit for the engine default (0.8).",
    )
    top_k: int | None = Field(
        None,
        ge=1,
        le=100,
        description="Top-k sampling threshold.  Omit for the engine default (30).",
    )
    repetition_penalty: float | None = Field(
        None,
        ge=1.0,
        le=20.0,
        description="Penalty applied to repeated tokens.  Omit for the engine default (10.0).",
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
        # Runs before the Literal check so the normalized default ('en') is
        # always a valid member.
        return normalize_language(v)

    @field_validator("emotion_text")
    @classmethod
    def _validate_emotion_text(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("emotion_text must contain non-whitespace characters")
        return v

    @field_validator("emotion_vector")
    @classmethod
    def _validate_emotion_vector(cls, v: list[float] | None) -> list[float] | None:
        # The JSON schema can express the item count (8) but not per-item
        # ranges; enforce [0, 1] per component here.
        if v is None:
            return v
        for index, component in enumerate(v):
            if not 0.0 <= component <= 1.0:
                raise ValueError(
                    f"emotion_vector component {index} ({EMOTION_VECTOR_LABELS[index]}) "
                    f"must be in [0.0, 1.0], got {component}"
                )
        return v

    @model_validator(mode="after")
    def _single_emotion_source(self) -> SynthesisRequest:
        # The engine mixes emotion from one mechanism per request and would
        # silently discard an external emotion clip when a vector is given,
        # so combining sources is a client bug — reject it loudly (422).
        sources = [
            name
            for name, value in (
                ("emotion_audio_base64", self.emotion_audio_base64),
                ("emotion_vector", self.emotion_vector),
                ("emotion_text", self.emotion_text),
            )
            if value not in (None, "")
        ]
        if len(sources) > 1:
            raise ValueError(
                f"provide at most one emotion source per request, got: {', '.join(sources)}"
            )
        return self


class SynthesisResponse(CoreSynthesisResponse):
    """The synthesis result (core fields from tts_engine_common, plus fid)."""

    fid: str = Field(..., description="Request ID (internal).")


class HealthResponse(BaseModel):
    """Health / readiness check."""

    status: Literal["ok"] = "ok"
    serverType: Literal["index-tts"] = "index-tts"
    model: str = MODEL_DIR
    device: str = DEVICE_REPORT


# ---------------------------------------------------------------------------
# Capabilities (derived from SynthesisRequest — single source of truth)
# ---------------------------------------------------------------------------

CAPABILITIES = build_capabilities(
    SynthesisRequest,
    engine="index-tts",
    model=MODEL_DIR,
    device=DEVICE_REPORT,
    sample_rate=SAMPLE_RATE,
    watermarked=False,
    endpoint="/synthesize",
    reference_audio={
        "required": True,
        "formats": ["wav", "mp3", "ogg", "flac"],
        "min_duration_s": MIN_REF_DURATION_S,
        "note": (
            "Voice cloning conditions on the audio alone — no transcript "
            "needed.  Only the first 15 s of the clip are used (the engine "
            "hard-cuts longer ones).  For emotion control, send "
            "emotion_audio_base64 (a second clip), emotion_vector (8 "
            "components) or emotion_text (QwenEmotion; server must be "
            "started with INDEXTTS_USE_QWEN_EMO=1) — at most one per request."
        ),
    },
    languages=list(LANGUAGE_CODES),
    overrides={
        "emotion_alpha": {"step": 0.05},
        "duration_factor": {"step": 0.1},
        "emotion_vector": {
            "advanced": True,
            # Same order as the request schema (the 8 components); lets a
            # client render one labeled row per component instead of 8
            # unlabeled boxes.
            "item_labels": list(EMOTION_VECTOR_LABELS),
        },
        "emotion_text": {"advanced": True},
        "temperature": {"step": 0.05, "advanced": True},
        "top_p": {"step": 0.01, "advanced": True},
        "top_k": {"step": 1, "advanced": True},
        "repetition_penalty": {"step": 0.1, "advanced": True},
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
    title="IndexTTS-2.5 Voice Cloning API",
    description=(
        "REST API around IndexTTS-2.5.  Send text + a reference audio sample "
        "and get back cloned speech, with optional emotion control.  "
        "Machine-readable parameter metadata at GET /capabilities."
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
# Runtime — thin wrapper around the IndexTTS2 model
# ---------------------------------------------------------------------------


@dataclass
class IndexTTSRuntime:
    """Holds the loaded model and its metadata for the lifetime of the server."""

    model: IndexTTS2
    device: str


_runtime: IndexTTSRuntime | None = None

# The model keeps speaker/emotion conditioning caches keyed by prompt file
# path and is not thread-safe (shared GPT KV state during generation), so
# concurrent infer() calls would corrupt each other.  Serialize synthesis;
# single-GPU throughput is the bottleneck anyway.
_synthesis_lock = threading.Lock()


def _ensure_model_resources() -> None:
    """Download the main checkpoint snapshot if the model dir lacks it.

    The engine fetches its auxiliary models (w2v-bert, ...) itself at load
    time, so the server only has to make the main snapshot available.
    config.yaml is the marker: it is the first file the constructor reads.
    """
    config_path = Path(MODEL_DIR) / "config.yaml"
    if config_path.is_file():
        return
    logger.info(
        "No IndexTTS-2.5 checkpoint in %r — downloading %s ...",
        MODEL_DIR,
        MODEL_REPO_ID,
    )
    snapshot_download(MODEL_REPO_ID, MODEL_DIR)
    if not config_path.is_file():
        raise RuntimeError(
            f"Downloaded {MODEL_REPO_ID} into {MODEL_DIR!r} but config.yaml "
            "is still missing; check the directory contents."
        )


def _get_runtime() -> IndexTTSRuntime:
    """Return the global runtime, loading the model once on first call."""
    global _runtime
    if _runtime is None:
        logger.info(
            "Loading IndexTTS-2.5 model from %r (bf16=%s, qwen_emo=%s) ...",
            MODEL_DIR,
            USE_BF16,
            USE_QWEN_EMO,
        )
        _ensure_model_resources()
        model = IndexTTS2(
            cfg_path=str(Path(MODEL_DIR) / "config.yaml"),
            model_dir=MODEL_DIR,
            use_bf16=USE_BF16,
            use_qwen_emo=USE_QWEN_EMO,
            **({"device": DEVICE} if DEVICE else {}),
        )
        _runtime = IndexTTSRuntime(model=model, device=str(model.device))
        logger.info("Model loaded successfully. Device: %s", _runtime.device)
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
    <head><title>IndexTTS-2.5 REST API</title></head>
    <body>
        <h1>IndexTTS-2.5 Voice Cloning REST API</h1>
        <p>Model: <code>{MODEL_REPO_ID}</code> (checkpoint dir <code>{MODEL_DIR}</code>) on
        <code>{DEVICE_REPORT}</code>.  This server is a REST API, not a web server.</p>
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

    IndexTTS-2.5 performs zero-shot voice cloning: a short reference audio
    clip provides the speaker's voice (no transcript needed), and emotion
    can optionally be steered with a second emotion clip, an 8-component
    emotion vector, or a free-text emotion description.

    The full parameter list is documented at GET /capabilities; the request
    schema mirrors it exactly (same model, no drift).
    """
    runtime = _get_runtime()

    # Resolve randomised seed.
    seed = req.seed if req.seed is not None else random.randint(SEED_MIN, SEED_MAX)

    # docs/02: the API speaks two-letter codes; the engine wants uppercase
    # codes ('EN').  No 'auto' — the engine has no auto-detection mode.
    engine_language = req.language.upper()

    # emotion_text needs the optional QwenEmotion model; reject before doing
    # any audio work (this is a configuration 400, not a malformed input).
    if req.emotion_text is not None and not USE_QWEN_EMO:
        raise HTTPException(
            status_code=400,
            detail=(
                "emotion_text requires the server to be started with "
                "INDEXTTS_USE_QWEN_EMO=1 (the QwenEmotion model is not loaded)."
            ),
        )

    logger.info(
        "Synthesizing: seed={}, text_len={}, lang={} (engine: {}), "
        "duration_factor={:.2f}, emotion_source={}",
        seed,
        len(req.text),
        req.language,
        engine_language,
        req.duration_factor,
        _emotion_source_label(req),
    )

    # Decode the speaker reference in memory; the engine reads it back from a
    # staged file (see stage_audio in tts_engine_common).
    try:
        raw_audio = decode_base64(req.audio_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")

    _check_reference_audio(raw_audio)
    spk_audio_path = stage_audio(raw_audio, _TEMP_AUDIO_DIR)

    emo_audio_path = None
    if req.emotion_audio_base64 is not None:
        try:
            raw_emo = decode_base64(req.emotion_audio_base64)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 emotion audio: {exc}")
        _check_reference_audio(raw_emo)
        emo_audio_path = stage_audio(raw_emo, _TEMP_AUDIO_DIR)

    # The engine's infer() expects a *pre-normalized* emotion vector (the
    # webui applies the same bias + 0.8-sum cap via normalize_emo_vec), so
    # mirror that here rather than trusting raw client numbers.
    emo_vector = (
        runtime.model.normalize_emo_vec(req.emotion_vector, apply_bias=True)
        if req.emotion_vector is not None
        else None
    )

    # Only forward sampling params the client actually set, so the engine's
    # own defaults apply otherwise.
    sampling = {
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "repetition_penalty": req.repetition_penalty,
    }
    sampling = {k: v for k, v in sampling.items() if v is not None}

    t0 = time.perf_counter()
    try:
        with _synthesis_lock:
            seed_everything(seed)
            result = runtime.model.infer(
                spk_audio_prompt=spk_audio_path,
                text=req.text,
                output_path=None,
                lang=engine_language,
                emo_audio_prompt=emo_audio_path,
                emo_alpha=req.emotion_alpha,
                emo_vector=emo_vector,
                use_emo_text=req.emotion_text is not None,
                emo_text=req.emotion_text,
                duration_factor=req.duration_factor,
                **sampling,
            )

        time_used = time.perf_counter() - t0
    except Exception as exc:
        logger.error("Synthesis failed: {}", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    # infer() returns (sample_rate, wav) where wav is int16 numpy (N, 1),
    # or None when the engine gives up (e.g. OOM) — a hard 500 either way
    # if there is no audio.
    if not (isinstance(result, tuple) and len(result) == 2):
        raise HTTPException(
            status_code=500,
            detail="Engine produced no audio (infer() returned None).",
        )
    sr, wav = result
    wav = _to_float_mono(wav)

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMP_AUDIO_DIR = temp_audio_dir("index_tts_rest_api")


def _emotion_source_label(req: SynthesisRequest) -> str:
    """Which emotion mechanism (if any) this request uses, for logging."""
    if req.emotion_audio_base64 is not None:
        return "audio"
    if req.emotion_vector is not None:
        return "vector"
    if req.emotion_text is not None:
        return "text"
    return "none"


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


def seed_everything(seed: int) -> None:
    """Set the random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_float_mono(wav: object) -> np.ndarray:
    """Normalize the engine's int16 (N, 1) output to a 1-D float in [-1, 1]."""
    array = np.asarray(wav)
    if array.dtype == np.int16:
        array = array.astype(np.float32) / 32768.0
    # Clip to [-1, 1]: the PCM_16 conversion wraps out-of-range floats instead
    # of clamping them, which would produce crackling artifacts.
    return np.clip(array, -1.0, 1.0).reshape(-1)


def _numpy_to_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    """Convert a numpy audio array to WAV-encoded bytes (PCM_16)."""
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
# Main (for running directly: python server_indexTTS.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("INDEXTTS_HOST", "0.0.0.0")
    port = int(os.getenv("INDEXTTS_PORT", "7500"))
    logger.info("Starting IndexTTS-2.5 REST API server on %s:%d", host, port)
    # Pass the app object directly instead of a module path string,
    # so this works regardless of how the file is invoked.
    uvicorn.run(app, host=host, port=port, log_level="info")
