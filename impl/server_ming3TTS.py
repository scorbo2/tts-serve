"""
FastAPI REST server for Ming-3.0-tts voice cloning (0.5b dense model;
github.com/inclusionAI/Ming-3.0-tts).

Ming-3.0-tts ships two checkpoints — a 0.5b dense model and a 7.9B-total /
1.3B-active MoE ("tiny") — with separate Python stacks.  This server wraps
the **0.5b dense** stack (``ming_tts.MingTTS``); the MoE has its own server
(``server_ming3TTS_moe.py``) because the two engines pin conflicting
dependencies and share no tokenizer or prompt layout.

Zero-shot voice cloning only (the engine's ``TTS`` task): clients send text,
a reference audio sample (base64), and optionally the sample's exact
transcript.  With the transcript the engine continues from the reference
clip's AudioVAE latents *and* its speaker embedding (in-context cloning);
without it the acoustic continuation is dropped and only the speaker
embedding conditions the voice — the same transparent fallback Qwen3-TTS
uses, so omitting ``reference_text`` is a mode switch, not an error.  The
engine's other task modes (voice design, built-in IP voices, emotion /
speed / dialect control) are deliberately not exposed: this server is a
voice-cloning server.

Inline paralinguistic tags are part of the ``TTS`` task and work verbatim
in ``text``: [Breathing], [Cough], [Sigh], [Laughter], [Surprise-ah],
[Surprise-oh], [Dissatisfaction-hnn], [Uhm], [Confirmation-en],
[Question-ei], [Question-ah], [Question-en], [Question-oh], [Question-huh].

The engine reads the reference clip from a file path and caches the encoded
prompt latents per path, so reference audio is staged under a content-hashed
name and *kept* on disk (repeat requests for the same clip reuse the
engine's cache).  Wipe the staging directory to reclaim space.

``seed`` is meaningful: the process RNG is re-seeded inside the synthesis
lock, immediately before ``synthesize()`` — the prompt VAE encoder samples
its posterior, so RNG state is part of the output.  Caveat: encoded prompts
are cached *in memory* per path, so the first request for a clip runs the
VAE encoder (consuming RNG that later requests skip).  Repeat requests with
the same seed therefore reproduce each other exactly, but the very first
(cold) request for a clip is not reproducible by later ones — reproducing
it takes a server restart, which is the only thing that flushes the
in-memory cache; deleting the staging directory merely reclaims disk.  On
CUDA, reproducibility is near-exact, not bit-exact (non-deterministic
kernels).

44.1 kHz output: the released 0.5b checkpoint declares
``audio_tokenizer_config.sample_rate: 44100`` in its config.json; the
response echoes the rate read from the loaded checkpoint, while the
capabilities document carries the constant (derived at import time, so a
foreign checkpoint via MING3_TTS_MODEL would still be advertised as 44100).

Text normalization (numbers/dates/units via FST grammars) needs ``pynini``,
which has no universal wheel; set MING3_TTS_TEXT_NORMALIZATION=0 to skip it
(numbers are then spoken as written).  ``flash-attn`` is optional; the
engine falls back to PyTorch SDPA automatically.

Configuration (environment variables):
    MING3_TTS_MODEL              Local checkpoint directory or HuggingFace id.
                                 Default: inclusionAI/Ming-3.0-tts-0.5b
    MING3_TTS_DEVICE             Torch device to load the model on: 'cuda',
                                 'cuda:N', 'cpu', or 'mps'.  The model runs
                                 in bfloat16; CPU is functional but slow.
                                 Default: cuda
    MING3_TTS_TEXT_NORMALIZATION Run the engine's FST text normalizer
                                 (needs pynini).  0/1.  Default: 1
    MING3_TTS_HOST               Bind host for `python server_ming3TTS.py`.
                                 Default: 0.0.0.0
    MING3_TTS_PORT               Bind port for `python server_ming3TTS.py`.
                                 Default: 7500

Extra dependencies beyond the Ming-3.0-tts repository:
    git clone https://github.com/inclusionAI/Ming-3.0-tts.git
    cd Ming-3.0-tts && pip install -r requirements.txt
    pip install fastapi uvicorn loguru soundfile
    pip install ../tts-engine-common # in-repo copy; or: pip install -e ../tts-engine-common

The server must be able to import the engine's top-level ``ming_tts``
module.  Running `python /path/to/server_ming3TTS.py` puts the *script's*
directory on sys.path, not your working directory, so the Ming-3.0-tts repo
root must be added explicitly:

    cd Ming-3.0-tts
    PYTHONPATH=. MING3_TTS_MODEL=/path/to/Ming-3.0-tts-0.5b \
        python ../tts-serve/impl/server_ming3TTS.py

Usage:
    python server_ming3TTS.py
    # or: uvicorn server_ming3TTS:app --host 0.0.0.0 --port 7500
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
# its top-level modules (ming_tts, plus its own imports of tts_tasks /
# frontend / modeling_bailingmm from the repo root) are importable only when
# the repo root is on PYTHONPATH.  Give the most common deployment failure
# mode a message people can actually act on.
try:
    from ming_tts import MingTTS
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"The Ming-3.0-tts engine module {exc.name!r} is not importable.  "
        "The Ming-3.0-tts repo root must be on PYTHONPATH (its requirements "
        "must be installed too) -- e.g. from the Ming-3.0-tts repo root:  "
        "PYTHONPATH=. python /path/to/tts-serve/impl/server_ming3TTS.py"
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
    compute_rtf,
    decode_base64,
    normalize_language,
    stage_audio,
    temp_audio_dir,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME_OR_PATH = os.getenv("MING3_TTS_MODEL", "inclusionAI/Ming-3.0-tts-0.5b")
DEVICE = os.getenv("MING3_TTS_DEVICE", "cuda")


def _parse_bool_env(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


TEXT_NORMALIZATION = _parse_bool_env(os.getenv("MING3_TTS_TEXT_NORMALIZATION", "1"))

# The released 0.5b checkpoint's audio VAE (audio_tokenizer_config.
# sample_rate in its config.json) synthesizes at 44.1 kHz.  The response
# echoes the loaded checkpoint's own rate; this constant only feeds the
# import-time capabilities document.
SAMPLE_RATE = 44100

SEED_MIN = 1
SEED_MAX = 1000

# Heuristic lower bound: the CampPlus speaker embedding needs enough speech
# for usable frames, and the AudioVAE prompt continuation needs enough
# audio to seed the first decode step.
MIN_PROMPT_DURATION_S = 2.0

# Sanity valve for the request payload (~85 s of 44.1 kHz 16-bit mono).
MAX_AUDIO_B64_LEN = 10_000_000

# Engine-specific defaults, mirroring ``MingTTS.synthesize``'s own signature
# (flow-matching sampler parameters).
DEFAULT_CFG = 2.0
DEFAULT_SIGMA = 0.25
DEFAULT_TEMPERATURE = 0.0

# Ming's advertised language set: Chinese and English natively, plus the
# five languages added in Ming-3.0 (the engine's MULTILINGUAL task).  The
# engine's ``language_id`` accepts these two-letter codes directly
# (tts_tasks._TARGET_LANGUAGE_NAMES lower-cases and maps them to the
# canonical "Target language: <Name>" line), so no server-side mapping
# table is needed — unlike the Qwen3-TTS servers.  Cantonese is a dialect
# (its own engine task, not exposed here), and the engine offers no
# auto-detection sentinel for these tasks, so no 'auto' member.
LANGUAGE_CODES = ("zh", "en", "ja", "ko", "de", "fr", "es")

_DEVICE_PATTERN = re.compile(r"^(cuda(:\d+)?|cpu|mps)$")


def _validate_config() -> None:
    """Fail fast on bad configuration instead of partway through a model download."""
    # Grammar check only, VoxCPM-style: 'cuda' / 'cuda:N' / 'cpu' / 'mps'.
    # Torch itself rejects an unavailable device at load time.
    if not _DEVICE_PATTERN.fullmatch(DEVICE):
        raise ValueError(
            "MING3_TTS_DEVICE must be 'cuda', 'cuda:N', 'cpu', or 'mps', "
            f"got {DEVICE!r}"
        )


_validate_config()

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

# Dynamic Literal over the checkpoint's language codes so the request
# schema and the /capabilities enum share one source (no 'auto': the
# engine's TTS task has no auto-detection sentinel to map onto).  Built via
# the shared helper so the enum is pinned to canonical sorted order — see
# tts_engine_common.canonical_language_literal for the pydantic Literal
# collision that hand-rolled Literals invite (the one that once leaked the
# 0.5b code order into the MoE server's capabilities snapshot).
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
    reference_text: str | None = Field(
        None,
        description=(
            "Exact transcript of the reference audio.  With it, cloning is "
            "in-context (the reference clip's acoustic continuation plus "
            "its speaker embedding) and matches the reference closely.  "
            "Omitted, the engine falls back to speaker-embedding-only "
            "cloning (timbre closer, prosody freer)."
        ),
    )
    language: Language | None = Field(
        DEFAULT_LANGUAGE,
        description=(
            "Two-letter language code (supported: "
            f"{', '.join(sorted(LANGUAGE_CODES))}).  Omitted or empty "
            "defaults to 'en'.  Set it to the language of *text*."
        ),
    )
    seed: int | None = Field(
        None,
        ge=SEED_MIN,
        le=SEED_MAX,
        description=(
            "Random seed for reproducibility.  If omitted, a random seed "
            f"in [{SEED_MIN}, {SEED_MAX}] is chosen and echoed in the "
            "response.  Note: the engine caches encoded reference prompts "
            "in memory per clip: repeat requests with the same seed "
            "reproduce each other, but the first request for a clip draws "
            "RNG the later ones skip, so it is only reproducible after a "
            "server restart (wiping the staging dir does not reset the "
            "cache)."
        ),
    )

    # --- engine-specific tuning (defaults mirror the engine's own) ---------
    cfg: float = Field(
        DEFAULT_CFG,
        gt=0.0,
        description=(
            "Flow-matching classifier-free guidance scale.  Engine default "
            f"{DEFAULT_CFG:.1f}.  Must be finite and > 0."
        ),
    )
    sigma: float = Field(
        DEFAULT_SIGMA,
        ge=0.0,
        description=(
            "Flow-matching noise scale.  Engine default "
            f"{DEFAULT_SIGMA:.2f}.  Must be finite and >= 0."
        ),
    )
    temperature: float = Field(
        DEFAULT_TEMPERATURE,
        ge=0.0,
        description=(
            "Sampling temperature.  Engine default 0.0 (greedy); raise for "
            "more variation.  Must be finite and >= 0."
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
    def _validate_reference_text(cls, v: str | None) -> str | None:
        # A *provided* blank transcript would silently request in-context
        # cloning with an empty prompt; an explicit null means 'no
        # transcript' (the speaker-embedding-only fallback).  The None
        # guard is load-bearing: Pydantic runs after-mode validators for
        # an explicitly-provided null, so without it a null 500s here.
        if v is None:
            return v
        if not v.strip():
            raise ValueError("reference_text must contain non-whitespace characters")
        return v

    @field_validator("cfg", "sigma", "temperature")
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
    """The synthesis result (core fields from tts_engine_common, plus fid)."""

    fid: str = Field(..., description="Request ID (internal).")


class HealthResponse(CoreHealthResponse):
    """The /health core from tts_engine_common, with this engine's serverType.

    model/device pick up the import-time config; version carries the shared
    RELEASE_VERSION constant — overriding it here is how a server drifts.
    """

    serverType: Literal["ming3-tts"] = "ming3-tts"
    model: str = MODEL_NAME_OR_PATH
    device: str = DEVICE


# ---------------------------------------------------------------------------
# Capabilities (derived from SynthesisRequest — single source of truth)
# ---------------------------------------------------------------------------

CAPABILITIES = build_capabilities(
    SynthesisRequest,
    engine="ming3-tts",
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
            "With reference_text the clone is in-context (acoustic "
            "continuation + speaker embedding); without it the engine "
            "falls back to speaker-embedding-only cloning.  The engine "
            "caches encoded prompts per clip, so repeated requests for the "
            "same clip are cheaper."
        ),
    },
    languages=list(canonical_language_codes(LANGUAGE_CODES)),
    overrides={
        "cfg": {"step": 0.1, "advanced": True},
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
    title="Ming-3.0-tts (0.5b) Voice Cloning API",
    description=(
        "REST API around Ming-3.0-tts 0.5b (dense).  Send text + a reference "
        "audio sample (and optionally its exact transcript) and get back "
        "cloned 44.1 kHz speech.  Synthesis requests are serialized (single "
        "shared model).  Machine-readable parameter metadata at GET "
        "/capabilities."
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
# Runtime — thin wrapper around the Ming-3.0-tts 0.5b model
# ---------------------------------------------------------------------------


@dataclass
class MingTTSRuntime:
    """Holds the loaded model and its metadata for the lifetime of the server."""

    model: MingTTS
    device: str
    sample_rate: int


_runtime: MingTTSRuntime | None = None

# synthesize() mutates shared model state (rope deltas on the backbone, the
# path-keyed prompt-latent cache) and runs under @torch.inference_mode(),
# which is thread-affine.  Serialize synthesis; single-GPU throughput is the
# bottleneck anyway.
_synthesis_lock = threading.Lock()


def _get_runtime() -> MingTTSRuntime:
    """Return the global runtime, loading the model once on first call.

    The engine loads in bfloat16 and relaxes any flash_attention_2 pin to
    SDPA when flash-attn is absent (its own config loader does this), so no
    extra flags are needed here.
    """
    global _runtime
    if _runtime is None:
        logger.info(
            "Loading Ming-3.0-tts (0.5b) model '%s' on device '%s' "
            "(text_normalization=%s) ...",
            MODEL_NAME_OR_PATH,
            DEVICE,
            TEXT_NORMALIZATION,
        )
        model = MingTTS(
            MODEL_NAME_OR_PATH,
            device=DEVICE,
            text_normalization=TEXT_NORMALIZATION,
        )
        _runtime = MingTTSRuntime(
            model=model,
            device=DEVICE,
            # Echo the rate the checkpoint itself declares (44100 Hz on the
            # released 0.5b), so the response stays self-consistent.
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
    <head><title>Ming-3.0-tts (0.5b) REST API</title></head>
    <body>
        <h1>Ming-3.0-tts (0.5b) Voice Cloning REST API</h1>
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

    Zero-shot voice cloning via the engine's ``TTS`` task: with
    ``reference_text`` the clone is in-context (the clip's acoustic
    continuation conditions the flow-matching decoder), without it the
    server transparently falls back to speaker-embedding-only cloning.

    The full parameter list is documented at GET /capabilities; the request
    schema mirrors it exactly (same model, no drift).
    """
    runtime = _get_runtime()

    # Resolve randomised seed for reproducibility.
    seed = req.seed if req.seed is not None else random.randint(SEED_MIN, SEED_MAX)

    logger.info(
        "Synthesizing: seed={}, mode={}, text_len={}, ref_text_len={}, "
        "lang={}, cfg={:.1f}, sigma={:.2f}, temperature={:.2f}",
        seed,
        "in-context-clone" if req.reference_text else "speaker-embedding-only",
        len(req.text),
        len(req.reference_text or ""),
        req.language,
        req.cfg,
        req.sigma,
        req.temperature,
    )

    # Decode and sanity-check the reference audio before touching the model.
    try:
        raw_audio = decode_base64(req.audio_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")

    _check_reference_audio(raw_audio)

    # The engine loads reference audio from a file path and caches the
    # encoded prompt latents per path (cache_prompts=True), so stage under
    # the content hash and keep the file (deleting it would race with a
    # queued request for the same clip and defeat the cache).
    prompt_audio_path = stage_audio(raw_audio, _TEMP_AUDIO_DIR)

    try:
        t0 = time.perf_counter()

        with _synthesis_lock:
            # The prompt VAE encoder samples its posterior, so RNG state is
            # part of the output; seed the whole pipeline inside the lock,
            # exactly like the engine's own infer.py seed_everything().
            seed_everything(seed)
            waveform = runtime.model.synthesize(
                task="TTS",
                text=req.text,
                # docs/02: two-letter codes; the engine maps codes to its
                # canonical "Target language" names internally.
                language_id=req.language,
                prompt_wav_path=prompt_audio_path,
                # None here is the documented fallback: the engine drops the
                # acoustic continuation and keeps only the speaker embedding.
                prompt_text=req.reference_text,
                use_spk_emb=True,
                cfg=req.cfg,
                sigma=req.sigma,
                temperature=req.temperature,
            )

        time_used = time.perf_counter() - t0

        # synthesize() returns a (1, samples) waveform on the model device.
        wav = _tensor_to_numpy(waveform)

        sample_rate = runtime.sample_rate
        rtf = compute_rtf(time_used, len(wav), sample_rate)

        audio_bytes = _numpy_to_wav_bytes(wav, sample_rate)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        audio_duration = len(wav) / sample_rate if sample_rate else 0.0
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

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Synthesis failed: {}", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMP_AUDIO_DIR = temp_audio_dir("ming3_tts_rest_api")


def seed_everything(seed: int) -> None:
    """Seed every RNG the pipeline draws from (mirrors the engine's infer.py).

    Deliberately inside the synthesis lock: prompt-VAE posterior sampling
    and flow matching both consume the global RNG, so a request is only
    reproducible when its seeding start is not interleaved with another
    request's draws.  (The engine's own script also pins cuDNN to
    deterministic kernels; that stays a deployment knob here because it
    costs throughput.)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Guard the CUDA seeding like every other seed_everything in impl/:
    # torch.cuda.* access raises on torch builds without CUDA compiled in,
    # and this server explicitly supports 'cpu'/'mps' via MING3_TTS_DEVICE.
    # (The engine's own infer.py skips the guard because it assumes CUDA —
    # this server does not.)
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


def _tensor_to_numpy(waveform) -> np.ndarray:
    """Convert the engine's (1, samples) torch tensor to 1-D float32 numpy."""
    wav = waveform.detach().cpu().float()
    if wav.ndim > 1:
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
# Main (for running directly: python server_ming3TTS.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("MING3_TTS_HOST", "0.0.0.0")
    port = int(os.getenv("MING3_TTS_PORT", "7500"))
    logger.info("Starting Ming-3.0-tts (0.5b) REST API server on %s:%d", host, port)
    # Pass the app object directly instead of a module path string,
    # so this works regardless of how the file is invoked.
    uvicorn.run(app, host=host, port=port, log_level="info")
