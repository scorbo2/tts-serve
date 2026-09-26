"""
FastAPI REST server for Breeze TTS 2 voice cloning and voice direction
(github.com/breezeblue-ai/breeze-tts).

Breeze TTS 2 is a real-time, bilingual (English / Chinese) TTS model.  This
server exposes the engine's two reference-based modes:

- **Voice clone** — reference audio + its exact transcript.  The engine
  preserves the speaker's timbre, rhythm, emotion, and style as-is.
- **Voice direction** — the same, plus a natural-language `instruction`
  that steers tone, emotion, pace, and delivery while keeping the speaker's
  identity.  `cfg_scale` strengthens instruction-following (the engine
  README recommends 4).

The engine's third mode, voice design (no reference audio; a
natural-language voice description instead), is deliberately NOT exposed:
the client application does not support it.

Loads the model once on startup, then exposes a single POST endpoint for
synthesis.  Clients send text, a reference audio sample (base64), the
sample's exact transcript, and optionally a direction instruction; the
server returns the generated audio as base64-encoded 24 kHz WAV.

Capabilities: GET /capabilities returns a machine-readable description of
every request parameter, derived from the Pydantic request model so it can
never drift from what the server actually validates (see the
tts-engine-common README).

The engine requires an NVIDIA GPU: the streaming runtime hard-rejects
non-CUDA devices at load time.  The checkpoint must be a *local
directory* — the engine downloads the backbone and text tokenizer from
HuggingFace when given an HF id, but it reads the checkpoint's bundled
`audio_tokenizer/` subdirectory from local disk, so download the full
checkpoint first (see Configuration below).

`seed` is meaningful: the process RNG is re-seeded per request (and the
engine re-seeds PyTorch again at the start of sampling), so the same seed
and inputs reproduce the same sampling start.  Reproducibility on CUDA is
near-exact, not bit-exact (non-deterministic GPU kernels).

Model weights (BreezeBlue/breeze-tts-2) are research / non-commercial; see
the engine repo's LICENSE for the self-hosted-output terms.

Configuration (environment variables):
    BREEZEBLUE_MODEL       Local directory of the Breeze TTS 2 checkpoint.
                           Download it from HuggingFace first:
                               huggingface-cli download BreezeBlue/breeze-tts-2 \\
                                   --local-dir /models/breeze-tts-2
                           The directory must contain the bundled
                           `audio_tokenizer/` subdirectory.
                           Default: BreezeBlue/breeze-tts-2
    BREEZEBLUE_DEVICE      CUDA device to load the model on: 'cuda' or
                           'cuda:N' for a specific GPU.  Must be a CUDA
                           device (the streaming runtime has no
                           CPU/MPS backend).
                           Default: cuda
    BREEZEBLUE_FAST_ALL    Enable the fast (CUDA-graph) inference path for
                           all stages: better TTFA/RTF, at the cost of a
                           one-time graph warmup at startup and ~14.4 GiB
                           of GPU memory (vs ~7.7 GiB for eager).  0/1.
                           Default: 0
    BREEZEBLUE_HOST        Bind host for `python server_breezeBlue.py`.
                           Default: 0.0.0.0
    BREEZEBLUE_PORT        Bind port for `python server_breezeBlue.py`.
                           Default: 7500

Extra dependencies beyond the breeze-tts repository:
    git clone https://github.com/breezeblue-ai/breeze-tts.git
    cd breeze-tts && python -m pip install -r requirements.txt
    pip install fastapi uvicorn loguru soundfile
    pip install ../tts-engine-common # in-repo copy; or: pip install -e ../tts-engine-common

The server must be able to import the engine's `breeze_infer` and `models`
packages.  Running `python <path>/server_breezeBlue.py` puts the *script's*
directory on sys.path, not your working directory, so the breeze-tts repo
root must be added explicitly:

    cd breeze-tts
    PYTHONPATH=. BREEZEBLUE_MODEL=/path/to/checkpoint \
        python ../tts-serve/impl/server_breezeBlue.py
"""

from __future__ import annotations

import base64
import io
import math
import os
import random
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# The engine ships as a plain git repo (no PyPI package, no setup.py), so
# its top-level packages are importable only when the repo root is on
# PYTHONPATH.  `python /path/to/server_breezeBlue.py` puts the *script's*
# directory on sys.path, not the working directory, so the most common
# deployment failure mode is a bare ModuleNotFoundError — give it a
# message people can actually act on.
try:
    from breeze_infer.runtime import (
        load_runtime,
        set_all_seeds,
        update_generation_config_for_breeze,
    )
    from breeze_infer.templates import get_template, prepare_inputs, select_template_name
    import models
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
    from models.warmup_profile import load_warmup_profile
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"The breeze-tts engine package {exc.name!r} is not importable.  "
        "The engine repo must be on PYTHONPATH -- e.g. from the breeze-tts "
        "repo root:  PYTHONPATH=. python /path/to/tts-serve/impl/server_breezeBlue.py"
    ) from exc

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

MODEL_NAME_OR_PATH = os.getenv("BREEZEBLUE_MODEL", "BreezeBlue/breeze-tts-2")
DEVICE = os.getenv("BREEZEBLUE_DEVICE", "cuda")


def _parse_fast_all(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


FAST_ALL = _parse_fast_all(os.getenv("BREEZEBLUE_FAST_ALL", "0"))

# The engine's fast-path warmup profile lives next to its `models` package,
# i.e. at the breeze-tts repo root (configs/fast.json).
FAST_CONFIG_PATH = Path(models.__file__).resolve().parents[1] / "configs" / "fast.json"

# Engine constants, mirroring the breeze-tts reference CLI / API
# (infer.py, breeze_infer/api.py).
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1
DEFAULT_CFG_SCALE = 1.0

# The streaming runtime's own `sample_rate` property reads 24 kHz from the
# codec config; the response echoes the runtime's value and this constant
# only feeds the capabilities document.
SAMPLE_RATE = 24000

SEED_MIN = 1
SEED_MAX = 1000

# Heuristic lower bound for usable speaker conditioning (clean speech).
MIN_PROMPT_DURATION_S = 2.0

# Sanity valve for the request payload (~5 min of 24 kHz audio).
MAX_AUDIO_B64_LEN = 10_000_000


def _validate_config() -> None:
    """Fail fast on bad configuration instead of partway through a model load."""
    # The streaming runtime hard-rejects non-CUDA devices at load time; the
    # prefix (rather than an exact 'cuda') also allows multi-GPU selection
    # like 'cuda:1'.
    if not DEVICE.startswith("cuda"):
        raise ValueError(
            "BREEZEBLUE_DEVICE must be a CUDA device ('cuda' or 'cuda:N'); "
            f"Breeze TTS 2's streaming runtime has no CPU/MPS backend, got {DEVICE!r}"
        )
    if FAST_ALL and not FAST_CONFIG_PATH.is_file():
        # The warmup profile is part of the breeze-tts repo, so a missing
        # file means the engine packages were imported from somewhere
        # unexpected (or a partial checkout).
        raise ValueError(
            "BREEZEBLUE_FAST_ALL is set but the engine's fast-path config "
            f"was not found at {FAST_CONFIG_PATH}; run with the breeze-tts "
            "repo root importable, or leave BREEZEBLUE_FAST_ALL unset"
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
        description=(
            "Text to synthesize, e.g. 'Hello there'.  Inline vocal events "
            "are supported: parentheses in English ('(laugh)', '(sigh)') "
            "and square brackets in Chinese ('[笑]', '[叹气]')."
        ),
    )
    audio_base64: str = Field(
        ...,
        min_length=1,
        max_length=MAX_AUDIO_B64_LEN,
        description=(
            "Reference voice sample (roughly 3-10 s of clean, non-looping "
            "speech) as a base64 string.  Any container soundfile can "
            "decode (WAV, MP3, OGG, FLAC, ...)."
        ),
    )
    reference_text: str = Field(
        ...,
        min_length=1,
        description=(
            "Exact transcript of the reference audio.  Required: both "
            "modes condition on the clip and its transcript together, so "
            "the words must match the clip exactly (include repetitions "
            "if the speech is repeated in the audio)."
        ),
    )
    language: str | None = Field(
        DEFAULT_LANGUAGE,
        description=(
            "Two-letter language code, e.g. 'en' or 'zh'.  Accepted for "
            "API consistency but not forwarded — the engine has no "
            "language parameter; it auto-detects from the input text "
            "(English and Chinese supported).  Omitted or empty defaults "
            "to 'en'."
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
    instruction: str | None = Field(
        None,
        min_length=1,
        description=(
            "Natural-language voice direction instruction (tone, emotion, "
            "pace, delivery), e.g. 'Speak slowly with a restrained, "
            "serious tone'.  Omit for plain voice cloning; providing one "
            "switches the engine to voice-direction mode (same speaker, "
            "steered delivery).  Match the instruction language to the "
            "target text."
        ),
    )
    cfg_scale: float = Field(
        DEFAULT_CFG_SCALE,
        gt=0.0,
        description=(
            "Classifier-free guidance scale.  Engine default is "
            f"{DEFAULT_CFG_SCALE:.1f}; the engine README recommends 4 to "
            "strengthen instruction-following for voice direction.  Must "
            "be finite and > 0.  A non-default value requires "
            "`instruction`: the CFG branch only exists in voice-direction "
            "mode, so plain voice cloning only runs at 1.0."
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

    @field_validator("instruction")
    @classmethod
    def _validate_instruction(cls, v: str | None) -> str | None:
        # A provided-but-blank instruction would silently switch the
        # engine into plain-clone mode; reject it loudly (an explicit
        # null is the way to say 'no instruction').
        if v is not None and not v.strip():
            raise ValueError("instruction must contain non-whitespace characters")
        return v

    @field_validator("cfg_scale")
    @classmethod
    def _check_cfg_scale(cls, v: float) -> float:
        # gt=0.0 rejects 0/negative values but passes +inf; the engine's
        # own CLI rejects any non-finite scale, so do the same.
        if not math.isfinite(v):
            raise ValueError("cfg_scale must be a finite number")
        return v

    @model_validator(mode="after")
    def _check_cfg_scale_requires_instruction(self) -> "SynthesisRequest":
        # The engine's CFG branch only exists on the voice-direction
        # template (ref_edit_tata has a negative branch; ref_clone_tata
        # does not), so a non-default scale without an instruction would
        # fail deep inside prepare_inputs() as a 500.  Reject the
        # combination at the boundary instead (422, like the other
        # client-side configuration errors on this server).
        if self.cfg_scale != 1.0 and self.instruction is None:
            raise ValueError(
                "cfg_scale != 1.0 requires an instruction (voice-direction "
                "mode); plain voice cloning only runs at cfg_scale = 1.0"
            )
        return self

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
        # 'auto' sentinel either — accept any well-formed code for API
        # consistency (LuxTTS no-support case) and do not forward it.
        return validate_language_code(v)


class SynthesisResponse(CoreSynthesisResponse):
    """The synthesis result (core fields from tts_engine_common, plus fid)."""

    fid: str = Field(..., description="Request ID (internal).")


class HealthResponse(BaseModel):
    """Health / readiness check."""

    status: Literal["ok"] = "ok"
    serverType: Literal["breeze-tts-2"] = "breeze-tts-2"
    model: str = MODEL_NAME_OR_PATH
    device: str = DEVICE


# ---------------------------------------------------------------------------
# Capabilities (derived from SynthesisRequest — single source of truth)
# ---------------------------------------------------------------------------

CAPABILITIES = build_capabilities(
    SynthesisRequest,
    engine="breeze-tts-2",
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
            "The reference clip and its exact transcript (reference_text) "
            "drive both modes; use clean, non-looping speech with minimal "
            "background noise.  Without `instruction` the voice is cloned "
            "as-is; with one, the speaker's identity is kept while the "
            "instruction steers tone, emotion, pace, and delivery."
        ),
    },
    languages=None,  # no fixed list: the engine auto-detects en/zh from text (docs/02)
    overrides={
        "cfg_scale": {"step": 0.1},
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
        # Drop every engine reference: the streaming runtime holds its own
        # copy of the model, so `del _runtime.model` alone would not free it.
        del _runtime.model
        del _runtime.audio_tokenizer
        del _runtime.tokenizer
        del _runtime.streaming
        _runtime = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded and CUDA cache cleared.")


app = FastAPI(
    title="Breeze TTS 2 Voice Cloning API",
    description=(
        "REST API around Breeze TTS 2.  Send text + a reference audio "
        "sample + its exact transcript (and optionally a direction "
        "instruction) and get back cloned speech.  Synthesis requests are "
        "serialized (single shared model).  Machine-readable parameter "
        "metadata at GET /capabilities."
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
# Runtime — thin wrapper around the Breeze TTS 2 inference runtime
# ---------------------------------------------------------------------------


@dataclass
class BreezeBlueRuntime:
    """Holds the loaded engine and its metadata for the lifetime of the server."""

    tokenizer: object
    model: object
    audio_tokenizer: object
    streaming: FastBreezeStreamingRuntime
    sample_rate: int
    device: str


_runtime: BreezeBlueRuntime | None = None

# The streaming runtime mutates shared state per call (KV caches, the
# streaming codec's per-request state, and the process-wide RNG via
# set_all_seeds) and is designed for single-request use — the engine's own
# API serializes every request behind a lock.  Do the same: concurrent
# requests would stomp on each other's codec/KV state.
_synthesis_lock = threading.Lock()


def _get_runtime() -> BreezeBlueRuntime:
    """Return the global runtime, loading the model once on first call."""
    global _runtime
    if _runtime is None:
        logger.info(
            "Loading Breeze TTS 2 model '{}' on device '{}' (fast_all={}) ...",
            MODEL_NAME_OR_PATH,
            DEVICE,
            FAST_ALL,
        )
        tokenizer, model, audio_tokenizer = load_runtime(
            Path(MODEL_NAME_OR_PATH),
            device=DEVICE,
            attn_implementation="eager",
        )
        # Applies the engine's own generation defaults (backbone and depth-
        # decoder sampling parameters); required before iter_audio_chunks.
        update_generation_config_for_breeze(model)
        config = FastStreamingConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            max_seq_len=MAX_SEQ_LEN,
            # None (not False) keeps the per-stage flags authoritative — we
            # expose only the master switch, all of which are False.
            fast_all=FAST_ALL or None,
            repetition_penalty=REPETITION_PENALTY,
        )
        streaming = FastBreezeStreamingRuntime(
            model, audio_tokenizer, config, tokenizer=tokenizer
        )
        if streaming.fast_enabled:
            # One-time CUDA-graph warmup from the engine's own profile, so
            # the first real request does not pay the capture cost.
            profile = load_warmup_profile(FAST_CONFIG_PATH)
            profile = replace(profile, codec_chunk_frames=streaming.codec_chunk_frames)
            manifest = streaming.warmup_from_profile(profile)
            logger.info(
                "Fast-path warmup complete: {:.2f} ms",
                manifest["total_elapsed_ms"],
            )
        _runtime = BreezeBlueRuntime(
            tokenizer=tokenizer,
            model=model,
            audio_tokenizer=audio_tokenizer,
            streaming=streaming,
            # Echo the rate the engine itself reports (24 kHz from the codec
            # config) rather than a hardcoded constant, so a future variant
            # stays self-consistent.
            sample_rate=streaming.sample_rate,
            device=DEVICE,
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
    <head><title>Breeze TTS 2 REST API</title></head>
    <body>
        <h1>Breeze TTS 2 Voice Cloning REST API</h1>
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
    summary="Synthesize speech (plain voice cloning or voice direction)",
)
def synthesize(req: SynthesisRequest) -> SynthesisResponse:
    """
    Synthesize audio from text + reference audio + its exact transcript.

    Omitting `instruction` gives plain voice cloning; providing it switches
    to voice direction (same speaker, steered delivery).  The engine's
    voice-design mode is deliberately not exposed.

    The full parameter list is documented at GET /capabilities; the request
    schema mirrors it exactly (same model, no drift).
    """
    runtime = _get_runtime()

    # Resolve randomised seed for reproducibility.
    seed = req.seed if req.seed is not None else random.randint(SEED_MIN, SEED_MAX)
    fid = str(uuid.uuid4())

    # A provided instruction (already validated non-blank) switches the
    # engine from voice-clone to voice-direction mode; strip it, like the
    # engine's own API does.
    instruction = req.instruction.strip() if req.instruction else None

    logger.info(
        "Synthesizing: seed={}, mode={}, text_len={}, ref_text_len={}, "
        "cfg={:.1f}, lang={} (not forwarded)",
        seed,
        "voice-direction" if instruction is not None else "voice-clone",
        len(req.text),
        len(req.reference_text),
        req.cfg_scale,
        req.language,
    )

    # Decode and sanity-check the reference audio before touching the model.
    try:
        raw_audio = decode_base64(req.audio_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")

    _check_reference_audio(raw_audio)

    # The engine loads reference audio from a file path; it re-encodes the
    # clip per request (no path-keyed cache), so a UUID temp file deleted
    # per request is fine.
    ref_audio_path = write_temp_audio(raw_audio, _TEMP_AUDIO_DIR)

    # The engine's request dict; the template is selected from which fields
    # are present (ref_clone_tata vs ref_edit_tata — the reference is always
    # present on this server, so only the instruction matters).
    request = {
        "id": fid,
        "text": req.text,
        "speaker": "S0",
        "ref_audio_path": ref_audio_path,
        "ref_text": req.reference_text.strip(),
    }
    if instruction is not None:
        request["instruction"] = instruction
    template_name = select_template_name(request)

    try:
        t0 = time.perf_counter()

        with _synthesis_lock:
            # Seed the process-wide RNG before input preparation, and pass
            # the same seed to iter_audio_chunks, which re-seeds PyTorch at
            # the start of sampling — mirroring the engine's own API, where
            # each request gets an isolated, reproducible sampling start.
            set_all_seeds(seed)
            inputs = prepare_inputs(
                runtime.tokenizer,
                runtime.audio_tokenizer,
                runtime.model,
                [request],
                get_template(template_name),
                guidance_scale=req.cfg_scale,
                guidance_scale_ref=None,
                guidance_scale_ins=None,
            )
            chunks = list(
                runtime.streaming.iter_audio_chunks(
                    inputs, request_id=fid, seed=seed
                )
            )

        time_used = time.perf_counter() - t0

        if not chunks:
            # Degenerate output (e.g. the model emitted only the terminal
            # pad frame); an empty audio payload is worse than a clear 500.
            raise HTTPException(
                status_code=500,
                detail=(
                    "Synthesis produced no audio; try a different seed, "
                    "shorter text, or a different reference clip."
                ),
            )

        audio = chunks[0].audio
        if len(chunks) > 1:
            audio = np.concatenate([chunk.audio for chunk in chunks])

        sample_rate = runtime.sample_rate
        rtf = compute_rtf(time_used, len(audio), sample_rate)

        audio_bytes = _numpy_to_wav_bytes(audio, sample_rate)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        audio_duration = len(audio) / sample_rate if sample_rate else 0.0
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
            fid=fid,
            time_used=time_used,
            rtf=rtf,
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

_TEMP_AUDIO_DIR = temp_audio_dir("breeze_blue_rest_api")


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
# Main (for running directly: python server_breezeBlue.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("BREEZEBLUE_HOST", "0.0.0.0")
    port = int(os.getenv("BREEZEBLUE_PORT", "7500"))
    logger.info("Starting Breeze TTS 2 REST API server on %s:%d", host, port)
    # Pass the app object directly instead of a module path string,
    # so this works regardless of how the file is invoked.
    uvicorn.run(app, host=host, port=port, log_level="info")
