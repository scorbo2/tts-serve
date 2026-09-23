"""
FastAPI REST server for Qwen3-TTS voice cloning via MLX (Apple Silicon).

This is the MLX-native counterpart to ``server_qwen3TTS.py`` (PyTorch/CUDA/
MPS), built for a controlled A/B comparison between the two backends on the
same Mac.  It wraps the same Qwen3-TTS Base checkpoint family through
``mlx-audio`` instead of ``qwen_tts``/PyTorch.

Loads the model once on startup, then exposes a single POST endpoint for
synthesis.  Clients send text, a reference audio sample (base64), and its
exact transcript; the server returns the generated audio as base64-encoded
24 kHz WAV.  Only ICL (in-context learning) voice cloning is exposed in this
first version: ``reference_text`` is required, and there is no
speaker-embedding-only fallback, streaming, or voice-library / preset-voice
modes.  (See ``server_qwen3TTS.py`` for those.)

Optional long-text chunking is supported: when ``chunking_enabled`` is set,
text is split into smaller chunks (``chunk_min_chars``/``chunk_max_chars``)
that are synthesized independently and then joined, either with silence
(``chunk_silence_ms``) or a short linear crossfade (``chunk_crossfade_ms``,
mutually exclusive with silence) at each join between chunks.

Capabilities: GET /capabilities returns a machine-readable description of
every request parameter, derived from the Pydantic request model so it can
never drift from what the server actually validates (see the
tts-engine-common README).

Model weights are downloaded from HuggingFace on first start unless a local
path is given.  The language list below is the Base checkpoint's (10
languages + auto) -- identical to server_qwen3TTS.py's table, since both
wrap the same Qwen3-TTS Base checkpoint family; the engine itself validates
against its own config.

Device: mlx-audio has no device-selection knob (MLX picks its own compute
backend, Metal on Apple Silicon) so there is no ``*_DEVICE`` environment
variable here.  This server reports device as the literal string ``"mlx"``
everywhere (health check, capabilities, logging) -- chosen over "metal"
because "mlx" names the framework actually in use, and mlx-audio itself has
no notion of "metal" as a selectable device value.

Configuration (environment variables):
    QWEN3TTS_MLX_MODEL   HuggingFace id or local path.
                        Default: mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit
    QWEN3TTS_MLX_HOST    Bind host for `python server_qwen3TTS_mlx.py`.
                        Default: 0.0.0.0
    QWEN3TTS_MLX_PORT    Bind port for `python server_qwen3TTS_mlx.py`.
                        Default: 7500

Extra dependencies beyond the mlx-audio package:
    pip install fastapi uvicorn loguru soundfile
    pip install ../tts-engine-common # in-repo copy; or: pip install -e ../tts-engine-common

Usage:
    python server_qwen3TTS_mlx.py
    # or: uvicorn server_qwen3TTS_mlx:app --host 0.0.0.0 --port 7500
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
from typing import Any, Literal

import mlx.core as mx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from mlx_audio.tts.utils import load_model
from mlx_audio.utils import resample_audio
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from tts_engine_common import (
    DEFAULT_LANGUAGE,
    CoreSynthesisResponse,
    build_capabilities,
    capabilities_endpoint,
    compute_rtf,
    decode_base64,
    normalize_language,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME_OR_PATH = os.getenv(
    "QWEN3TTS_MLX_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
)

# No device env var: mlx-audio has no device-selection argument, and MLX
# itself auto-selects its compute backend (Metal on Apple Silicon).  "mlx" is
# the string reported everywhere a device value is expected.
DEVICE = "mlx"

# Verified against the installed mlx-audio source: Qwen3TTSModel.ModelConfig
# defaults sample_rate to 24000, and the target checkpoint's config.json does
# not override it (no "sample_rate" key present) -- same rate as
# server_qwen3TTS.py's PyTorch path.  The actual value used for each response
# still comes from the loaded model's own `.sample_rate` property at
# synthesis time (see synthesize()), never assumed.
SAMPLE_RATE = 24000

SEED_MIN = 1
SEED_MAX = 1000

# Heuristic lower bound: below this the reference codes degrade to
# near-garbage. Mirrors server_qwen3TTS.py's MIN_REF_DURATION_S.
MIN_REF_DURATION_S = 2.0

# Sanity valve for the request payload (~5 min of 24 kHz audio).
MAX_AUDIO_B64_LEN = 10_000_000

# Identical to server_qwen3TTS.py's table: both servers wrap the same
# Qwen3-TTS Base checkpoint family, and mlx-audio's Qwen3TTS Model.generate()
# takes the same lowercase language *names* via its `lang_code` argument
# (verified in mlx_audio/tts/models/qwen3_tts/qwen3_tts.py -- `lang_code` is
# threaded straight through as the `language=` argument of
# `_generate_icl()`/`_prepare_generation_inputs()`, i.e. it is the target
# synthesis language, not a source-language hint).
#
# The API contract is two-letter codes (docs/02-language-handling.md); this
# table is this server's internal code -> engine-name mapping.
LANGUAGE_CODE_TO_NAME = {
    "zh": "chinese",
    "en": "english",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "pt": "portuguese",
    "ru": "russian",
    "es": "spanish",
}
LANGUAGE_CODES = tuple(LANGUAGE_CODE_TO_NAME)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

# Dynamic Literal over the Base checkpoint's language codes (+ the engine's
# 'auto' auto-detection sentinel, mlx-audio's own default for `lang_code`) so
# the request schema and the /capabilities enum share one source.
Language = Literal[("auto", *LANGUAGE_CODES)]


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
            "Reference voice sample as a base64 string.  Any container "
            "soundfile can decode (WAV, MP3, OGG, FLAC, ...).  ~3 s is enough "
            "for high-quality cloning."
        ),
    )
    reference_text: str = Field(
        ...,
        min_length=1,
        description=(
            "Exact transcript of the reference audio.  Required: this server "
            "runs ICL (in-context learning) voice cloning only, which "
            "conditions on the reference audio and its transcript together."
        ),
    )
    language: Language | None = Field(
        DEFAULT_LANGUAGE,
        description=(
            "Two-letter language code, e.g. 'en', 'zh', or 'auto' for "
            f"auto-detection (supported: {', '.join(sorted(LANGUAGE_CODES))}).  "
            "Omitted or empty defaults to 'en'."
        ),
    )
    seed: int | None = Field(
        None,
        ge=SEED_MIN,
        le=SEED_MAX,
        description=(
            "Random seed for reproducibility (seeds MLX's global RNG via "
            "mx.random.seed(), which drives the talker's token sampling). "
            f"If omitted, a random seed in [{SEED_MIN}, {SEED_MAX}] is chosen "
            "and echoed in the response."
        ),
    )

    # --- optional long-text chunking ----------------------------------------
    chunking_enabled: bool = Field(
        False,
        description=(
            "Split synthesis text into smaller chunks before generation. "
            "Disabled by default."
        ),
    )
    chunk_min_chars: int = Field(
        250,
        ge=1,
        description="Preferred minimum text chunk size when chunking is enabled.",
    )
    chunk_max_chars: int = Field(
        500,
        ge=1,
        description="Maximum text chunk size when chunking is enabled.",
    )
    chunk_silence_ms: int = Field(
        0,
        ge=0,
        description=(
            "Silence inserted between generated text chunks, in milliseconds. "
            "Default is 0."
        ),
    )
    chunk_crossfade_ms: int = Field(
        0,
        ge=0,
        le=50,
        description=(
            "Linear crossfade applied between generated text chunks, in "
            "milliseconds (0-50). Mutually exclusive with chunk_silence_ms. "
            "Default is 0 (no crossfade)."
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
        # ICL mode feeds the transcript straight into the model's context; a
        # blank one would mis-condition the clone, so reject it loudly.
        if not v.strip():
            raise ValueError("reference_text must contain non-whitespace characters")
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

    @model_validator(mode="after")
    def _validate_chunk_bounds(self):
        if self.chunk_min_chars > self.chunk_max_chars:
            raise ValueError(
                "chunk_min_chars must be less than or equal to chunk_max_chars"
            )
        return self

    @model_validator(mode="after")
    def _validate_chunk_join_mode(self):
        if self.chunk_silence_ms > 0 and self.chunk_crossfade_ms > 0:
            raise ValueError(
                "chunk_silence_ms and chunk_crossfade_ms are mutually "
                "exclusive; set at most one of them above 0"
            )
        return self


class SynthesisResponse(CoreSynthesisResponse):
    """The synthesis result (core fields from tts_engine_common, plus fid)."""

    fid: str = Field(..., description="Request ID (internal).")


class HealthResponse(BaseModel):
    """Health / readiness check."""

    status: Literal["ok"] = "ok"
    serverType: Literal["Qwen3-TTS-MLX"] = "Qwen3-TTS-MLX"
    model: str = MODEL_NAME_OR_PATH
    device: str = DEVICE


# ---------------------------------------------------------------------------
# Capabilities (derived from SynthesisRequest — single source of truth)
# ---------------------------------------------------------------------------

CAPABILITIES = build_capabilities(
    SynthesisRequest,
    engine="qwen3-tts-mlx",
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
            "reference_text (the exact transcript) is required: this server "
            "runs ICL voice cloning only, conditioning on the reference "
            "audio and transcript together. ~3 s is enough for "
            "high-quality cloning."
        ),
    },
    languages=sorted(LANGUAGE_CODES),
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
    mx.clear_cache()
    logger.info("Model unloaded and MLX cache cleared.")


app = FastAPI(
    title="Qwen3-TTS MLX Voice Cloning API",
    description=(
        "REST API around Qwen3-TTS via mlx-audio (Apple Silicon native). "
        "Send text + a reference audio sample + its transcript and get back "
        "cloned speech.  Machine-readable parameter metadata at "
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
# Runtime — thin wrapper around the mlx-audio Qwen3-TTS model
# ---------------------------------------------------------------------------


@dataclass
class Qwen3TTSMLXRuntime:
    """Holds the loaded model and its metadata for the lifetime of the server."""

    model: Any
    device: str
    sample_rate: int


_runtime: Qwen3TTSMLXRuntime | None = None

# mlx-audio's Model keeps an internal ICL cache (self._icl_cache) that is
# mutated during generation; serialize synthesis so concurrent requests don't
# race on shared model state (mirrors the *_synthesis_lock pattern used by
# the other engine servers with shared mutable model state).
_synthesis_lock = threading.Lock()


def _get_runtime() -> Qwen3TTSMLXRuntime:
    """Return the global runtime, loading the model once on first call."""
    global _runtime
    if _runtime is None:
        logger.info("Loading Qwen3-TTS MLX model '{}' ...", MODEL_NAME_OR_PATH)
        model = load_model(MODEL_NAME_OR_PATH)
        _runtime = Qwen3TTSMLXRuntime(
            model=model, device=DEVICE, sample_rate=model.sample_rate
        )
        logger.info("Model loaded successfully (sample_rate={} Hz).", model.sample_rate)
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
    <head><title>Qwen3-TTS MLX REST API</title></head>
    <body>
        <h1>Qwen3-TTS MLX Voice Cloning REST API</h1>
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
    Synthesize audio using the provided text, reference audio, and transcript.

    Qwen3-TTS (via mlx-audio) performs zero-shot voice cloning in ICL mode: a
    short reference audio clip and its exact transcript are kept in the
    model's context, and new speech is generated in the same voice.

    When chunking is enabled, long text is split into smaller text chunks and
    each chunk is synthesized independently with the same reference voice,
    language, and seed sequence. Generated audio is concatenated in order,
    optionally with silence inserted between text chunks.

    The full parameter list is documented at GET /capabilities; the request
    schema mirrors it exactly (same model, no drift).
    """
    runtime = _get_runtime()

    # Resolve randomised seed for reproducibility.
    seed = req.seed if req.seed is not None else random.randint(SEED_MIN, SEED_MAX)

    # docs/02: the API speaks two-letter codes; the engine wants lowercase
    # names. 'auto' passes through (the engine's own auto-detection mode).
    engine_language = LANGUAGE_CODE_TO_NAME.get(req.language, req.language)

    logger.info(
        "Synthesizing: seed={}, text_len={}, ref_text_len={}, " "lang={} (engine: {})",
        seed,
        len(req.text),
        len(req.reference_text),
        req.language,
        engine_language,
    )

    try:
        raw_audio = decode_base64(req.audio_base64)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 audio: {exc}",
        )

    _check_reference_audio(raw_audio)

    try:
        ref_wav, ref_sr = _decode_wav(raw_audio)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode reference audio: {exc}",
        )

    ref_audio_for_model = _prepare_ref_audio(
        ref_wav,
        ref_sr,
        runtime.sample_rate,
    )

    try:
        t0 = time.perf_counter()

        text_chunks = (
            _split_text_chunks(
                req.text,
                req.chunk_min_chars,
                req.chunk_max_chars,
            )
            if req.chunking_enabled
            else [req.text]
        )

        logger.info(
            "Chunking: enabled={}, chunks={}, min_chars={}, max_chars={}, "
            "silence_ms={}, crossfade_ms={}",
            req.chunking_enabled,
            len(text_chunks),
            req.chunk_min_chars,
            req.chunk_max_chars,
            req.chunk_silence_ms,
            req.chunk_crossfade_ms,
        )

        with _synthesis_lock:
            mx.random.seed(seed)

            audio_chunks: list[np.ndarray] = []
            sr = runtime.sample_rate

            for chunk_index, text_chunk in enumerate(text_chunks, start=1):
                generated_for_chunk: list[np.ndarray] = []

                logger.debug(
                    "Generating text chunk {}/{}: {} chars",
                    chunk_index,
                    len(text_chunks),
                    len(text_chunk),
                )

                for result in runtime.model.generate(
                    text=text_chunk,
                    lang_code=engine_language,
                    ref_audio=ref_audio_for_model,
                    ref_text=req.reference_text,
                    stream=False,
                ):
                    mx.eval(result.audio)

                    generated_for_chunk.append(
                        np.asarray(
                            result.audio,
                            dtype=np.float32,
                        ).reshape(-1)
                    )

                    sr = result.sample_rate

                if not generated_for_chunk:
                    logger.warning(
                        "Synthesis produced no audio for chunk {}/{}: "
                        "seed={}, lang={} (engine: {})",
                        chunk_index,
                        len(text_chunks),
                        seed,
                        req.language,
                        engine_language,
                    )

                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "The model produced no audio for one of the supplied "
                            "text chunks."
                        ),
                    )

                chunk_audio = (
                    np.concatenate(generated_for_chunk)
                    if len(generated_for_chunk) > 1
                    else generated_for_chunk[0]
                )

                audio_chunks.append(chunk_audio)

        time_used = time.perf_counter() - t0

        if len(audio_chunks) == 1:
            wav = audio_chunks[0]
        elif req.chunk_crossfade_ms > 0:
            # Mutually exclusive with silence (enforced by
            # _validate_chunk_join_mode), so no silence gap here.
            wav = _crossfade_audio_chunks(
                audio_chunks,
                sr,
                req.chunk_crossfade_ms,
            )
        else:
            silence_samples = int(sr * req.chunk_silence_ms / 1000)

            silence = (
                np.zeros(
                    silence_samples,
                    dtype=np.float32,
                )
                if silence_samples > 0
                else None
            )

            parts: list[np.ndarray] = []

            for index, chunk_audio in enumerate(audio_chunks):
                if index > 0 and silence is not None:
                    parts.append(silence)

                parts.append(chunk_audio)

            wav = np.concatenate(parts)

        rtf = compute_rtf(
            time_used,
            len(wav),
            sr,
        )

        audio_bytes = _numpy_to_wav_bytes(
            wav,
            sr,
        )

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

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(
            "Synthesis failed: {}",
            exc,
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_text_chunks(
    text: str,
    min_chars: int,
    max_chars: int,
) -> list[str]:
    """Split text into chunks, preferring natural punctuation boundaries."""
    text = text.strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        split_at = -1

        # Prefer sentence boundaries.
        for marker in (". ", "! ", "? "):
            pos = window.rfind(marker)
            if pos >= min_chars:
                split_at = max(split_at, pos + 1)

        # Then softer punctuation.
        if split_at == -1:
            for marker in ("; ", ": ", ", "):
                pos = window.rfind(marker)
                if pos >= min_chars:
                    split_at = max(split_at, pos + 1)

        # Then a word boundary.
        if split_at == -1:
            pos = window.rfind(" ")
            if pos >= min_chars:
                split_at = pos

        # Final fallback for pathological long tokens / no whitespace.
        if split_at == -1:
            split_at = max_chars

        chunk = remaining[:split_at].strip()
        if chunk:
            chunks.append(chunk)

        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


def _crossfade_audio_chunks(
    chunks: list[np.ndarray],
    sample_rate: int,
    crossfade_ms: int,
) -> np.ndarray:
    """Join intentional-chunk waveforms with a short linear crossfade.

    Only called between completed intentional-chunk waveforms (never between
    the generator sub-results that make up a single chunk -- those are
    concatenated normally before this runs). The overlap length is derived
    from ``crossfade_ms`` and the actual output sample rate, then clamped per
    join to the shorter of the audio available on either side, so a short
    chunk can never produce a negative-length or amplified overlap region.
    """
    result = np.asarray(chunks[0], dtype=np.float32)
    requested_overlap = round(sample_rate * crossfade_ms / 1000)

    for next_chunk in chunks[1:]:
        next_chunk = np.asarray(next_chunk, dtype=np.float32)
        overlap = min(requested_overlap, len(result), len(next_chunk))

        if overlap <= 0:
            result = np.concatenate([result, next_chunk])
            continue

        fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        blended = result[-overlap:] * fade_out + next_chunk[:overlap] * fade_in

        result = np.concatenate([result[:-overlap], blended, next_chunk[overlap:]])

    return result


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


def _decode_wav(raw_bytes: bytes) -> tuple[np.ndarray, int]:
    """Full decode to a mono 1-D float32 waveform plus its sample rate."""
    wav, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    if wav.ndim > 1:  # multi-channel -> mono (the engine expects 1-D)
        wav = wav.mean(axis=1)
    return wav, sr


def _prepare_ref_audio(ref_wav: np.ndarray, ref_sr: int, sample_rate: int):
    """Convert a decoded mono float32 waveform into an mx.array at the
    model's sample rate, resampling first if necessary.
    """
    arr = np.asarray(ref_wav, dtype=np.float32)
    if arr.ndim > 1:  # _decode_wav already returns mono; defensive only
        arr = arr.reshape(-1)

    if ref_sr != sample_rate:
        arr = resample_audio(arr, ref_sr, sample_rate)

    return mx.array(arr)


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
# Main (for running directly: python server_qwen3TTS_mlx.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("QWEN3TTS_MLX_HOST", "0.0.0.0")
    port = int(os.getenv("QWEN3TTS_MLX_PORT", "7500"))
    logger.info("Starting Qwen3-TTS MLX REST API server on %s:%d", host, port)
    # Pass the app object directly instead of a module path string,
    # so this works regardless of how the file is invoked.
    uvicorn.run(app, host=host, port=port, log_level="info")
