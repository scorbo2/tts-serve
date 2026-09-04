"""Pydantic models for the /capabilities document and the frozen response core."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .core import SCHEMA_VERSION


class ParamSpec(BaseModel):
    """Machine-readable metadata for a single request parameter.

    The ``type`` values mirror the JSON Schema types the Pydantic model emits
    for the supported field kinds, so a client app can drive input rendering
    (checkbox, number spinner, select, text) straight from this document.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    type: Literal["string", "integer", "number", "boolean"]
    required: bool = Field(
        False,
        description=(
            "Whether the request must supply this field. Derived from the "
            "presence of a default in the Pydantic model, so it cannot drift."
        ),
    )
    default: Any = Field(None, description="Default value, when not required. null means 'no default'.")
    description: str = ""
    min: float | None = Field(None, description="Inclusive lower bound (number/integer only).")
    max: float | None = Field(None, description="Inclusive upper bound (number/integer only).")
    step: float | None = Field(None, description="Suggested UI step for number inputs.")
    enum: list[str] | None = Field(None, description="Allowed string values (enum/Literal fields).")
    min_length: int | None = Field(None, description="Minimum string length.")
    max_length: int | None = Field(None, description="Maximum string length.")
    group: Literal["common", "engine"] = Field(
        "engine",
        description="'common' fields come from the shared vocabulary; 'engine' fields are engine-specific.",
    )
    advanced: bool = Field(False, description="UI hint: tuck this parameter into an advanced section.")


class ReferenceAudioSpec(BaseModel):
    """What the engine wants to hear as the reference clip (None = no cloning)."""

    model_config = ConfigDict(extra="forbid")

    required: bool = Field(..., description="Whether a reference audio is mandatory for synthesis.")
    formats: list[str] = Field(default_factory=list, description="Accepted container/codec names (e.g. 'wav', 'mp3').")
    min_duration_s: float | None = Field(None, description="Recommended/required minimum clip duration in seconds.")
    max_duration_s: float | None = Field(None, description="Recommended/required maximum clip duration in seconds.")
    note: str = Field("", description="Human-readable guidance (truncation, quality, etc.).")


class Capabilities(BaseModel):
    """The GET /capabilities document for one engine server."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(SCHEMA_VERSION, description="Bump only on breaking document changes.")
    engine: str = Field(..., min_length=1, description="Stable engine slug, e.g. 'chatterbox'.")
    model: str = Field(..., min_length=1, description="Model identifier/checkpoint in use.")
    device: str = Field(..., min_length=1, description="Inference device, e.g. 'cuda', 'mps', 'cpu'.")
    sample_rate: int = Field(..., gt=0, description="Audio output sample rate in Hz.")
    watermarked: bool = Field(False, description="Whether output audio carries a watermark.")
    endpoint: str = Field("/synthesize", description="POST path for synthesis requests.")
    reference_audio: ReferenceAudioSpec | None = Field(None, description="Reference clip requirements, if the engine clones.")
    languages: list[str] | None = Field(
        None,
        description="Supported language codes, or null when the engine is language-agnostic / accepts free-form.",
    )
    parameters: list[ParamSpec] = Field(default_factory=list)


def _is_nonfinite(value: Any) -> bool:
    return isinstance(value, float) and (math.isinf(value) or math.isnan(value))


class CoreSynthesisResponse(BaseModel):
    """Frozen response core shared by every engine server.

    Engine servers subclass this to add extras (e.g. ``fid``, ``num_steps``);
    the core fields and their types never change without a schema_version bump.

    ``rtf`` and ``time_used`` are sanitized because engines can emit inf/nan on
    degenerate inputs and JSON cannot represent them (inf/nan -> null, nan -> 0.0).
    """

    audio_base64: str = Field(..., description="Generated WAV audio (PCM_16) as base64.")
    sample_rate: int = Field(..., gt=0, description="Audio sample rate in Hz.")
    seed: int = Field(..., description="Seed actually used for this generation.")
    time_used: float = Field(..., ge=0, description="Wall-clock generation time in seconds.")
    rtf: float | None = Field(
        ...,
        description="Real-time factor (time_used / audio duration); null when the audio duration is zero.",
    )

    @field_validator("rtf", mode="before")
    @classmethod
    def _sanitize_rtf(cls, value: Any) -> Any:
        return None if _is_nonfinite(value) else value

    @field_validator("time_used", mode="before")
    @classmethod
    def _sanitize_time_used(cls, value: Any) -> Any:
        return 0.0 if _is_nonfinite(value) else value
