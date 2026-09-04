"""Derive the /capabilities document from a Pydantic request model.

The request model is the single source of truth (design decision D4): the
exact object that validates POST /synthesize produces the discovery document,
so the two can never drift. A small per-server override map layers on the UI
sugar the JSON schema cannot express (step, advanced, ...).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from .core import CORE_FIELDS
from .models import Capabilities, ParamSpec, ReferenceAudioSpec

# JSON Schema types we understand. Anything else is a loud error (R1: never
# silently emit a capabilities document that misdescribes validation).
_SUPPORTED_TYPES = ("string", "integer", "number", "boolean")

# Override keys that would lie about what the request model actually accepts.
_FORBIDDEN_OVERRIDE_KEYS = ("name", "type")


class DerivationError(ValueError):
    """A field's JSON schema cannot be mapped to a ParamSpec."""


def _pick_type_branch(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve the non-null branch of an anyOf/oneOf union.

    Pydantic v2 renders ``X | None`` as anyOf [X-schema, {type: null}]; we
    want the X-schema. Anything else (bare types, const, allOf) is passed
    through or rejected, respectively.
    """
    for key in ("anyOf", "oneOf"):
        branches = schema.get(key)
        if not branches:
            continue
        non_null = [b for b in branches if isinstance(b, dict) and b.get("type") != "null"]
        if len(non_null) != 1:
            raise DerivationError(f"Cannot derive a ParamSpec from ambiguous union: {schema!r}")
        return non_null[0]
    if "type" in schema:
        return schema
    raise DerivationError(f"Cannot derive a ParamSpec from unrecognized schema: {schema!r}")


def _bound(branch: dict[str, Any], *keys: str) -> float | None:
    """First numeric bound present, checking inclusive then exclusive keys."""
    for key in keys:
        value = branch.get(key)
        # bool is an int subclass in Python; JSON schema booleans are not numbers.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _enum_values(branch: dict[str, Any]) -> list[str] | None:
    """Enum values as strings (numeric literals are coerced for rendering)."""
    values = branch.get("enum")
    if values is None:
        return None
    return [v if isinstance(v, str) else str(v) for v in values]


def spec_from_schema(name: str, schema: dict[str, Any]) -> ParamSpec:
    """Map one JSON Schema property to a ParamSpec.

    Raises DerivationError (a ValueError subclass) for shapes we do not
    support — deliberately loud rather than guessing.
    """
    branch = _pick_type_branch(schema)
    field_type = branch.get("type")
    if field_type not in _SUPPORTED_TYPES:
        raise DerivationError(f"Unsupported JSON schema type {field_type!r} for field {name!r}")

    # Pydantic emits an explicit "default" only when the field has one; a
    # missing key means the client must supply the field.
    has_default = "default" in schema

    return ParamSpec(
        name=name,
        type=field_type,  # type: ignore[arg-type]  # checked against _SUPPORTED_TYPES above
        required=not has_default,
        default=schema.get("default"),
        description=schema.get("description", ""),
        min=_bound(branch, "minimum", "exclusiveMinimum"),
        max=_bound(branch, "maximum", "exclusiveMaximum"),
        enum=_enum_values(branch),
        min_length=branch.get("minLength"),
        max_length=branch.get("maxLength"),
        group="common" if name in CORE_FIELDS else "engine",
    )


def _apply_overrides(spec: ParamSpec, name: str, overrides: dict[str, dict[str, Any]]) -> ParamSpec:
    """Merge the per-field override map onto a derived spec, validated.

    Invalid values raise ValueError so a misconfigured server fails at import
    time, not when the first client reads /capabilities.
    """
    override = overrides.get(name)
    if not override:
        return spec
    bad = sorted(set(override) & set(_FORBIDDEN_OVERRIDE_KEYS))
    if bad:
        raise ValueError(
            f"Overrides for field {name!r} may not set {bad}; that would "
            f"misdescribe the request model."
        )
    try:
        return ParamSpec(**{**spec.model_dump(), **override})
    except ValidationError as exc:
        raise ValueError(f"Invalid capabilities override for field {name!r}: {exc}") from exc


def build_capabilities(
    request_model: type[BaseModel],
    *,
    engine: str,
    model: str,
    device: str,
    sample_rate: int,
    watermarked: bool,
    endpoint: str = "/synthesize",
    reference_audio: ReferenceAudioSpec | dict[str, Any] | None = None,
    languages: list[str] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> Capabilities:
    """Project a Pydantic request schema into the capabilities document.

    Parameters
    ----------
    request_model:
        The same Pydantic model used to validate POST /synthesize.
    engine / model / device / sample_rate / watermarked / endpoint:
        Static per-server metadata the schema cannot express.
    reference_audio:
        Reference clip requirements (dict or ReferenceAudioSpec); None when the
        engine does not take a reference clip.
    languages:
        Supported language codes, or None when the engine is language-agnostic
        (client passes whatever the model understands).
    overrides:
        Optional ``{field_name: {param_spec_field: value}}`` map for UI sugar
        (step, advanced, group, ...). Keys must be request model fields;
        'name' and 'type' may not be overridden.

    Raises
    ------
    ValueError
        If an override references an unknown field, sets a forbidden key, or
        carries an invalid value.
    DerivationError
        If a field's JSON schema is not one of the supported shapes.
    """
    overrides = overrides or {}
    field_names = set(request_model.model_fields)
    unknown = sorted(set(overrides) - field_names)
    if unknown:
        raise ValueError(
            f"Capabilities overrides reference unknown request fields {unknown}; "
            f"known fields are {sorted(field_names)}"
        )

    properties = request_model.model_json_schema().get("properties", {})

    parameters: list[ParamSpec] = []
    for name in request_model.model_fields:  # preserve declaration order
        spec = spec_from_schema(name, properties[name])
        parameters.append(_apply_overrides(spec, name, overrides))

    if isinstance(reference_audio, dict):
        reference_audio = ReferenceAudioSpec(**reference_audio)

    return Capabilities(
        engine=engine,
        model=model,
        device=device,
        sample_rate=sample_rate,
        watermarked=watermarked,
        endpoint=endpoint,
        reference_audio=reference_audio,
        languages=languages,
        parameters=parameters,
    )
