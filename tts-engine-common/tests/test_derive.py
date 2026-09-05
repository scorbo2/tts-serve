"""Tests for build_capabilities / spec_from_schema — the schema mapping table.

Every shape Pydantic v2 emits for our supported field kinds is covered here,
because a capabilities document that misdescribes validation is worse than no
document at all.
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from tts_engine_common import SCHEMA_VERSION, DerivationError, build_capabilities

METADATA = dict(
    engine="test-engine",
    model="test-model",
    device="cuda",
    sample_rate=24000,
    watermarked=False,
)


def _spec(doc, name):
    return next(p for p in doc.parameters if p.name == name)


class TestIntegerFields:
    def test_derive_int_requiredWithBounds_integerRequiredWithMinMax(self) -> None:
        # GIVEN a required int field with inclusive bounds:
        class Request(BaseModel):
            seed: int = Field(..., ge=1, le=1000)

        # WHEN derived:
        doc = build_capabilities(Request, **METADATA)
        spec = _spec(doc, "seed")

        # THEN it maps to a required integer with the same bounds and no default:
        assert spec.type == "integer"
        assert spec.required is True
        assert spec.default is None
        assert spec.min == 1
        assert spec.max == 1000

    def test_derive_intOptionalWithBounds_integerNotRequiredWithDefaultNull(self) -> None:
        # GIVEN an optional int field (X | None) with bounds:
        class Request(BaseModel):
            seed: int | None = Field(None, ge=1, le=1000)

        # WHEN derived (the bounds live on the non-null anyOf branch):
        doc = build_capabilities(Request, **METADATA)
        spec = _spec(doc, "seed")

        # THEN it maps to a non-required integer whose bounds survived the union:
        assert spec.type == "integer"
        assert spec.required is False
        assert spec.default is None
        assert spec.min == 1
        assert spec.max == 1000


class TestNumberFields:
    def test_derive_float_withDefaultAndBounds_numberNotRequiredWithDefault(self) -> None:
        # GIVEN a float field with a default and inclusive bounds:
        class Request(BaseModel):
            temperature: float = Field(0.8, ge=0.0, le=2.0)

        # WHEN derived:
        doc = build_capabilities(Request, **METADATA)
        spec = _spec(doc, "temperature")

        # THEN the default and bounds are preserved:
        assert spec.type == "number"
        assert spec.required is False
        assert spec.default == 0.8
        assert spec.min == 0.0
        assert spec.max == 2.0

    def test_derive_floatOptional_withBounds_numberNotRequiredWithNullDefault(self) -> None:
        # GIVEN an optional float field (X | None) with bounds and no explicit default:
        class Request(BaseModel):
            top_p: float | None = Field(None, ge=0.0, le=1.0)

        # WHEN derived:
        doc = build_capabilities(Request, **METADATA)
        spec = _spec(doc, "top_p")

        # THEN it is non-required with a null default and the bounds from the anyOf branch:
        assert spec.type == "number"
        assert spec.required is False
        assert spec.default is None
        assert spec.min == 0.0
        assert spec.max == 1.0


class TestStringFields:
    def test_derive_str_requiredWithMinLength_stringRequiredWithMinLength(self) -> None:
        # GIVEN a required string field with a minimum length:
        class Request(BaseModel):
            text: str = Field(..., min_length=1)

        # WHEN derived:
        doc = build_capabilities(Request, **METADATA)
        spec = _spec(doc, "text")

        # THEN length constraints map to min_length (and max_length stays absent):
        assert spec.type == "string"
        assert spec.required is True
        assert spec.min_length == 1
        assert spec.max_length is None

    def test_derive_str_withLengthBounds_stringNotRequiredWithBothLengths(self) -> None:
        # GIVEN a string field with a default and both length bounds:
        class Request(BaseModel):
            audio_base64: str = Field(..., min_length=1, max_length=10_000_000)

        # WHEN derived:
        doc = build_capabilities(Request, **METADATA)
        spec = _spec(doc, "audio_base64")

        # THEN both bounds are preserved:
        assert spec.min_length == 1
        assert spec.max_length == 10_000_000

    def test_derive_strOptionalFreeForm_stringNotRequiredNoEnum(self) -> None:
        # GIVEN an unconstrained optional string field:
        class Request(BaseModel):
            language: str | None = None

        # WHEN derived:
        doc = build_capabilities(Request, **METADATA)
        spec = _spec(doc, "language")

        # THEN there is no enum (free-form) and it is not required:
        assert spec.type == "string"
        assert spec.required is False
        assert spec.enum is None


class TestBooleanFields:
    def test_derive_bool_withDefault_booleanNotRequiredWithDefault(self) -> None:
        # GIVEN a bool field with a default:
        class Request(BaseModel):
            denoise: bool = True

        # WHEN derived:
        doc = build_capabilities(Request, **METADATA)
        spec = _spec(doc, "denoise")

        # THEN the default is preserved:
        assert spec.type == "boolean"
        assert spec.required is False
        assert spec.default is True


class TestLiteralFields:
    def test_derive_literal_required_stringRequiredWithEnum(self) -> None:
        # GIVEN a required Literal field:
        class Request(BaseModel):
            language: Literal["en", "de", "fr"]

        # WHEN derived:
        doc = build_capabilities(Request, **METADATA)
        spec = _spec(doc, "language")

        # THEN the allowed values come through as an enum:
        assert spec.type == "string"
        assert spec.required is True
        assert spec.enum == ["en", "de", "fr"]

    def test_derive_literalOptional_stringNotRequiredWithEnumAndNullDefault(self) -> None:
        # GIVEN an optional Literal field (enum on the non-null anyOf branch):
        class Request(BaseModel):
            language: Literal["en", "de"] | None = None

        # WHEN derived:
        doc = build_capabilities(Request, **METADATA)
        spec = _spec(doc, "language")

        # THEN the enum survives the union and the field is not required:
        assert spec.required is False
        assert spec.default is None
        assert spec.enum == ["en", "de"]

    def test_derive_literalDynamicTuple_stringWithEnumInOrder(self) -> None:
        # GIVEN a Literal built dynamically from a tuple (how servers wire
        # engine-provided language lists):
        codes = ("ar", "da", "de", "en")
        langs = Literal[codes]  # noqa: F821  (3.12 allows a tuple argument)

        class Request(BaseModel):
            language: langs | None = None

        # WHEN derived:
        doc = build_capabilities(Request, **METADATA)
        spec = _spec(doc, "language")

        # THEN all codes are present, in order:
        assert spec.enum == list(codes)


class TestMisc:
    def test_derive_descriptionIsPassedThrough(self) -> None:
        # GIVEN a field with a description:
        class Request(BaseModel):
            seed: int | None = Field(None, ge=1, le=1000, description="Seed for determinism.")

        # WHEN derived:
        doc = build_capabilities(Request, **METADATA)

        # THEN the description text is carried into the document:
        assert _spec(doc, "seed").description == "Seed for determinism."

    def test_derive_parametersPreserveDeclarationOrder(self) -> None:
        # GIVEN fields declared in a specific order:
        class Request(BaseModel):
            zeta: int = 1
            alpha: str = "x"
            mid: bool = False

        # WHEN derived:
        doc = build_capabilities(Request, **METADATA)

        # THEN the parameters keep the declaration order:
        assert [p.name for p in doc.parameters] == ["zeta", "alpha", "mid"]

    def test_derive_arrayField_raisesDerivationError(self) -> None:
        # GIVEN a field whose JSON schema is an array type (unsupported by
        # the ParamSpec contract):
        class Request(BaseModel):
            tags: list[str] = []

        # WHEN derived,
        # THEN it raises loudly instead of emitting a misdescribing document:
        with pytest.raises(DerivationError):
            build_capabilities(Request, **METADATA)


class TestGrouping:
    def test_derive_coreFields_areGroupedCommon(self) -> None:
        # GIVEN a model containing both core and engine-specific fields:
        class Request(BaseModel):
            text: str = Field(..., min_length=1)
            seed: int | None = Field(None, ge=1, le=1000)
            exaggeration: float = Field(0.5, ge=0.0, le=2.0)

        # WHEN derived:
        doc = build_capabilities(Request, **METADATA)

        # THEN core vocabulary fields are 'common' and the rest 'engine':
        assert _spec(doc, "text").group == "common"
        assert _spec(doc, "seed").group == "common"
        assert _spec(doc, "exaggeration").group == "engine"


class TestOverrides:
    def test_overrides_validValues_areApplied(self) -> None:
        # GIVEN a model and an override adding UI sugar:
        class Request(BaseModel):
            exaggeration: float = Field(0.5, ge=0.0, le=2.0)

        # WHEN derived with the override:
        doc = build_capabilities(
            Request, **METADATA, overrides={"exaggeration": {"step": 0.05, "advanced": True}}
        )
        spec = _spec(doc, "exaggeration")

        # THEN the override values are merged onto the derived spec:
        assert spec.step == 0.05
        assert spec.advanced is True

    def test_overrides_unknownField_raisesValueError(self) -> None:
        # GIVEN an override referencing a field the model does not have:
        class Request(BaseModel):
            exaggeration: float = 0.5

        # WHEN derived,
        # THEN it fails fast with a clear message (drift guard):
        with pytest.raises(ValueError, match="unknown request fields"):
            build_capabilities(Request, **METADATA, overrides={"exaggeration_typo": {"step": 1}})

    def test_overrides_forbiddenKey_raisesValueError(self) -> None:
        # GIVEN an override trying to rename or retype a field:
        class Request(BaseModel):
            exaggeration: float = 0.5

        # WHEN derived,
        # THEN it refuses — overrides may not lie about the schema:
        with pytest.raises(ValueError, match="may not set"):
            build_capabilities(Request, **METADATA, overrides={"exaggeration": {"name": "x"}})

    def test_overrides_invalidValue_raisesValueError(self) -> None:
        # GIVEN an override whose value is not a valid ParamSpec value:
        class Request(BaseModel):
            exaggeration: float = 0.5

        # WHEN derived,
        # THEN validation of the merged spec rejects it:
        with pytest.raises(ValueError, match="Invalid capabilities override"):
            build_capabilities(Request, **METADATA, overrides={"exaggeration": {"step": "fast"}})


class TestDocumentMetadata:
    def test_build_capabilities_staticMetadata_isPassedThrough(self) -> None:
        # GIVEN static per-server metadata:
        class Request(BaseModel):
            text: str = Field(..., min_length=1)

        # WHEN built with explicit metadata:
        doc = build_capabilities(
            Request,
            engine="chatterbox",
            model="v3",
            device="cuda",
            sample_rate=24000,
            watermarked=True,
            languages=["en", "de"],
            reference_audio={"required": True, "formats": ["wav"], "note": "10 s cap"},
        )

        # THEN it appears verbatim in the document:
        assert doc.engine == "chatterbox"
        assert doc.model == "v3"
        assert doc.device == "cuda"
        assert doc.sample_rate == 24000
        assert doc.watermarked is True
        assert doc.languages == ["en", "de"]
        assert doc.reference_audio.required is True
        assert doc.reference_audio.formats == ["wav"]
        assert doc.reference_audio.note == "10 s cap"

    def test_build_capabilities_defaults_endpointAndSchemaVersion(self) -> None:
        # GIVEN a minimal build:
        class Request(BaseModel):
            text: str = Field(..., min_length=1)

        # WHEN built with only required metadata:
        doc = build_capabilities(Request, **METADATA)

        # THEN the endpoint defaults and the schema version is current
        # (compare against the constant, not a hardcoded number, so future
        # bumps fail here deliberately rather than drifting):
        assert doc.endpoint == "/synthesize"
        assert doc.schema_version == SCHEMA_VERSION
