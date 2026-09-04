"""Contract tests: the capabilities document must stay in lockstep with the model.

These mirror the acceptance criteria from docs/01-server-generification.md:
every parameter the document advertises must exist on the request model, and
the document must survive a JSON round-trip (it is what clients parse).
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from tts_engine_common import CORE_FIELDS, Capabilities, build_capabilities


class _SampleRequest(BaseModel):
    """A deliberately mixed model: core fields, engine fields, several shapes."""

    text: str = Field(..., min_length=1)
    audio_base64: str = Field(..., min_length=1, max_length=10_000_000)
    reference_text: str | None = None
    language: Literal["en", "de", "fr"] | None = None
    seed: int | None = Field(None, ge=1, le=1000)
    num_steps: int = Field(32, ge=4, le=128)
    denoise: bool = True


def _doc() -> Capabilities:
    return build_capabilities(
        _SampleRequest,
        engine="test-engine",
        model="test-model",
        device="cuda",
        sample_rate=24000,
        watermarked=False,
        overrides={"num_steps": {"step": 4}},
    )


def test_contract_docParameters_allExistOnRequestModel() -> None:
    # GIVEN a derived capabilities document:
    doc = _doc()

    # THEN every advertised parameter exists on the request model
    # (the document can never describe a field validation would reject):
    model_fields = set(_SampleRequest.model_fields)
    advertised = {p.name for p in doc.parameters}
    assert advertised <= model_fields
    assert advertised == model_fields  # and every field is advertised


def test_contract_docFields_groupingIsConsistent() -> None:
    # GIVEN a derived document:
    doc = _doc()
    by_name = {p.name: p for p in doc.parameters}

    # THEN core vocabulary fields are 'common' and everything else 'engine':
    for name, spec in by_name.items():
        if name in CORE_FIELDS:
            assert spec.group == "common", f"{name} should be grouped common"
        else:
            assert spec.group == "engine", f"{name} should be grouped engine"


def test_contract_document_survivesJsonRoundTrip() -> None:
    # GIVEN a derived document:
    doc = _doc()

    # WHEN it is serialized to JSON and re-parsed into the model:
    restored = Capabilities.model_validate_json(json.dumps(doc.model_dump(mode="json")))

    # THEN the document is identical (clients parse exactly this shape):
    assert restored == doc


def test_contract_document_isJsonSerializable() -> None:
    # GIVEN a derived document (possibly with nulls and nested models):
    doc = _doc()

    # WHEN dumped with mode="json",
    # THEN json.dumps succeeds (no NaN/inf, no non-JSON types):
    json.dumps(doc.model_dump(mode="json"))
