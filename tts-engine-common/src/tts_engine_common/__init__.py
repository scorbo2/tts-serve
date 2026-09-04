"""tts-engine-common: shared building blocks for tts-serve TTS engine servers.

Deliberately pure fastapi + pydantic — no torch, no engine libraries — so it
installs light, imports with no side effects, and tests on any dev box.
"""

from .core import CORE_FIELDS, SCHEMA_VERSION, compute_rtf, decode_base64
from .derive import DerivationError, build_capabilities, spec_from_schema
from .models import (
    Capabilities,
    CoreSynthesisResponse,
    ParamSpec,
    ReferenceAudioSpec,
)
from .route import add_capabilities_route, capabilities_endpoint

__version__ = "0.1.0"

__all__ = [
    "CORE_FIELDS",
    "SCHEMA_VERSION",
    "Capabilities",
    "CoreSynthesisResponse",
    "DerivationError",
    "ParamSpec",
    "ReferenceAudioSpec",
    "__version__",
    "add_capabilities_route",
    "build_capabilities",
    "capabilities_endpoint",
    "compute_rtf",
    "decode_base64",
    "spec_from_schema",
]
