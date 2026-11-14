"""tts-engine-common: shared building blocks for tts-serve TTS engine servers.

Deliberately pure fastapi + pydantic — no torch, no engine libraries — so it
installs light, imports with no side effects, and tests on any dev box.
"""

from .core import (
    CORE_FIELDS,
    RELEASE_VERSION,
    SCHEMA_VERSION,
    compute_rtf,
    decode_base64,
)
from .derive import DerivationError, build_capabilities, spec_from_schema
from .devices import (
    DEFAULT_DEVICE_FAMILIES,
    DeviceSpec,
    cuda_ordinal_error,
    parse_device,
    require_cuda_available,
)
from .language import (
    DEFAULT_LANGUAGE,
    canonical_language_codes,
    canonical_language_literal,
    is_language_code,
    normalize_language,
    validate_language_code,
)
from .models import (
    Capabilities,
    CoreHealthResponse,
    CoreSynthesisResponse,
    ParamSpec,
    ReferenceAudioSpec,
)
from .route import add_capabilities_route, capabilities_endpoint
from .staging import (
    cleanup_temp,
    stage_audio,
    temp_audio_dir,
    write_temp_audio,
)

__version__ = "0.1.0"

__all__ = [
    "CORE_FIELDS",
    "DEFAULT_DEVICE_FAMILIES",
    "DEFAULT_LANGUAGE",
    "RELEASE_VERSION",
    "SCHEMA_VERSION",
    "Capabilities",
    "CoreHealthResponse",
    "CoreSynthesisResponse",
    "DerivationError",
    "DeviceSpec",
    "ParamSpec",
    "ReferenceAudioSpec",
    "__version__",
    "add_capabilities_route",
    "build_capabilities",
    "capabilities_endpoint",
    "canonical_language_codes",
    "canonical_language_literal",
    "cleanup_temp",
    "compute_rtf",
    "cuda_ordinal_error",
    "decode_base64",
    "is_language_code",
    "normalize_language",
    "parse_device",
    "require_cuda_available",
    "spec_from_schema",
    "stage_audio",
    "temp_audio_dir",
    "validate_language_code",
    "write_temp_audio",
]
