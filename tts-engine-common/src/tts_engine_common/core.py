"""Core vocabulary and small dependency-free helpers shared by every engine server."""

from __future__ import annotations

import base64
import math

# Bumped only for breaking changes to the capabilities document (see
# docs/01-server-generification.md, section 3.5).
# History: v2 (2026-09) — `language` contract solidified to two-letter codes
# with null/empty -> 'en' normalization (docs/02-language-handling.md).
SCHEMA_VERSION = 2

# The "common" parameter vocabulary: the field names a client application may
# assume exist by name when an engine advertises them. Anything outside this
# set is engine-specific and must be rendered generically from metadata.
#
# Note (design doc, open question Q1): reference_text is part of the common
# vocabulary but is *optional* — most voice cloners want a transcript of the
# reference clip, but some condition purely on the audio (Chatterbox) and
# simply omit the field. Clients must therefore only send it when the
# capabilities document advertises it.
CORE_FIELDS = frozenset({"text", "audio_base64", "reference_text", "language", "seed"})


def decode_base64(data: str) -> bytes:
    """Strictly decode base64; raises ValueError on malformed input."""
    # validate=True rejects non-alphabet characters, which the default lenient
    # mode would silently strip — a near-miss we'd rather 422 than guess at.
    return base64.b64decode(data, validate=True)


def compute_rtf(time_used_s: float, num_samples: int, sample_rate: int) -> float | None:
    """Real-time factor: generation wall-time divided by audio duration.

    Returns None when the duration is zero or uncomputable, or when the result
    is not finite (degenerate inputs can produce inf; JSON cannot represent it,
    and the response model maps such values to null anyway).
    """
    if sample_rate <= 0 or num_samples <= 0:
        return None
    duration = num_samples / sample_rate
    if duration <= 0:
        return None
    rtf = time_used_s / duration
    if not math.isfinite(rtf):
        return None
    return rtf
