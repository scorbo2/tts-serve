"""Core vocabulary and small dependency-free helpers shared by every engine server."""

from __future__ import annotations

import base64
import math

# The tts-serve release version: the one constant every /health endpoint
# reports and every FastAPI app metadata carries (the old per-server
# `version="0.x.y"` stamps that drifted between 0.1.0 and 0.3.0 and matched
# no release anyone could name are gone — this rules them all now).
# Bump it at the *start* of each release cycle, in the same commit that adds
# the README changelog placeholder; the final value must match the git tag
# (git tags carry no 'v' prefix: "1.4", not "v1.4").
# NOT the same thing as SCHEMA_VERSION below: this number changes on every
# release, backwards-compatible or not; SCHEMA_VERSION changes only when the
# client contract breaks.  See the version-bump skill (.agents/skills/) for
# the full workflow.
RELEASE_VERSION = "1.5"

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
