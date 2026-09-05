"""Shared contract for the ``language`` request parameter.

The client application speaks two-letter lowercase language codes (see
docs/02-language-handling.md); engines that internally require a different
form (full names, uppercase codes, ...) map it at their own server.  The
API-side half of that contract lives here so no server re-implements (or
re-interprets) it differently.
"""

from __future__ import annotations

import re

# Convention (docs/02): a null or empty ``language`` means English.
DEFAULT_LANGUAGE = "en"

# Two lowercase letters, e.g. 'en', 'fr', 'de'.  Deliberately NOT 3-letter
# ISO 639-2 codes or BCP-47 tags: the client contract is two letters, plain.
_TWO_LETTER_CODE = re.compile(r"[a-z]{2}")


def is_language_code(value: object) -> bool:
    """Whether ``value`` is a two-letter lowercase code like 'en' or 'fr'."""
    return isinstance(value, str) and _TWO_LETTER_CODE.fullmatch(value) is not None


def normalize_language(value: object) -> str:
    """Apply the null/empty -> 'en' convention and reject non-strings.

    Runs as a ``mode="before"`` validator, i.e. on *raw* JSON input before
    Pydantic coerces anything — so a leaked int/list/dict must fail here with
    ``ValueError`` (a 422), not with ``AttributeError`` (a 500).  Non-empty
    string values pass through untouched: *which* codes the engine accepts is
    the server's job (Literal enum or format check), so this helper never
    rejects a string's *content* — it only fills in the default.
    """
    if value is None:
        return DEFAULT_LANGUAGE
    if not isinstance(value, str):
        raise ValueError(
            f"language must be a two-letter lowercase code like 'en' or 'fr', "
            f"got {value!r}"
        )
    if not value.strip():
        return DEFAULT_LANGUAGE
    return value


def validate_language_code(value: str, *, allow_auto: bool = False) -> str:
    """Return ``value`` if it is a two-letter code, else raise ValueError.

    With ``allow_auto=True``, the special value ``'auto'`` (the auto-detection
    sentinel for engines that offer one, docs/02) is accepted verbatim; the
    flag extends the contract, it does not replace it.

    Intended for use as a Pydantic ``field_validator`` on engines that accept
    any language (no fixed enum), so garbage like 'x?' fails with a 422
    instead of reaching the engine.
    """
    if value == "auto" and allow_auto:
        return value
    if is_language_code(value):
        return value
    extra = ", or 'auto'" if allow_auto else ""
    raise ValueError(
        f"language must be a two-letter lowercase code like 'en' or 'fr'"
        f"{extra}, got {value!r}"
    )
