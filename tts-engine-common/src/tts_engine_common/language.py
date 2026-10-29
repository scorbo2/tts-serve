"""Shared contract for the ``language`` request parameter.

The client application speaks two-letter lowercase language codes (see
docs/02-language-handling.md); engines that internally require a different
form (full names, uppercase codes, ...) map it at their own server.  The
API-side half of that contract lives here so no server re-implements (or
re-interprets) it differently.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Literal

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


def canonical_language_codes(codes: Iterable[str], *, auto: bool = False) -> tuple[str, ...]:
    """Canonical member order for a fixed-enum ``language`` contract.

    Single ordering oracle: dedupes, rejects malformed tables at import
    time, and returns ``'auto'`` first (only when the engine offers the
    auto-detection sentinel) then the sorted codes.  Servers must feed BOTH
    halves of their advertised contract through it: the request ``Literal``
    (via ``canonical_language_literal``) and the ``languages`` list handed
    to ``build_capabilities`` — so neither half can drift into declaration
    order, and ``test_capabilities_languageEnum_advertisedInCanonicalSortedOrder``
    (which derives its expectation from the capabilities document itself)
    can trust them to agree.

    Why pinning the order matters at all: CPython compares/hashes
    ``Literal[...]`` by the **frozenset** of its args, so pydantic's core-
    schema cache conflates same-code-set Literals and whichever model
    builds its JSON schema first supplies the enum for *all* of them —
    silently reordering another server's advertised ``/capabilities`` enum
    (this shipped once: the MoE Ming snapshot carried the 0.5b server's
    code order).  With every server pinned to this order, a collision can
    no longer change anything observable.

    Raises ValueError at import time for an empty code list or anything
    that is not a two-letter lowercase code — a typo'd engine table should
    fail at startup, not when a client reads /capabilities.  Pass the
    auto-detection sentinel via ``auto=`` rather than hiding it in
    ``codes``; the advertised ``languages`` list deliberately excludes
    ``'auto'`` (docs/02), so call without the flag for that half.
    """
    ordered = sorted(set(codes))
    if not ordered:
        raise ValueError("canonical_language_codes needs at least one code")
    invalid = [code for code in ordered if not is_language_code(code)]
    if invalid:
        raise ValueError(
            f"language codes must be two-letter lowercase like 'en'/'fr' "
            f"(pass 'auto' via auto=, never in codes); got {invalid}"
        )
    return ("auto", *ordered) if auto else tuple(ordered)


def canonical_language_literal(codes: Iterable[str], *, auto: bool = False) -> Any:
    """Build the ``Literal`` type for a fixed-enum ``language`` request field.

    Every server with an enum-based language field must build its
    ``Language`` type through this helper rather than a hand-rolled
    ``Literal[tuple(...)]``; the members are exactly
    ``canonical_language_codes(codes, auto=auto)``, whose docstring carries
    the pydantic Literal-cache-collision lore that makes the pinning
    mandatory.
    """
    return Literal[canonical_language_codes(codes, auto=auto)]
