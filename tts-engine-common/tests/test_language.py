"""Tests for the shared language contract in tts_engine_common.language."""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import BaseModel, ValidationError

from tts_engine_common.language import (
    DEFAULT_LANGUAGE,
    canonical_language_codes,
    canonical_language_literal,
    is_language_code,
    normalize_language,
    validate_language_code,
)


class TestDefaultLanguage:
    def test_default_language_isEnglish(self) -> None:
        # GIVEN the docs/02 convention (null/empty means English),
        # THEN the shared default is 'en':
        assert DEFAULT_LANGUAGE == "en"


class TestIsLanguageCode:
    @pytest.mark.parametrize("code", ["en", "fr", "de", "zh"])
    def test_is_language_code_validCodes_returnsTrue(self, code: str) -> None:
        # GIVEN a two-letter lowercase code:
        assert is_language_code(code) is True

    def test_is_language_code_uppercase_returnsFalse(self) -> None:
        # GIVEN an uppercase code (the client contract is lowercase):
        assert is_language_code("EN") is False

    def test_is_language_code_mixedCase_returnsFalse(self) -> None:
        # GIVEN a mixed-case code:
        assert is_language_code("En") is False

    def test_is_language_code_threeLetters_returnsFalse(self) -> None:
        # GIVEN a three-letter ISO 639-2 code (the contract is two letters):
        assert is_language_code("eng") is False

    def test_is_language_code_punctuation_returnsFalse(self) -> None:
        # GIVEN the docs/02 example of a garbage value:
        assert is_language_code("x?") is False

    def test_is_language_code_nonString_returnsFalse(self) -> None:
        # GIVEN a non-string value (e.g. a leaked JSON null/int):
        assert is_language_code(None) is False
        assert is_language_code(7) is False


class TestNormalizeLanguage:
    def test_normalize_language_none_returnsDefault(self) -> None:
        # GIVEN an omitted/null language:
        # WHEN normalized,
        # THEN the docs/02 convention applies: English.
        assert normalize_language(None) == "en"

    def test_normalize_language_emptyString_returnsDefault(self) -> None:
        # GIVEN an empty language:
        assert normalize_language("") == "en"

    def test_normalize_language_whitespace_returnsDefault(self) -> None:
        # GIVEN a whitespace-only language:
        assert normalize_language("   ") == "en"

    @pytest.mark.parametrize("code", ["en", "fr", "auto"])
    def test_normalize_language_nonEmpty_returnsUnchanged(self, code: str) -> None:
        # GIVEN a non-empty value (valid or not — rejection is the
        # server's job, not the normalizer's):
        assert normalize_language(code) == code

    @pytest.mark.parametrize("bad", [7, 12.5, ["en"], {"code": "en"}, True])
    def test_normalize_language_nonString_raisesValueError(self, bad: object) -> None:
        # GIVEN a non-string value (a leaked JSON int/list/dict/bool —
        # the mode="before" validator sees raw input before Pydantic
        # coerces anything):
        # WHEN normalized,
        # THEN it raises ValueError (a 422) instead of AttributeError
        # (a 500):
        with pytest.raises(ValueError, match="two-letter lowercase code"):
            normalize_language(bad)


class TestValidateLanguageCode:
    @pytest.mark.parametrize("code", ["en", "fr", "de"])
    def test_validate_language_code_validCode_returnsIt(self, code: str) -> None:
        # GIVEN a valid two-letter code:
        assert validate_language_code(code) == code

    def test_validate_language_code_uppercase_raisesValueError(self) -> None:
        # GIVEN an uppercase code:
        # WHEN validated,
        # THEN it raises (so the server can 422):
        with pytest.raises(ValueError, match="two-letter lowercase code"):
            validate_language_code("EN")

    def test_validate_language_code_punctuation_raisesValueError(self) -> None:
        # GIVEN the docs/02 example of a garbage value:
        with pytest.raises(ValueError, match="two-letter lowercase code"):
            validate_language_code("x?")

    def test_validate_language_code_name_raisesValueError(self) -> None:
        # GIVEN a full language name (engines map names internally, the
        # API contract is codes):
        with pytest.raises(ValueError, match="two-letter lowercase code"):
            validate_language_code("English")

    def test_validate_language_code_allowAuto_autoPasses(self) -> None:
        # GIVEN the auto-detection sentinel with allow_auto=True (engines
        # that offer auto-detection, docs/02):
        assert validate_language_code("auto", allow_auto=True) == "auto"

    def test_validate_language_code_allowAuto_validCode_stillPasses(self) -> None:
        # GIVEN a plain code with allow_auto=True — the flag extends the
        # contract, it does not replace it:
        assert validate_language_code("en", allow_auto=True) == "en"

    def test_validate_language_code_autoWithoutFlag_raisesValueError(self) -> None:
        # GIVEN 'auto' but allow_auto left at its default (the engine has
        # no auto-detection to offer):
        with pytest.raises(ValueError, match="two-letter lowercase code"):
            validate_language_code("auto")

    @pytest.mark.parametrize("bad", ["AUTO", "auto_detect", "x?", "english"])
    def test_validate_language_code_allowAuto_garbageRaisesValueError(self, bad: str) -> None:
        # GIVEN a non-code value with allow_auto=True — the sentinel is
        # exactly 'auto', not 'auto-ish':
        with pytest.raises(ValueError, match="two-letter lowercase code"):
            validate_language_code(bad, allow_auto=True)

    def test_validate_language_code_allowAuto_rejectionMentionsAuto(self) -> None:
        # GIVEN a rejection under allow_auto=True,
        # THEN the message names 'auto' so clients know it is allowed:
        with pytest.raises(ValueError, match="or 'auto'"):
            validate_language_code("English", allow_auto=True)


class TestCanonicalLanguageCodes:
    def test_canonical_language_codes_unsortedCodes_returnsSortedTuple(self) -> None:
        # GIVEN codes in declaration (unsorted) order,
        # WHEN the canonical tuple is built,
        # THEN it holds the sorted codes — the shared ordering oracle for
        # both the Literal and the capabilities 'languages' list:
        assert canonical_language_codes(("zh", "en", "de", "es", "fr", "ja", "ko")) == (
            "de", "en", "es", "fr", "ja", "ko", "zh",
        )

    def test_canonical_language_codes_autoFlag_autoLeadsThenSortedCodes(self) -> None:
        # GIVEN an engine with the auto-detection sentinel,
        # THEN 'auto' leads and the codes follow sorted:
        assert canonical_language_codes(("zh", "en", "fr"), auto=True) == ("auto", "en", "fr", "zh")

    def test_canonical_language_codes_duplicateCodes_areDeduplicated(self) -> None:
        # GIVEN a sloppy engine table listing 'en' twice:
        assert canonical_language_codes(("en", "fr", "en")) == ("en", "fr")

    def test_canonical_language_codes_emptyCodes_raisesValueError(self) -> None:
        # GIVEN no codes at all (an engine with no fixed list should use the
        # free-form validator, not this helper):
        with pytest.raises(ValueError, match="at least one code"):
            canonical_language_codes(())

    @pytest.mark.parametrize("bad", ["EN", "eng", "x?", "English"])
    def test_canonical_language_codes_malformedCode_raisesValueError(self, bad: str) -> None:
        # GIVEN a code that violates the two-letter lowercase contract:
        with pytest.raises(ValueError, match="two-letter lowercase"):
            canonical_language_codes(("en", bad))

    def test_canonical_language_codes_autoInCodes_raisesValueError(self) -> None:
        # GIVEN 'auto' smuggled into the code list instead of passed via
        # auto= (it is not a two-letter code, and hiding it in the list
        # would advertise it on engines without auto-detection):
        with pytest.raises(ValueError, match="auto="):
            canonical_language_codes(("en", "auto"))

    @pytest.mark.parametrize("auto", [False, True], ids=["noAuto", "withAuto"])
    def test_canonical_language_literal_membersAreExactlyTheCanonicalCodes(self, auto: bool) -> None:
        # GIVEN the same code list handed to both helpers,
        # THEN the Literal's members are exactly the canonical tuple: the
        # request schema and the capabilities 'languages' list can never
        # disagree, because they are one expression under the hood:
        codes = ("zh", "en", "de", "es", "fr", "ja", "ko")
        assert get_args(canonical_language_literal(codes, auto=auto)) == canonical_language_codes(
            codes, auto=auto
        )


class TestCanonicalLanguageLiteral:
    # Error handling (empty/malformed/auto-smuggling/dedup) is pinned once
    # at the source in TestCanonicalLanguageCodes — this helper is a
    # one-liner over it, and the consistency test there proves the
    # coupling.  Only what *this* wrapper uniquely provides is covered
    # here: member order in the built type, the cross-model enum
    # collision, and annotation usability.

    def test_canonical_language_literal_unsortedCodes_argsAreSorted(self) -> None:
        # GIVEN codes handed in declaration (unsorted) order,
        # WHEN the Literal is built,
        # THEN its members are the sorted codes — declaration order is not
        # allowed to reach the /capabilities enum:
        literal = canonical_language_literal(("zh", "en", "de", "es", "fr", "ja", "ko"))
        assert get_args(literal) == ("de", "en", "es", "fr", "ja", "ko", "zh")

    def test_canonical_language_literal_sameSetDifferentOrders_advertiseIdenticalEnums(self) -> None:
        # GIVEN two servers declaring the same code set in different
        # orders (exactly the Ming 0.5b-vs-MoE bug): CPython hashes
        # Literal by the frozenset of its args, so pydantic's core-schema
        # cache hands whichever model built its schema first to *both* —
        # WHEN both models are compiled in one process and emit JSON
        # schemas,
        # THEN both advertise the identical sorted enum: the helper makes
        # the collision unobservable.
        literalA = canonical_language_literal(("zh", "en", "ja", "ko", "de", "fr", "es"))
        literalB = canonical_language_literal(("zh", "en", "de", "es", "fr", "ja", "ko"))

        class ServerA(BaseModel):
            language: literalA | None = "en"

        class ServerB(BaseModel):
            language: literalB | None = "en"

        def enum_of(model: type[BaseModel]) -> list[str]:
            branch = model.model_json_schema()["properties"]["language"]["anyOf"][0]
            return branch["enum"]

        assert enum_of(ServerA) == enum_of(ServerB)
        assert enum_of(ServerA) == ["de", "en", "es", "fr", "ja", "ko", "zh"]

    def test_canonical_language_literal_usableAsFieldAnnotation_validatesMembership(self) -> None:
        # GIVEN a model field typed with a helper-built Literal (with auto),
        # WHEN a listed code or the sentinel is submitted, validation
        # passes; an unlisted code fails:
        literal = canonical_language_literal(("en", "fr"), auto=True)

        class Request(BaseModel):
            language: literal | None = "en"

        assert Request(language="fr").language == "fr"
        assert Request(language="auto").language == "auto"
        with pytest.raises(ValidationError):
            Request(language="de")
