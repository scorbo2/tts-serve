"""Tests for the shared language contract in tts_engine_common.language."""

from __future__ import annotations

import pytest

from tts_engine_common.language import (
    DEFAULT_LANGUAGE,
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
