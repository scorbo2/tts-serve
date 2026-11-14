"""Tests for the small dependency-free helpers in tts_engine_common.core."""

from __future__ import annotations

import pytest

from tts_engine_common.core import RELEASE_VERSION, SCHEMA_VERSION, compute_rtf, decode_base64


class TestDecodeBase64:
    def test_decode_base64_validInput_returnsRawBytes(self) -> None:
        # GIVEN a valid base64 string:
        payload = "aGVsbG8="  # "hello"

        # WHEN decoded strictly:
        raw = decode_base64(payload)

        # THEN we get the original bytes back:
        assert raw == b"hello"

    def test_decode_base64_nonAlphabetCharacter_raisesValueError(self) -> None:
        # GIVEN base64 with an invalid character (! is not in the alphabet):
        payload = "aGVsbG8!"

        # WHEN decoded with validate=True,
        # THEN it raises ValueError (so the server can 422 instead of guessing):
        with pytest.raises(ValueError):
            decode_base64(payload)

    def test_decode_base64_badPadding_raisesValueError(self) -> None:
        # GIVEN base64 whose length is not a multiple of 4:
        payload = "aGVsbG"

        # WHEN decoded strictly:
        with pytest.raises(ValueError):
            decode_base64(payload)


class TestComputeRtf:
    def test_compute_rtf_twoSecondsForOneSecondAudio_returnsTwo(self) -> None:
        # GIVEN 1 second of 24 kHz audio (24000 samples) generated in 2 s:
        rtf = compute_rtf(time_used_s=2.0, num_samples=24000, sample_rate=24000)

        # THEN the real-time factor is 2.0:
        assert rtf == pytest.approx(2.0)

    def test_compute_rtf_fasterThanRealtime_returnsFraction(self) -> None:
        # GIVEN 1 second of audio generated in 0.5 s:
        rtf = compute_rtf(time_used_s=0.5, num_samples=24000, sample_rate=24000)

        # THEN the factor is 0.5:
        assert rtf == pytest.approx(0.5)

    def test_compute_rtf_zeroSamples_returnsNone(self) -> None:
        # GIVEN no audio at all:
        rtf = compute_rtf(time_used_s=1.0, num_samples=0, sample_rate=24000)

        # THEN the factor is uncomputable:
        assert rtf is None

    def test_compute_rtf_zeroSampleRate_returnsNone(self) -> None:
        # GIVEN a nonsensical sample rate:
        rtf = compute_rtf(time_used_s=1.0, num_samples=24000, sample_rate=0)

        # THEN the factor is uncomputable:
        assert rtf is None

    def test_compute_rtf_negativeSampleRate_returnsNone(self) -> None:
        # GIVEN a negative sample rate:
        rtf = compute_rtf(time_used_s=1.0, num_samples=24000, sample_rate=-8000)

        # THEN the factor is uncomputable:
        assert rtf is None


class TestReleaseVersion:
    def test_releaseVersion_matchesGitTagForm(self) -> None:
        # GIVEN the single release-version constant (git tags carry no 'v'
        # prefix: "1.4", not "v1.4"):
        # WHEN split into dotted segments,
        # THEN every segment is numeric and there are at least two:
        segments = RELEASE_VERSION.split(".")
        assert len(segments) >= 2
        assert all(segment.isdigit() for segment in segments)

    def test_releaseVersion_isDistinctFromPackageVersion(self) -> None:
        # GIVEN the tts-serve release line and the tts-engine-common package
        # version — two independent numbers (the shared package is versioned
        # on its own schedule and historically stays at 0.1.0):
        import tts_engine_common

        # WHEN compared,
        # THEN they are different constants and the package version is
        # never used as the release line by accident:
        assert RELEASE_VERSION != tts_engine_common.__version__

    def test_releaseVersion_isNotTheSchemaVersion(self) -> None:
        # GIVEN the release line (str, bumped every release) and the
        # capabilities contract version (int, bumped only on breaking
        # changes) — a client gates on SCHEMA_VERSION, never on the
        # release line:
        # WHEN compared,
        # THEN they are different types and never conflated:
        assert isinstance(RELEASE_VERSION, str)
        assert isinstance(SCHEMA_VERSION, int)

    def test_releaseVersion_isExportedFromPackageRoot(self) -> None:
        # GIVEN the package root's public API:
        import tts_engine_common

        # WHEN inspected,
        # THEN RELEASE_VERSION is importable from the root and listed in
        # __all__, so servers never reach into submodules for it:
        assert tts_engine_common.RELEASE_VERSION == RELEASE_VERSION
        assert "RELEASE_VERSION" in tts_engine_common.__all__
