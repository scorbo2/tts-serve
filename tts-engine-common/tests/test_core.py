"""Tests for the small dependency-free helpers in tts_engine_common.core."""

from __future__ import annotations

import pytest

from tts_engine_common.core import compute_rtf, decode_base64


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
