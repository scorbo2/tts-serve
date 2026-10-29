"""Stub for the ``ming_tts`` module (test machines only).

The engine ships as a top-level ``ming_tts.py`` module in the Ming-3.0-tts
repo (not a package), so this stub is a module too.  Only the surface
``server_ming3TTS.py`` references at import time is provided; real model
loading is never exercised in the tests.
"""


class MingTTS:
    """Placeholder — real model loading is never exercised in the tests."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ming_tts stub: constructing MingTTS is not available in tests"
        )

    def synthesize(self, *args, **kwargs):
        raise NotImplementedError(
            "ming_tts stub: synthesize() is not available in tests"
        )
