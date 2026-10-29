"""Stub for the ``ming_tiny`` package (test machines only).

The real package exposes ``MingTinyTTS`` via a lazy module-level
``__getattr__`` (to keep CUDA initialization out of plain ``import
ming_tiny``); the stub just provides the name directly.  Only the surface
``server_ming3TTS_moe.py`` references at import time is present; real model
loading is never exercised in the tests.
"""


class MingTinyTTS:
    """Placeholder — real model loading is never exercised in the tests."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError(
            "ming_tiny stub: from_pretrained() is not available in tests"
        )

    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            "ming_tiny stub: generate() is not available in tests"
        )
