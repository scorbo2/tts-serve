"""Stub of ``zipvoice.luxvoice`` for test machines without the real package.

The server imports only the ``LuxTTS`` class (no module-level engine
constants), so this placeholder keeps the import surface identical without
pulling in torch.  Model loading is never exercised in the tests.
"""


class LuxTTS:
    """Placeholder — real model loading is never exercised in the tests."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "zipvoice stub: LuxTTS() is not available in tests"
        )

    def encode_prompt(self, *args, **kwargs):
        raise NotImplementedError(
            "zipvoice stub: encode_prompt() is not available in tests"
        )

    def generate_speech(self, *args, **kwargs):
        raise NotImplementedError(
            "zipvoice stub: generate_speech() is not available in tests"
        )
