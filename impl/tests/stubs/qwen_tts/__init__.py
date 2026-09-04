"""Stub for the ``qwen_tts`` package (test machines only)."""


class Qwen3TTSModel:
    """Placeholder — real model loading is never exercised in the tests."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError(
            "qwen_tts stub: from_pretrained() is not available in tests"
        )

    def generate_voice_clone(self, *args, **kwargs):
        raise NotImplementedError(
            "qwen_tts stub: generate_voice_clone() is not available in tests"
        )
