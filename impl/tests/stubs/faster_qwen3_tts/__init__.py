"""Stub for the ``faster_qwen3_tts`` package (test machines only)."""


class FasterQwen3TTS:
    """Placeholder — real model loading is never exercised in the tests."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError(
            "faster_qwen3_tts stub: from_pretrained() is not available in tests"
        )

    def warmup(self, *args, **kwargs):
        raise NotImplementedError(
            "faster_qwen3_tts stub: warmup() is not available in tests"
        )

    def generate_voice_clone(self, *args, **kwargs):
        raise NotImplementedError(
            "faster_qwen3_tts stub: generate_voice_clone() is not available in tests"
        )
