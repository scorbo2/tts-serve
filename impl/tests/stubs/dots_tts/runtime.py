"""Stub of ``dots_tts.runtime`` for test machines without the real package."""


class DotsTtsRuntime:
    """Placeholder — real model loading is never exercised in the tests."""

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        raise NotImplementedError(
            "dots_tts stub: from_pretrained() is not available in tests"
        )

    def generate(self, **kwargs):
        raise NotImplementedError("dots_tts stub: generate() is not available in tests")
