"""Stub for the ``omnivoice`` package (test machines only)."""

from typing import Any


class OmniVoice:
    """Placeholder — real model loading is never exercised in the tests."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError(
            "omnivoice stub: from_pretrained() is not available in tests"
        )

    def generate(self, *args, **kwargs):
        raise NotImplementedError("omnivoice stub: generate() is not available in tests")


class OmniVoiceGenerationConfig:
    """Mirrors the real dataclass's constructor surface used by the server."""

    def __init__(
        self,
        num_step: int = 32,
        guidance_scale: float = 2.0,
        denoise: bool = True,
        **kwargs: Any,
    ) -> None:
        self.num_step = num_step
        self.guidance_scale = guidance_scale
        self.denoise = denoise
        self.extra = kwargs

    @classmethod
    def from_dict(cls, d: dict) -> "OmniVoiceGenerationConfig":
        return cls(**d)


class VoiceClonePrompt:
    """Present for parity with the real package's public exports."""
