"""Stub of ``voxcpm.model.utils`` for test machines without the real package.

Faithful copies of the constants the server imports at module level.
"""

import os
import torch


def materialize_generation_seed(seed):
    """Return a concrete seed for a generation request."""
    if seed is not None:
        return int(seed)
    return int(torch.seed() & 0xFFFFFFFF)


def apply_generation_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_runtime_device(device, configured_device="cuda"):
    """Resolve the actual runtime device.

    Stub version: always returns the configured device (no real hardware check).
    """
    explicit = None if device is None else device.strip().lower()
    if explicit is None or explicit == "auto":
        return configured_device
    if explicit.startswith("cuda"):
        return explicit
    if explicit == "mps":
        return "mps"
    if explicit == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device '{device}'")


class LoRAConfig:
    """Placeholder — LoRA is not exercised in tests."""

    def __init__(self, enable_lm=True, enable_dit=True, enable_proj=False):
        self.enable_lm = enable_lm
        self.enable_dit = enable_dit
        self.enable_proj = enable_proj


class VoxCPM:
    """Placeholder — real model loading is never exercised in the tests."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError(
            "voxcpm stub: from_pretrained() is not available in tests"
        )

    def generate(self, *args, **kwargs):
        raise NotImplementedError("voxcpm stub: generate() is not available in tests")
