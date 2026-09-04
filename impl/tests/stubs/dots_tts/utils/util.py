"""Stub of ``dots_tts.utils.util`` for test machines without the real package.

The server imports ``seed_everything`` from here; a no-op keeps the import
surface identical without pulling in torch.
"""


def seed_everything(seed: int = 42) -> None:
    pass
