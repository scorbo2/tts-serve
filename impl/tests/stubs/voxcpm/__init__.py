"""Stub of ``voxcpm`` for test machines without the real package.

Re-exports VoxCPM at the package level so ``from voxcpm import VoxCPM``
works exactly as it does with the real package.
"""

from .model.utils import VoxCPM

__all__ = ["VoxCPM"]
