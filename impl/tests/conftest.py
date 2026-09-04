"""Pytest bootstrap for the impl/ server tests.

Delegates to ``_bootstrap.setup()`` (shared with ``update_snapshots.py``) so
sys.path and the pinned environment are identical under pytest and when
regenerating snapshots.  conftest is always imported before any test module,
so the environment is pinned before the first server import.
"""

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import _bootstrap  # noqa: E402

_bootstrap.setup()
