"""Make tools/speak.py importable as ``speak`` from this test suite.

Inserted at position 0 deliberately (the reverse of impl/tests' stub
strategy): these tests must exercise *this repo's* speak.py, so it must win
over any coincidentally-named installed package.
"""

import os
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = str(Path(__file__).resolve().parents[1])
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)


@pytest.fixture(autouse=True)
def _strip_tts_speak_env(monkeypatch):
    """Hermeticity: strip ambient TTS_SPEAK_* variables from every test.

    A dev box may legitimately export TTS_SPEAK_SERVER / TTS_SPEAK_PERSONA_DIR,
    and an exported value would silently change what several end-to-end tests
    exercise (e.g. the "no reference audio" error would never fire).  Tests
    that need a value set it explicitly via monkeypatch.  Stripping by prefix
    means future TTS_SPEAK_* variables stay hermetic too.
    """
    for name in list(os.environ):
        if name.startswith("TTS_SPEAK_"):
            monkeypatch.delenv(name, raising=False)
