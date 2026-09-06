"""Make tools/speak.py importable as ``speak`` from this test suite.

Inserted at position 0 deliberately (the reverse of impl/tests' stub
strategy): these tests must exercise *this repo's* speak.py, so it must win
over any coincidentally-named installed package.
"""

import sys
from pathlib import Path

_TOOLS_DIR = str(Path(__file__).resolve().parents[1])
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
