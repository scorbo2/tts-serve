"""Shared import/environment bootstrap for the impl/ server tests.

Imported by both ``conftest.py`` (pytest) and ``update_snapshots.py``
(standalone) so the two always agree on sys.path and environment.  If they
ever diverge, the committed snapshots would no longer match what the tests
expect — so there is exactly one copy of this logic.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # impl/tests
IMPL_DIR = HERE.parent                            # impl
REPO_ROOT = IMPL_DIR.parent                       # repo root
COMMON_SRC = REPO_ROOT / "tts-engine-common" / "src"
STUBS_DIR = HERE / "stubs"


def setup() -> None:
    # 1. Server modules + tts_engine_common on the import path.
    for path in (str(IMPL_DIR), str(COMMON_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)

    # 2. Stubs LAST, so real installed packages (site-packages) always win;
    #    the stubs only fill in what is missing on this machine (torch,
    #    numpy, soundfile, loguru, and the engine packages on a dev box).
    if str(STUBS_DIR) not in sys.path:
        sys.path.append(str(STUBS_DIR))

    # 3. Pin the environment to the servers' documented defaults so the
    #    /capabilities snapshots are deterministic regardless of the ambient
    #    environment.  The servers read config at import time, so this must
    #    run before any server module is imported.
    os.environ["CHATTERBOX_DEVICE"] = "cuda"
    os.environ.pop("CHATTERBOX_T3_MODEL", None)  # default: v3
    os.environ["OMNIVOICE_DEVICE"] = "cuda"
    os.environ.pop("OMNIVOICE_MODEL", None)      # default: k2-fsa/OmniVoice
    os.environ["QWEN3TTS_DEVICE"] = "cuda"
    os.environ.pop("QWEN3TTS_MODEL", None)       # default: Qwen/Qwen3-TTS-12Hz-1.7B-Base
    os.environ.pop("DOTS_TTS_MODEL", None)       # default: rednote-hilab/dots.tts-soar
