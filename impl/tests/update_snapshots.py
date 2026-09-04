"""Regenerate the /capabilities snapshot files.

Run from anywhere:

    python impl/tests/update_snapshots.py

The snapshots are committed to the repo; regenerate them after changing a
server's request schema (or its engine constants) and review the diff.
The dots.tts snapshot records the device as observed on this machine — the
test compares everything except ``device`` for that engine (it is
machine-dependent), so this stays stable across CUDA/CPU boxes.
"""

import json
import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import _bootstrap  # noqa: E402

_bootstrap.setup()

from fastapi.testclient import TestClient  # noqa: E402

import server_chatterbox  # noqa: E402
import server_dotsTTS  # noqa: E402
import server_omnivoice  # noqa: E402
import server_qwen3TTS  # noqa: E402

SNAPSHOTS_DIR = Path(__file__).resolve().parent / "snapshots"

# (server module, snapshot filename)
ENGINES = (
    (server_chatterbox, "chatterbox_capabilities.json"),
    (server_omnivoice, "omnivoice_capabilities.json"),
    (server_qwen3TTS, "qwen3_capabilities.json"),
    (server_dotsTTS, "dots_capabilities.json"),
)


def main() -> None:
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    for module, filename in ENGINES:
        # No context manager: skip lifespan (model load).
        client = TestClient(module.app)
        response = client.get("/capabilities")
        assert response.status_code == 200, response.text
        path = SNAPSHOTS_DIR / filename
        path.write_text(json.dumps(response.json(), indent=2) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
