"""Shared helpers for the impl/ server tests."""

import base64
import io
import json
import math
import struct
import wave
from pathlib import Path

SNAPSHOTS_DIR = Path(__file__).resolve().parent / "snapshots"


def make_wav_bytes(duration_s: float, sr: int = 24000, freq: float = 440.0, amplitude: float = 0.1) -> bytes:
    """Generate a minimal valid 16-bit PCM mono WAV of the given duration.

    Built with the stdlib ``wave`` module so it works on machines without
    soundfile/NumPy.  A pure sine — good enough for header parsing and
    duration checks; nobody is judging the timbre in CI.
    """
    n_frames = max(1, int(duration_s * sr))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit PCM
        w.setframerate(sr)
        frames = bytearray()
        for i in range(n_frames):
            sample = amplitude * 32767.0 * math.sin(2.0 * math.pi * freq * i / sr)
            frames += struct.pack("<h", max(-32768, min(32767, int(sample))))
        w.writeframes(bytes(frames))
    return buf.getvalue()


def b64(data: bytes) -> str:
    """Base64-encode bytes for the audio_base64 request field."""
    return base64.b64encode(data).decode("ascii")


def load_snapshot(name: str):
    """Load a committed JSON snapshot; fail with a hint if it is missing."""
    path = SNAPSHOTS_DIR / name
    if not path.exists():
        raise AssertionError(
            f"Snapshot {path} is missing. Generate it with:\n"
            "    python tests/update_snapshots.py"
        )
    return json.loads(path.read_text())
