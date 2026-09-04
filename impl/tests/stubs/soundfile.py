"""Minimal ``soundfile`` stub for machines without the real package installed.

Implements just enough of the API surface for the servers' reference-audio
handling to be exercisable in tests:

* ``info()`` parses WAV headers via the stdlib ``wave`` module, so the
  duration check (>= 2 s) and the undecodable-audio 400 path are testable
  with real (tiny) WAV bytes.
* ``read()`` / ``write()`` cannot be faked faithfully and raise when called —
  the tests stop before those code paths.
"""

import io
import wave


class _Info:
    def __init__(self, frames: int, samplerate: int, channels: int) -> None:
        self.frames = frames
        self.samplerate = samplerate
        self.channels = channels


def _open(file):
    if isinstance(file, (bytes, bytearray)):
        file = io.BytesIO(bytes(file))
    if hasattr(file, "read"):
        # file object: let wave() drive it; mode defaults to 'r'
        return wave.open(file)
    return wave.open(file, "rb")


def info(file):
    with _open(file) as w:
        return _Info(w.getnframes(), w.getframerate(), w.getnchannels())


def read(*args, **kwargs):
    raise NotImplementedError("soundfile stub: read() is not available in tests")


def write(*args, **kwargs):
    raise NotImplementedError("soundfile stub: write() is not available in tests")
