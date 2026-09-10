"""Temporary-file staging for engines that load prompt audio from disk.

Some engines refuse in-memory audio and load the prompt from a file path
(librosa / soundfile).  The mechanism is identical across servers, so it
lives here instead of being copy-pasted per engine server; each server keeps
its *own* staging directory, because some engines key internal caches by
that path.

Two strategies — pick the one that matches the engine:

``stage_audio()``
    Content-addressed (SHA-256 file name), written atomically, and the file
    is *kept* on disk.  For engines that cache speaker/emotion conditioning
    per path (faster-qwen3-tts, IndexTTS-2.5): repeat requests with the same
    clip reuse the staged file and hit the engine's cache instead of
    re-encoding.

``write_temp_audio()`` / ``cleanup_temp()``
    Uniquely named (UUID), and deleted by the caller once the engine is done
    with it.  For one-shot engines that do not cache by path
    (chatterbox, dots.tts).
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "cleanup_temp",
    "stage_audio",
    "temp_audio_dir",
    "write_temp_audio",
]


def temp_audio_dir(name: str) -> Path:
    """Per-server staging directory under the system temp dir (created).

    Uses ``tempfile.gettempdir()`` rather than a hardcoded ``/tmp`` so the
    servers respect ``$TMPDIR`` — and run at all on platforms without /tmp.
    """
    directory = Path(tempfile.gettempdir()) / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def stage_audio(raw_bytes: bytes, directory: Path) -> str:
    """Make raw audio bytes available at a stable content-addressed path.

    The engine loads prompt audio from a file path and caches the
    speaker/emotion conditioning per path, so the filename is the SHA-256 of
    the audio content rather than a UUID: repeat requests with the same clip
    hit that cache instead of re-encoding.  The .wav extension is cosmetic —
    the loader sniffs the header, so MP3/OGG/FLAC bytes work fine.

    The file is deliberately kept rather than deleted per request: staging
    happens *before* the synthesis lock is taken, so with a shared
    content-addressed name a per-request cleanup could delete the file after
    another request for the same clip has staged it but before that request
    reaches its turn on the lock.  Disk usage grows only with the number of
    *unique* clips; wipe the directory to reclaim space.

    Writing is atomic: bytes go to a unique sibling file that is then
    ``os.replace``d into place, so no reader can ever observe a half-written
    clip, and a crash mid-write cannot leave a corrupt file behind the
    content hash (which would poison every future request for that clip).
    Two concurrent requests staging the *same* new clip both rename into
    place; the renames are atomic and the content is identical, so the
    result is consistent either way.
    """
    digest = hashlib.sha256(raw_bytes).hexdigest()
    path = directory / f"{digest}.wav"
    if path.exists():
        # Same clip already staged: reusing it is what makes the engine's
        # path-keyed conditioning cache useful, and skipping the write also
        # keeps a queued request from clobbering a file that the request
        # holding the synthesis lock is currently reading.
        logger.debug("Reusing staged reference audio: %s", path)
        return str(path)
    staging = path.with_name(f"{digest}.wav.{uuid.uuid4().hex}.tmp")
    try:
        staging.write_bytes(raw_bytes)
        os.replace(staging, path)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    logger.debug("Staged reference audio: %s", path)
    return str(path)


def write_temp_audio(raw_bytes: bytes, directory: Path) -> str:
    """Write raw audio bytes to a uniquely-named temp file; caller owns it.

    For one-shot engines: the UUID name cannot collide with another request,
    so the caller is expected to delete the file with :func:`cleanup_temp`
    once the engine has read it.  Use :func:`stage_audio` instead if the
    engine caches conditioning per path.

    The .wav extension is cosmetic — the loader sniffs the container from
    the header, so MP3/OGG/FLAC bytes work fine.
    """
    path = directory / f"{uuid.uuid4().hex}.wav"
    path.write_bytes(raw_bytes)
    logger.debug("Wrote temporary reference audio: %s", path)
    return str(path)


def cleanup_temp(path: str | Path) -> None:
    """Best-effort removal of a file written by :func:`write_temp_audio`.

    Swallows errors on purpose: cleanup happens on the way out of a request
    handler, and a failed delete (file already gone, permissions) must not
    turn a successful synthesis into a 500.  Leftovers are inert — the names
    are unguessable UUIDs.
    """
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass
