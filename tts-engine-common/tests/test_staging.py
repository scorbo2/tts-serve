"""Tests for ``tts_engine_common.staging``.

The staging helpers move prompt audio to disk for engines that load it from
a file path.  They are plain filesystem functions — no model, no torch, no
engine package.
"""

import hashlib
import os
import tempfile
import threading
from pathlib import Path

import pytest

from tts_engine_common import (
    cleanup_temp,
    stage_audio,
    temp_audio_dir,
    write_temp_audio,
)

CLIP_A = b"clip-a-bytes"
CLIP_B = b"clip-b-bytes"


@pytest.fixture
def staging(tmp_path):
    """A per-test staging directory (already created)."""
    directory = tmp_path / "staging"
    directory.mkdir()
    return directory


# ---------------------------------------------------------------------------
# temp_audio_dir
# ---------------------------------------------------------------------------


def test_temp_audio_dir_withNewName_createsDirectoryUnderSystemTemp(tmp_path, monkeypatch):
    # GIVEN the system temp dir is pinned to a sandbox:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    # WHEN we ask for a staging directory,
    # THEN it is created (and returned) under that temp dir:
    directory = temp_audio_dir("my_engine_rest_api")

    assert directory == tmp_path / "my_engine_rest_api"
    assert directory.is_dir()


def test_temp_audio_dir_withExistingDirectory_isIdempotent(tmp_path, monkeypatch):
    # GIVEN the directory already exists:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    first = temp_audio_dir("my_engine_rest_api")

    # WHEN we ask for it again,
    # THEN the same path is returned and no error is raised:
    second = temp_audio_dir("my_engine_rest_api")

    assert second == first


# ---------------------------------------------------------------------------
# stage_audio — content-addressed, atomic, kept
# ---------------------------------------------------------------------------


def test_stage_audio_withNewContent_writesContentAddressedFile(staging):
    # WHEN we stage fresh bytes,
    # THEN the returned path is <sha256>.wav inside the directory and
    # holds exactly those bytes:
    path = stage_audio(CLIP_A, staging)

    digest = hashlib.sha256(CLIP_A).hexdigest()
    assert Path(path) == staging / f"{digest}.wav"
    assert Path(path).read_bytes() == CLIP_A


def test_stage_audio_withSameContentTwice_reusesPath_andSkipsRewrite(staging):
    # GIVEN the clip is staged once:
    first = stage_audio(CLIP_A, staging)
    mtime_before = Path(first).stat().st_mtime_ns

    # WHEN we stage the identical clip again,
    # THEN the path is reused and the file is not rewritten
    # (a rewrite would bump mtime_ns and re-encode the same bytes for nothing):
    second = stage_audio(CLIP_A, staging)

    assert second == first
    assert Path(first).stat().st_mtime_ns == mtime_before


def test_stage_audio_withDifferentContent_writesDistinctFiles(staging):
    # GIVEN two distinct clips:
    # WHEN we stage each,
    # THEN the content hashes differ, so do the paths and contents:
    path_a = stage_audio(CLIP_A, staging)
    path_b = stage_audio(CLIP_B, staging)

    assert path_a != path_b
    assert Path(path_a).read_bytes() == CLIP_A
    assert Path(path_b).read_bytes() == CLIP_B
    assert len(list(staging.iterdir())) == 2


def test_stage_audio_withNonWavBytes_stillUsesWavExtension(staging):
    # The .wav extension is cosmetic — the engine sniffs the header, so any
    # container is fine under a .wav name:
    path = stage_audio(b"\xff\xfe not really a wav", staging)

    assert Path(path).suffix == ".wav"


def test_stage_audio_afterSuccessfulWrite_leavesNoTmpFiles(staging):
    # GIVEN a freshly staged clip:
    stage_audio(CLIP_A, staging)

    # THEN no atomic-rename staging files leaked:
    remaining = [p.name for p in staging.iterdir()]

    assert not any(name.endswith(".tmp") for name in remaining)


def test_stage_audio_whenRenameFails_propagatesAndCleansUpTmpFile(staging, monkeypatch):
    # GIVEN the atomic rename is made to fail (e.g. cross-device or permissions):
    def _boom(*args, **kwargs):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", _boom)

    # WHEN we stage a clip,
    # THEN the error propagates and no half-written .tmp file is left behind:
    with pytest.raises(OSError, match="simulated rename failure"):
        stage_audio(CLIP_A, staging)

    assert list(staging.iterdir()) == []


def test_stage_audio_whenManyThreadsStageSameNewClip_allConvergeOnOneValidFile(staging):
    # GIVEN several workers staging the *same* never-seen clip concurrently
    # (the exact race the shared content-addressed name invites):
    errors: list[BaseException] = []
    results: list[str] = []

    def worker():
        try:
            results.append(stage_audio(CLIP_A, staging))
        except BaseException as exc:  # pragma: no cover - diagnostics only
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # THEN every worker agrees on the single content-addressed path, the
    # file is intact, and no .tmp siblings leaked:
    assert not errors
    assert len(set(results)) == 1
    assert Path(results[0]).read_bytes() == CLIP_A
    assert [p.name for p in staging.iterdir()] == [Path(results[0]).name]


# ---------------------------------------------------------------------------
# write_temp_audio / cleanup_temp — UUID, caller-owned
# ---------------------------------------------------------------------------


def test_write_temp_audio_withSameContentTwice_returnsDistinctPaths(staging):
    # UUID names: even identical clips get distinct files, because the
    # caller deletes each one after its engine has read it:
    path_a = write_temp_audio(CLIP_A, staging)
    path_b = write_temp_audio(CLIP_A, staging)

    assert path_a != path_b
    assert Path(path_a).read_bytes() == CLIP_A
    assert Path(path_b).read_bytes() == CLIP_A


def test_write_temp_audio_writesExactContent(staging):
    path = write_temp_audio(CLIP_B, staging)

    assert Path(path).read_bytes() == CLIP_B
    assert Path(path).suffix == ".wav"


def test_cleanup_temp_withExistingFile_removesIt(staging):
    # GIVEN a temp audio file exists:
    path = write_temp_audio(CLIP_A, staging)

    # WHEN we clean it up,
    # THEN it is gone:
    cleanup_temp(path)

    assert not Path(path).exists()


def test_cleanup_temp_withMissingFile_doesNotRaise(staging):
    # Missing files are fine (a concurrent cleanup may already have removed
    # the file):
    cleanup_temp(str(staging / "never_existed.wav"))


def test_cleanup_temp_whenUnlinkFails_swallowsOSError(staging):
    # A path whose unlink raises OSError (a directory, on POSIX) must not
    # turn a successful synthesis into a 500:
    cleanup_temp(staging)
