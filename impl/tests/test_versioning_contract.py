"""Cross-server conformance tests for the tts-serve versioning scheme.

tts-serve carries **two independent version numbers**, and this module is
the guard rail that keeps them independent *and* consistently reported:

* ``RELEASE_VERSION`` (str, e.g. ``"1.5"``) — the release line.  Bumped at
  the start of every release cycle, backwards-compatible or not.  Every
  server reports it in ``GET /health`` (``version`` field) and carries it
  in its FastAPI app metadata (visible in ``/docs`` / ``/openapi.json``).
  It must match the README changelog heading and the git tag.

* ``SCHEMA_VERSION`` (int, e.g. ``2``) — the client-contract version,
  reported as ``schema_version`` in ``GET /capabilities`` only.  Bumped
  *only* for breaking changes to the capabilities document / request
  vocabulary (docs/01 section 3.5).  Adding an engine or an additive
  parameter must NOT bump it.

Confusing the two is the failure mode this prevents: a release that only
adds engines must not scare version-gating clients (speak.py hard-aborts
on an unknown schema_version), and a breaking release must not hide behind
a release-number bump that no client inspects.

A new server must be registered in ``SERVER_MODULES`` below — an unwired
server fails here even if its own suite happens to pass.
"""

import pytest
from fastapi.testclient import TestClient

from tts_engine_common import RELEASE_VERSION, SCHEMA_VERSION
from tts_engine_common.models import CoreHealthResponse

import server_breezeBlue
import server_chatterbox
import server_dotsTTS
import server_fasterQwen3TTS
import server_indexTTS
import server_luxTTS
import server_ming3TTS
import server_ming3TTS_moe
import server_omnivoice
import server_pocketTTS
import server_qwen3TTS
import server_qwen3TTS_mlx
import server_voxcpm

SERVER_MODULES = [
    server_chatterbox,
    server_omnivoice,
    server_qwen3TTS,
    server_qwen3TTS_mlx,
    server_fasterQwen3TTS,
    server_dotsTTS,
    server_indexTTS,
    server_luxTTS,
    server_ming3TTS,
    server_ming3TTS_moe,
    server_voxcpm,
    server_breezeBlue,
    server_pocketTTS,
]
SERVER_IDS = [
    "chatterbox",
    "omnivoice",
    "qwen3-tts",
    "qwen3-tts-mlx",
    "faster-qwen3-tts",
    "dots-tts",
    "index-tts",
    "lux-tts",
    "ming3-tts",
    "ming3-tts-moe",
    "voxcpm",
    "breeze-blue",
    "pocket-tts",
]


@pytest.fixture(params=SERVER_MODULES, ids=SERVER_IDS)
def server_module(request):
    """The server module under test."""
    return request.param


def test_health_reportsSharedReleaseVersion(server_module):
    # Every server's /health must report the *shared* constant: one bump in
    # tts_engine_common.core changes it everywhere, and a server that
    # hard-codes its own version (the old 0.1.0/0.2.0/0.3.0 zoo) fails here.
    client = TestClient(server_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["version"] == RELEASE_VERSION


def test_healthResponseSubclassesSharedCoreHealthResponse(server_module):
    # The /health core (status/model/device/version) lives once in
    # CoreHealthResponse; a server that hand-rolls its own health model
    # from BaseModel reintroduces the ten-fold copy-paste the base model
    # exists to prevent, and fails here.
    assert issubclass(server_module.HealthResponse, CoreHealthResponse)


def test_appMetadataCarriesReleaseVersion(server_module):
    # The FastAPI app metadata (surfaced in /openapi.json) must carry the
    # same constant — the openapi contract is part of the docs some clients
    # read, and the old hand-maintained stamps drifted silently.
    assert server_module.app.version == RELEASE_VERSION


def test_capabilitiesDoNotLeakReleaseVersion(server_module):
    # /capabilities is derived from the request model and advertises only
    # the contract version.  The release version is a /health concern;
    # leaking it into the derived document would invite clients to gate on
    # a number that changes on non-breaking releases.
    client = TestClient(server_module.app)
    doc = client.get("/capabilities").json()
    assert doc["schema_version"] == SCHEMA_VERSION
    assert "version" not in doc
    assert "release_version" not in doc


def test_releaseVersionIsWellFormed():
    # Matches the git tags ("1.4", "1.5", ...): dotted numeric segments,
    # no 'v' prefix.  Anything else means someone typed the README heading
    # instead of the tag form.
    segments = RELEASE_VERSION.split(".")
    assert len(segments) >= 2
    assert all(segment.isdigit() for segment in segments)


def test_releaseAndSchemaVersionsAreIndependentConstants():
    # Different concepts, different types: release ("1.5") vs contract (2).
    # If someone "fixes" these to match, version-gating clients break.
    assert isinstance(RELEASE_VERSION, str)
    assert isinstance(SCHEMA_VERSION, int)
    assert not isinstance(SCHEMA_VERSION, bool)
