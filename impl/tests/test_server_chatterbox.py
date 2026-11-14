"""Tests for ``server_chatterbox.py``.

These exercise the HTTP surface that does NOT require a loaded model:
/capabilities (snapshot), /health, the landing page, request-body validation
(422s), and the reference-audio pre-flight checks (400s).  Synthesis itself
needs a real model + GPU, so it is intentionally out of scope here.
"""

import os
import subprocess
import sys
import types

import pytest
from fastapi.testclient import TestClient

import _bootstrap
import server_chatterbox as srv
from helpers import b64, load_snapshot, make_wav_bytes


@pytest.fixture(scope="module")
def client():
    # Deliberately NOT used as a context manager: entering it would run the
    # FastAPI lifespan, which loads the (stubbed) model.  The endpoints under
    # test here don't need the model, so we skip lifespan entirely.
    return TestClient(srv.app)


# ---------------------------------------------------------------------------
# GET /capabilities — snapshot (single source of truth: the Pydantic model)
# ---------------------------------------------------------------------------


def test_capabilities_matches_snapshot(client):
    response = client.get("/capabilities")
    assert response.status_code == 200
    doc = response.json()
    assert doc == load_snapshot("chatterbox_capabilities.json")


def test_capabilities_core_fields_are_required(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    # The two fields the client must always supply.
    assert by_name["text"]["required"] is True
    assert by_name["audio_base64"]["required"] is True
    # Optional tuning knobs default to the model's own defaults.
    assert by_name["seed"]["required"] is False
    assert by_name["exaggeration"]["default"] == 0.5


def test_capabilities_language_enum_matches_engine_table(client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    assert by_name["language"]["enum"] == list(srv.SUPPORTED_LANGUAGES)
    assert doc["languages"] == list(srv.SUPPORTED_LANGUAGES)


# ---------------------------------------------------------------------------
# GET /health and the landing page
# ---------------------------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["serverType"] == "Chatterbox"
    assert body["device"] == srv.DEVICE
    assert body["model"] == srv.MODEL_LABEL
    # The tts-serve release version comes from the single shared constant;
    # pinning to srv.RELEASE_VERSION (not a literal) keeps this release-proof
    # while still catching a server that forgot to wire it in.
    assert body["version"] == srv.RELEASE_VERSION


def test_root_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Chatterbox" in response.text
    assert "/synthesize" in response.text


# ---------------------------------------------------------------------------
# POST /synthesize — request-body validation (422).  Validation happens before
# the handler runs, so no model is involved.
# ---------------------------------------------------------------------------


def _post(client, payload):
    return client.post("/synthesize", json=payload)


def _valid_payload():
    return {
        "text": "Hello there",
        "audio_base64": b64(make_wav_bytes(3.0)),
    }


def test_synthesize_unknown_field_rejected(client):
    payload = _valid_payload()
    payload["bogus_field"] = 1
    assert _post(client, payload).status_code == 422


def test_synthesize_missing_required_fields_rejected(client):
    assert _post(client, {}).status_code == 422


def test_synthesize_empty_text_rejected(client):
    payload = _valid_payload()
    payload["text"] = ""
    assert _post(client, payload).status_code == 422


def test_synthesize_whitespace_text_rejected(client):
    payload = _valid_payload()
    payload["text"] = "   "
    assert _post(client, payload).status_code == 422


def test_synthesize_empty_audio_rejected(client):
    payload = _valid_payload()
    payload["audio_base64"] = ""
    assert _post(client, payload).status_code == 422


def test_synthesize_unsupported_language_rejected(client):
    payload = _valid_payload()
    payload["language"] = "xx"
    assert _post(client, payload).status_code == 422


# The shared docs/02 language contract (case, names, garbage, non-strings,
# null/empty -> 'en') is asserted once for every server in
# test_language_contract.py; keep only engine-specific language tests here.


def test_synthesize_seed_below_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 0
    assert _post(client, payload).status_code == 422


def test_synthesize_seed_above_range_rejected(client):
    payload = _valid_payload()
    payload["seed"] = 1001
    assert _post(client, payload).status_code == 422


def test_synthesize_exaggeration_above_range_rejected(client):
    payload = _valid_payload()
    payload["exaggeration"] = 5.0
    assert _post(client, payload).status_code == 422


def test_synthesize_cfg_weight_above_range_rejected(client):
    payload = _valid_payload()
    payload["cfg_weight"] = 1.5
    assert _post(client, payload).status_code == 422


# ---------------------------------------------------------------------------
# POST /synthesize — reference-audio pre-flight (400).  The handler runs up to
# the audio check, so the (stubbed) runtime is faked out to skip model load.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setattr(
        srv, "_runtime", types.SimpleNamespace(sample_rate=srv.S3GEN_SR, device=srv.DEVICE)
    )


def test_synthesize_undecodable_audio_rejected(client, fake_runtime):
    payload = _valid_payload()
    payload["audio_base64"] = b64(b"this is definitely not audio")
    response = _post(client, payload)
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_synthesize_too_short_audio_rejected(client, fake_runtime):
    payload = _valid_payload()
    payload["audio_base64"] = b64(make_wav_bytes(0.5))  # 0.5 s < 2.0 s minimum
    response = _post(client, payload)
    assert response.status_code == 400
    assert "2" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Import-time failure: stale chatterbox install (the PyPI 0.1.7 shape)
# ---------------------------------------------------------------------------


def test_import_with_stale_chatterbox_package_raises_with_install_hint(tmp_path):
    # GIVEN a stale chatterbox install, shaped like PyPI chatterbox-tts 0.1.7:
    # the mtl_tts module exists but predates the v3 MULTILINGUAL_T3_MODELS
    # table this server imports -- the import is what tells the two builds
    # apart (both report version 0.1.7).  The fake also omits
    # SUPPORTED_LANGUAGES, which real 0.1.7 does export; harmless, since the
    # first missing name is MULTILINGUAL_T3_MODELS either way.
    stale_pkg = tmp_path / "stale_chatterbox" / "chatterbox"
    stale_pkg.mkdir(parents=True)
    (stale_pkg / "__init__.py").write_text("")
    (stale_pkg / "mtl_tts.py").write_text(
        "S3GEN_SR = 24000\n\n\nclass ChatterboxMultilingualTTS:\n    pass\n"
    )

    # AND a subprocess with the same import environment the suite uses:
    # stale package first, then impl/ + tts-engine-common src + stubs.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(tmp_path / "stale_chatterbox"),
            str(_bootstrap.COMMON_SRC),
            str(_bootstrap.IMPL_DIR),
            str(_bootstrap.STUBS_DIR),
        ]
    )

    # WHEN the server module is imported,
    # THEN it should fail fast with an actionable ImportError that names the
    # pinned git install -- not just a bare "cannot import name" traceback.
    result = subprocess.run(
        [sys.executable, "-c", "import server_chatterbox"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    # THEN the failure is non-zero and the hint is actionable, naming the
    # exact pinned install the server module advertises:
    assert result.returncode != 0
    assert "chatterbox-tts (0.1.7)" in result.stderr
    assert "pip install" in result.stderr
    assert srv._CHATTERBOX_INSTALL in result.stderr


# ---------------------------------------------------------------------------
# Install instructions: the .md must advertise the same pinned commit the
# ImportError message does (the "keep in sync" contract in the server module).
# ---------------------------------------------------------------------------


def test_install_instructions_advertise_the_pinned_commit():
    # The ImportError hint is built from _CHATTERBOX_INSTALL, so a commit
    # bump updates the user-facing error automatically; this pins the install
    # doc to the same constant -- the exact drift that broke the instructions.
    doc = (_bootstrap.IMPL_DIR / "server_chatterbox.md").read_text()
    assert srv._CHATTERBOX_INSTALL in doc
