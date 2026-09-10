"""Cross-server conformance tests for the docs/02 language contract.

Every server in impl/ speaks the same client-facing contract for the
``language`` parameter: two-letter lowercase codes, omitted/null/empty/
whitespace normalizes to ``en``, and anything else (full names, uppercase,
garbage, non-strings) is rejected with a 422 at the request boundary.

The per-server suites keep only what is genuinely engine-specific (enum
contents, capabilities shape, ...). This module asserts the shared half
exactly once per server, so a new server must be registered in ``SERVERS``
below before it can pass the suite with a half-implemented contract.
"""

import types

import pytest
from fastapi.testclient import TestClient

import server_chatterbox
import server_dotsTTS
import server_fasterQwen3TTS
import server_indexTTS
import server_omnivoice
import server_qwen3TTS
from helpers import b64, make_wav_bytes

# (server module, does the engine offer an 'auto' auto-detection sentinel?)
SERVERS = [
    (server_chatterbox, False),
    (server_omnivoice, False),
    (server_qwen3TTS, True),
    (server_fasterQwen3TTS, True),
    (server_dotsTTS, True),
    (server_indexTTS, False),
]
SERVER_IDS = ["chatterbox", "omnivoice", "qwen3-tts", "faster-qwen3-tts", "dots-tts", "index-tts"]


@pytest.fixture(params=SERVERS, ids=SERVER_IDS)
def server_contract(request):
    """The (server module, auto_allowed) pair under test."""
    return request.param


@pytest.fixture
def client(server_contract):
    module, _ = server_contract
    # Deliberately NOT a context manager: entering it would run the FastAPI
    # lifespan, which loads the (stubbed) model.  Not needed for these tests.
    return TestClient(module.app)


@pytest.fixture
def fake_runtime(server_contract, monkeypatch):
    module, _ = server_contract
    # The union of the runtime attributes every synthesize handler touches
    # before its audio pre-flight (mirrors each per-server fake_runtime,
    # minus per-server specifics that the pre-flight never reads).
    monkeypatch.setattr(
        module, "_runtime", types.SimpleNamespace(sample_rate=24000, device="cpu")
    )


def _payload(module):
    # The two fields every server requires.  A real 3 s clip so the payload
    # is valid even for tests where the handler is expected to run (400
    # audio pre-flight); undecodable audio is swapped in where needed.
    payload = {"text": "Hello there", "audio_base64": b64(make_wav_bytes(3.0))}
    # faster-qwen3-tts exposes ICL (advanced) mode only, so its reference
    # transcript is a hard requirement rather than an optional refinement.
    if module is server_fasterQwen3TTS:
        payload["reference_text"] = "A short, exact transcript of the reference clip."
    return payload


def _undecodable_audio_payload(module):
    payload = _payload(module)
    payload["audio_base64"] = b64(b"this is definitely not audio")
    return payload


# ---------------------------------------------------------------------------
# Model level: the null/empty -> 'en' convention (docs/02)
# ---------------------------------------------------------------------------


def test_model_omittedLanguage_defaultsToEnglish(server_contract):
    module, _ = server_contract
    request = module.SynthesisRequest(**_payload(module))
    assert request.language == "en"


@pytest.mark.parametrize("raw_language", [None, "", "   "])
def test_model_emptyLanguage_normalizesToEnglish(server_contract, raw_language):
    module, _ = server_contract
    request = module.SynthesisRequest(**_payload(module), language=raw_language)
    assert request.language == "en"


# ---------------------------------------------------------------------------
# HTTP level: non-conforming values are 422s at the request boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_language",
    [
        pytest.param("x?", id="garbage"),  # docs/02's canonical example
        pytest.param("EN", id="uppercase"),
        pytest.param("english", id="full-name"),  # the old free-form contract
        pytest.param(7, id="int"),
        pytest.param(["en"], id="list"),
        pytest.param({"code": "en"}, id="dict"),
        pytest.param(True, id="bool"),
    ],
)
def test_synthesize_language_nonConforming_rejected(server_contract, client, raw_language):
    module, _ = server_contract
    payload = _payload(module)
    payload["language"] = raw_language
    response = client.post("/synthesize", json=payload)
    assert response.status_code == 422
    # The error must point at the language field, not be some unrelated
    # validation failure:
    assert any("language" in err.get("loc", ()) for err in response.json()["detail"])


# ---------------------------------------------------------------------------
# HTTP level: conforming values pass validation (handler runs, 400 pre-flight)
# ---------------------------------------------------------------------------


def _valid_codes(module):
    """Codes the engine under test actually accepts.

    'en' is the contract default on every server; the other codes come from
    the server's advertised language list.  Engines without a fixed list
    (capabilities ``languages: null``) accept any two-letter code, so the
    historical arbitrary pair applies there.
    """
    languages = module.CAPABILITIES.languages
    if languages is None:
        return ["en", "fr", "de"]
    extras = [code for code in languages if code != "en"]
    return ["en", *extras[:2]]


def test_synthesize_language_validCode_passesValidation(server_contract, client, fake_runtime):
    # A conforming code must clear request validation; with undecodable
    # audio the handler then fails its 400 audio pre-flight — proving we
    # got past validation (actual synthesis needs a real model + GPU).
    module, _ = server_contract
    for code in _valid_codes(module):
        payload = _undecodable_audio_payload(module)
        payload["language"] = code
        assert client.post("/synthesize", json=payload).status_code == 400


def test_synthesize_language_autoSentinel_matchesDeclaration(server_contract, client, fake_runtime):
    module, auto_allowed = server_contract
    payload = _undecodable_audio_payload(module)
    payload["language"] = "auto"
    response = client.post("/synthesize", json=payload)
    if auto_allowed:
        # 'auto' cleared validation, so the handler ran and failed the 400
        # audio pre-flight — the same trick as the valid-code test above.
        assert response.status_code == 400
    else:
        # 'auto' is not a two-letter code; without auto-detection support
        # it must 422 at the boundary.
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /capabilities: the contract is advertised in the document
# ---------------------------------------------------------------------------


def test_capabilities_language_contract(server_contract, client):
    doc = client.get("/capabilities").json()
    by_name = {p["name"]: p for p in doc["parameters"]}
    language = by_name["language"]
    assert language["type"] == "string"
    assert language["required"] is False
    # docs/02: null/empty means English — the default advertises it.
    assert language["default"] == "en"
