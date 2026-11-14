"""Cross-server conformance tests for the ``<PREFIX>_DEVICE`` contract.

tts-serve has one **grammar** for the device environment variables (shared,
``tts_engine_common.devices``, docs/01 decision D8) and per-engine
**capability** declared as data by each server.  Before that, six hand-rolled
variants answered ``cuda:9`` six different ways: only Audio8 checked the
ordinal against ``torch.cuda.device_count()``, two accepted anything that
merely *started with* "cuda", and four could not address a second GPU at all.

What is asserted once here, for every server:

* the module states its device posture explicitly -- either an env var with
  ``DEVICE_ENV`` / ``DEVICE_FAMILIES`` / ``DEVICE_ALLOW_INDEX`` / ``DEVICE_SPEC``,
  or ``DEVICE_ENV = None`` for "this engine has no device knob";
* that declaration is the variable the module actually reads, and the shared
  parser is what interprets it (no pasted regex, no ``startswith("cuda")``);
* an index is accepted exactly when the server declares it is, and junk is a
  startup failure naming the offending variable;
* a CUDA request the box cannot serve is refused *before* any model load, with
  one shared wording -- unless the server explicitly declares that its engine
  resolves availability itself (``DEVICE_ENGINE_RESOLVES``).

A new server must be registered in ``SERVER_MODULES`` below;
``test_registeredServersCoverEveryImplServerScript`` is what makes forgetting
that registration fail the suite even if the new server's own suite passes.

The grammar probes run in a subprocess because these servers read their
environment at *import* time, and an in-process re-import would also replay
the pydantic core-schema caching the language contract tests exist to guard.
"""

import ast
import inspect
import json
import os
import re
import subprocess
import sys

import pytest
from _bootstrap import COMMON_SRC, IMPL_DIR, STUBS_DIR
from tts_engine_common.devices import (
    DEVICE_FAMILIES as GRAMMAR_FAMILIES,
    DeviceSpec,
    parse_device,
)

import server_audio8TTS
import server_breezeBlue
import server_chatterbox
import server_dotsTTS
import server_fasterQwen3TTS
import server_indexTTS
import server_kittenTTS
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
    server_audio8TTS,
    server_kittenTTS,
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
    "audio8-tts",
    "kitten-tts",
]

# Servers whose engine answers the availability question itself.  A server-side
# "this box has no CUDA" guard would pre-empt a fallback that works (LuxTTS:
# zipvoice/luxvoice.py falls back cuda -> mps -> cpu), so the posture must be
# declared rather than quietly missing.
ENGINE_RESOLVES = "DEVICE_ENGINE_RESOLVES"

# Spelled like a device, but not one.  'cuda-ish' and 'cuda:' are what the old
# `startswith("cuda")` checks used to accept.  An empty value is deliberately
# absent: leaving INDEXTTS_DEVICE unset means 'auto' there, by documentation.
# Case is also absent: folding is the server's choice (KittenTTS lower-cases
# what it reads), not the grammar's.
_BAD_VALUES = ["tpu", "cuda-ish", "cuda:", "cuda:1:2", "cpu:0", "mps:0"]

# Far past any GPU a box could plausibly have.
_OUT_OF_RANGE = "cuda:99"

# Legally spelled, but padded -- what an env file or a shell quoting slip
# produces.  The grammar strips it, so the question this raises is whether the
# *stripped* value is the one the server then uses (issue #73 review finding).
_PADDED = " cuda "


@pytest.fixture(params=SERVER_MODULES, ids=SERVER_IDS)
def server_module(request):
    """The server module under test."""
    return request.param


def _source(server_module) -> str:
    """The module's source, minus its docstring and comment lines.

    Prose about devices is exactly what this repo documents thoroughly, so any
    check of "did the code hand-roll a grammar" has to look at code only.
    """
    src = inspect.getsource(server_module)
    tree = ast.parse(src)
    if tree.body and isinstance(tree.body[0], ast.Expr):
        node = tree.body[0].value
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lines = src.splitlines(keepends=True)
            del lines[node.lineno - 1 : node.end_lineno]
            src = "".join(lines)
    return "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )


def _has_device_knob(server_module) -> bool:
    return getattr(server_module, "DEVICE_ENV", None) is not None


def _guards_availability(server_module) -> bool:
    """Whether this server checks device availability itself."""
    return not getattr(server_module, ENGINE_RESOLVES, False)


def _pin_cuda(monkeypatch, server_module, available: bool, count: int) -> None:
    """Pin the CUDA probes so these tests never inherit the host's GPU."""
    monkeypatch.setattr(server_module.torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(server_module.torch.cuda, "device_count", lambda: count)


# ---------------------------------------------------------------------------
# Registration: every impl/server_*.py is covered by this contract.
# ---------------------------------------------------------------------------


def test_registeredServersCoverEveryImplServerScript():
    # GIVEN the server scripts that actually exist on disk,
    # THEN this contract's registry covers exactly those modules -- an engine
    # that skips registration fails here even when its own suite passes (the
    # same trap test_language_contract / test_versioning_contract document).
    on_disk = sorted(path.stem for path in IMPL_DIR.glob("server_*.py"))
    assert sorted(module.__name__ for module in SERVER_MODULES) == on_disk


# ---------------------------------------------------------------------------
# Declared posture: a knob with a name, or an explicit "no knob".
# ---------------------------------------------------------------------------


def test_serverDeclaresDevicePostureExplicitly(server_module):
    # GIVEN any server at all,
    # THEN it states its device posture: a device knob needs the four shared
    # declarations, and a server with no knob says so with DEVICE_ENV = None
    # rather than leaving the reader to guess which of the two happened.
    assert hasattr(server_module, "DEVICE_ENV"), (
        f"{server_module.__name__} must declare DEVICE_ENV: the env var it "
        "reads, or None if its engine has no device knob"
    )
    if server_module.DEVICE_ENV is None:
        assert not hasattr(server_module, "DEVICE_SPEC")
        assert not hasattr(server_module, "_check_device_available")
    else:
        for name in ("DEVICE_FAMILIES", "DEVICE_ALLOW_INDEX", "DEVICE_SPEC"):
            assert hasattr(server_module, name), (
                f"{server_module.__name__} reads {server_module.DEVICE_ENV} "
                f"but does not declare {name}"
            )


def test_declaredEnvVarIsTheOneTheModuleActuallyReads(server_module):
    # GIVEN the *_DEVICE names the module's code mentions,
    literals = set(
        re.findall(r"[\"']([A-Za-z0-9_]*_DEVICE)[\"']", _source(server_module))
    )

    # THEN the declared name is the one it reads, and nothing else is read
    # behind the declaration: a module that declares LUX_TTS_DEVICE but reads
    # LUX_DEVICE would pass every other check in this file while validating
    # nothing at all.
    declared = server_module.DEVICE_ENV
    assert literals == (set() if declared is None else {declared})


def test_declaredFamiliesAreGrammarFamilies(server_module):
    # GIVEN a server with a device knob,
    # THEN its capability declaration uses only families the shared grammar
    # knows about -- a typo there is a bug in the server, not a device, and it
    # must not silently widen or narrow what the server accepts.
    if not _has_device_knob(server_module):
        pytest.skip("server has no device knob")
    assert server_module.DEVICE_FAMILIES
    assert set(server_module.DEVICE_FAMILIES) <= set(GRAMMAR_FAMILIES)
    assert isinstance(server_module.DEVICE_ALLOW_INDEX, bool)
    assert server_module.DEVICE_SPEC.family in server_module.DEVICE_FAMILIES


def test_declaredSpecMatchesThePinnedEnvironment(server_module):
    # GIVEN the test environment pins this server's device variable (the values
    # _bootstrap pins, which are also what the committed /capabilities
    # snapshots were generated from),
    if not _has_device_knob(server_module):
        pytest.skip("server has no device knob")
    raw = os.environ.get(server_module.DEVICE_ENV)
    if raw is None:
        pytest.skip(
            f"{server_module.DEVICE_ENV} is deliberately unset in the pinned "
            "environment (that server's auto-selection path)"
        )

    # WHEN the module's own declarations are applied to that value,
    # THEN they reproduce the spec the module built for itself: the data this
    # test reads and the data the server reads cannot drift apart.
    assert server_module.DEVICE_SPEC == parse_device(
        raw,
        env_var=server_module.DEVICE_ENV,
        families=server_module.DEVICE_FAMILIES,
        allow_index=server_module.DEVICE_ALLOW_INDEX,
    )


# ---------------------------------------------------------------------------
# Grammar probes: real imports, one fresh interpreter per server.
# ---------------------------------------------------------------------------

_PROBE_SCRIPT = """
import importlib, json, os, sys

module, env_var = sys.argv[1], sys.argv[2]
report = []
for raw in json.loads(sys.argv[3]):
    os.environ[env_var] = raw
    sys.modules.pop(module, None)
    try:
        loaded = importlib.import_module(module)
    except BaseException as exc:  # a startup refusal is a probe result, not a crash
        report.append({"value": raw, "imported": False,
                       "error": type(exc).__name__ + ": " + str(exc)})
    else:
        spec = getattr(loaded, "DEVICE_SPEC", None)
        report.append({"value": raw, "imported": True,
                       "spec": None if spec is None else [spec.family, spec.index],
                       # the string this server hands to its engine and reports
                       # in /health + /capabilities
                       "device": getattr(loaded, "DEVICE", None)})
print("DEVICE-PROBE-REPORT " + json.dumps(report))
"""


@pytest.fixture
def device_probes(server_module):
    """Import-time probe results for one server, from a fresh interpreter."""
    if not _has_device_knob(server_module):
        pytest.skip("server has no device knob")
    configured = os.environ.get(server_module.DEVICE_ENV, "cuda")
    values = [configured, *_BAD_VALUES, _OUT_OF_RANGE, "cuda:1", _PADDED]

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(COMMON_SRC), str(IMPL_DIR), str(STUBS_DIR)]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE_SCRIPT,
            server_module.__name__,
            server_module.DEVICE_ENV,
            json.dumps(values),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"the probe interpreter itself failed: {result.stderr[-2000:]}"
    )
    line = next(
        (
            line
            for line in result.stdout.splitlines()
            if line.startswith("DEVICE-PROBE-REPORT ")
        ),
        None,
    )
    assert line is not None, f"probe produced no report: {result.stdout[-2000:]}"
    return {entry["value"]: entry for entry in json.loads(line.split(" ", 1)[1])}


def test_probeHarnessCanImportTheServerAtAll(server_module, device_probes):
    # GIVEN the device value the test environment pins,
    # WHEN the server is imported with it,
    # THEN it imports.  Without this, every "refused to start" assertion below
    # would also be satisfied by an unrelated import failure (a missing stub,
    # say) -- a green suite meaning nothing.
    configured = os.environ.get(server_module.DEVICE_ENV, "cuda")
    probe = device_probes[configured]
    assert probe["imported"], probe["error"]


def test_malformedDeviceValues_areRefusedNamingTheEnvVar(server_module, device_probes):
    # GIVEN each malformed spelling of the variable,
    # WHEN the server module is imported with it,
    # THEN the import fails (a startup refusal, not a torch error during a
    # download) and the message names *this server's* variable -- an operator
    # has to be able to tell which of fifteen servers refused to start.
    for value in _BAD_VALUES:
        probe = device_probes[value]
        assert not probe["imported"], (
            f"{server_module.DEVICE_ENV}={value!r} was accepted"
        )
        assert server_module.DEVICE_ENV in probe["error"], probe["error"]
        assert repr(value) in probe["error"], (
            f"{server_module.DEVICE_ENV} message does not quote the bad value: "
            f"{probe['error']}"
        )


def test_indexedCudaIsAcceptedExactlyWhenDeclared(server_module, device_probes):
    # GIVEN a well-formed request for the second GPU,
    # WHEN the module is imported,
    # THEN it is accepted exactly when the server declares that its engine can
    # address one -- and when accepted, the parsed spec really carries the
    # ordinal, so "accepted" cannot mean "silently truncated to cuda".
    probe = device_probes["cuda:1"]
    if server_module.DEVICE_ALLOW_INDEX:
        assert probe["imported"], probe["error"]
        assert probe["spec"] == ["cuda", 1]
    else:
        assert not probe["imported"], (
            f"{server_module.DEVICE_ENV}=cuda:1 was accepted, but the server "
            "declares DEVICE_ALLOW_INDEX = False"
        )
        assert server_module.DEVICE_ENV in probe["error"], probe["error"]


def test_whitespacePaddedDeviceValue_isStrippedInTheStringTheServerUses(
    server_module, device_probes
):
    # GIVEN a device value carrying stray whitespace, which the grammar accepts
    # on purpose (stripping is the friendly reading of an env-file slip),
    # WHEN the server imports it,
    # THEN the parse really sees 'cuda' AND the module-level DEVICE the server
    # hands to its engine / reports in /health and /capabilities is the stripped
    # string too.  Validating a normalized copy while passing the raw one to
    # torch is the exact drift the shared grammar exists to prevent -- and a
    # padded value would otherwise fail *inside* the model load, with a message
    # blaming the engine for the shell's typo.
    probe = device_probes[_PADDED]
    assert probe["imported"], probe["error"]
    assert probe["spec"] == ["cuda", None], probe
    assert probe["device"] == "cuda", (
        f"{server_module.__name__} validated 'cuda' but uses "
        f"{probe['device']!r} downstream"
    )


def test_indexGrammarIsCheckedWithoutProbingTheMachine(server_module, device_probes):
    # GIVEN an index far beyond any GPU this box could have,
    # WHEN the module is imported,
    # THEN it imports.  Import time answers "is that a device we can talk
    # about"; only load time may answer "does this machine have it".  The
    # pinned test environment has no GPU at all, so an import-time probe would
    # make this fail and leave a CPU-only box unable to serve /capabilities.
    if not server_module.DEVICE_ALLOW_INDEX:
        pytest.skip("server declares no index support")
    probe = device_probes[_OUT_OF_RANGE]
    assert probe["imported"], probe["error"]
    assert probe["spec"] == ["cuda", 99]


# ---------------------------------------------------------------------------
# The load-time guard: same verdict, same wording, every guarded server.
# ---------------------------------------------------------------------------


def _skip_unless_guarded(server_module) -> None:
    if not _has_device_knob(server_module) or not _guards_availability(server_module):
        pytest.skip("server has no knob, or leaves availability to its engine")


def test_guardRefusesAnOrdinalTheBoxDoesNotHave(server_module, monkeypatch):
    # GIVEN a server that guards availability itself, a one-GPU box, and a
    # request for GPU 9 (torch.device('cuda:9') constructs happily and
    # is_available() only says *some* GPU exists -- that is the whole gap):
    _skip_unless_guarded(server_module)
    monkeypatch.setattr(server_module, "DEVICE_SPEC", DeviceSpec("cuda", 9, "cuda:9"))
    _pin_cuda(monkeypatch, server_module, available=True, count=1)

    # WHEN the guard runs,
    with pytest.raises(RuntimeError) as excinfo:
        server_module._check_device_available()

    # THEN it names the variable, the requested ordinal, and what the box has,
    # in one shared wording: the operator learns the answer once, and a log
    # grep does not need fifteen patterns.
    message = str(excinfo.value)
    assert server_module.DEVICE_ENV in message
    assert "asks for GPU 9, but this box exposes 1 CUDA device" in message


def test_guardRefusesCudaOnABoxWithoutIt(server_module, monkeypatch):
    # GIVEN an explicit CUDA request on a box with no CUDA: an explicit request
    # must fail loudly, because unlike 'auto', nobody asked it to make do.
    _skip_unless_guarded(server_module)
    monkeypatch.setattr(server_module, "DEVICE_SPEC", DeviceSpec("cuda", None, "cuda"))
    _pin_cuda(monkeypatch, server_module, available=False, count=0)

    # WHEN the guard runs, THEN the refusal says so, naming the variable:
    with pytest.raises(RuntimeError) as excinfo:
        server_module._check_device_available()
    message = str(excinfo.value)
    assert server_module.DEVICE_ENV in message
    assert "requests CUDA, but CUDA is not available" in message


def test_guardIgnoresAutoOnABoxWithoutIt(server_module, monkeypatch):
    # GIVEN 'auto' on a CPU-only box -- for a server that offers it, "no GPU" is
    # a valid answer, not a failure:
    _skip_unless_guarded(server_module)
    if "auto" not in server_module.DEVICE_FAMILIES:
        pytest.skip("server's engine offers no auto-selection")
    monkeypatch.setattr(server_module, "DEVICE_SPEC", DeviceSpec("auto", None, "auto"))
    _pin_cuda(monkeypatch, server_module, available=False, count=0)

    # WHEN the guard runs, THEN it stays out of the way (the CPU fallback is the
    # entire reason 'auto' exists).
    server_module._check_device_available()


def test_guardRunsInsideTheLoadPath(server_module):
    # GIVEN a guarded server,
    # WHEN its source is read,
    # THEN the guard is defined *and* called -- the promise that a device this
    # box cannot serve never reaches the checkpoint download has to be
    # structural, not prose in a docstring.
    _skip_unless_guarded(server_module)
    src = _source(server_module)
    assert src.count("_check_device_available(") >= 2, (
        f"{server_module.__name__} defines _check_device_available but never "
        "calls it from its load path"
    )


def test_engineResolvedServersDeclareTheFallbackAndIndexNothing(server_module):
    # GIVEN a server that leaves availability to its engine,
    # THEN it says so (DEVICE_ENGINE_RESOLVES) and builds no guard of its own --
    # a guard would pre-empt a fallback that works.  It must not accept an
    # index either: an engine that resolves the device itself cannot be assumed
    # to interpret an ordinal, and quietly ignoring one is worse than refusing
    # it (LuxTTS's loader really does hard-code GPU 0).
    if not getattr(server_module, ENGINE_RESOLVES, False):
        pytest.skip("server guards availability itself")
    assert server_module.DEVICE_ALLOW_INDEX is False
    assert not hasattr(server_module, "_check_device_available")

    # AND the operator is told how to pick a GPU instead, in the docstring the
    # /docs page renders:
    assert "CUDA_VISIBLE_DEVICES" in (server_module.__doc__ or ""), (
        f"{server_module.__name__} rejects cuda:<index>, so its docstring must "
        "name CUDA_VISIBLE_DEVICES as the alternative"
    )


# ---------------------------------------------------------------------------
# Anti-regression: the grammar lives once.
# ---------------------------------------------------------------------------

_FORBIDDEN_PATTERNS = [
    # The hole the issue calls out: it accepted 'cuda-ish' and 'cuda:'.
    (r"\.startswith\(\s*[\"']cuda", "startswith('cuda') as a device check"),
    # A pasted device regex is the failure mode the shared grammar exists to
    # prevent; the shared parser is the only allowed place to own one.
    (r"re\.compile\([^)]*(?:cuda|mps|xpu)", "a hand-rolled device regex"),
    # The device-name membership tuple: the variant that could not address
    # GPU 1 at all, and that differed per server for no reason.  Anchored on
    # the variable name so the one legal tuple of device names -- the
    # capability declaration DEVICE_FAMILIES -- is not itself the finding.
    (
        r"(?:[A-Z_]*_)?DEVICE(?:S|_DEVICES)?\s*=\s*\(\s*[\"'](?:cuda|mps|cpu|xpu)[\"']\s*,",
        "a device-name tuple",
    ),
]


def test_noServerHandRollsADeviceGrammar(server_module):
    # GIVEN the server's own code,
    # THEN none of the six historical variants reappears.  This is the test a
    # new engine fails by copy-pasting an old server instead of declaring its
    # capability through the shared parser.
    src = _source(server_module)
    for pattern, why in _FORBIDDEN_PATTERNS:
        found = re.search(pattern, src)
        assert found is None, f"{server_module.__name__} reintroduces {why}: {found}"

    if _has_device_knob(server_module):
        assert "parse_device(" in src, (
            f"{server_module.__name__} validates its device value without the "
            "shared parser"
        )


def test_serversWithoutAKnobDoNotReadOne(server_module):
    # GIVEN a server that declares no device knob,
    # THEN it also reads no *_DEVICE variable and defines no guard: the three
    # engines without a knob (dots.tts auto-selects, Pocket TTS is CPU-only,
    # mlx-audio has no such argument) stay that way on purpose, and a knob
    # added later must come with the declarations above.
    if _has_device_knob(server_module):
        pytest.skip("server has a device knob")
    src = _source(server_module)
    assert not re.search(r"os\.getenv\(\s*[\"'][A-Za-z0-9_]*_DEVICE[\"']", src)
    assert not hasattr(server_module, "_check_device_available")
