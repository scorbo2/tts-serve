"""GPU-free unit tests for tools/speak.py.

Covers the areas docs/03-speak-script.md calls out: capabilities-parser
mapping, payload assembly, persona-dir resolution, base64 handling, and
reference-text whitespace collapsing, plus end-to-end ``main()`` flows with
the network and aplay stubbed out.  The capabilities-driven tests run
against the real committed server snapshots in impl/tests/snapshots/, so a
server schema change that breaks the mapping shows up here.
"""

import argparse
import base64
import json
import os
import types
from pathlib import Path

import pytest

import speak

SNAPSHOTS = Path(__file__).resolve().parents[2] / "impl" / "tests" / "snapshots"


def _caps(filename: str) -> dict:
    return json.loads((SNAPSHOTS / filename).read_text(encoding="utf-8"))


def _dots_parser():
    parser = speak.build_base_parser()
    names = speak.add_dynamic_arguments(parser, _caps("dots_capabilities.json"))
    return parser, names


def _stage_two_args(parser, extra: list[str] | None = None):
    """A command line that satisfies all stage-1 required arguments."""
    base = ["Hello", "--server", "http://10.0.0.5:8000", "--ref-audio", "a.wav"]
    return parser.parse_args(base + (extra or []))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_kebab_withUnderscores_replacesWithDashes():
    # GIVEN a server parameter name with underscores
    # WHEN mapped to a CLI flag
    # THEN it follows the kebab-case convention
    assert speak.kebab("num_steps") == "num-steps"


def test_kebab_withoutUnderscores_isUnchanged():
    assert speak.kebab("guidance") == "guidance"


@pytest.mark.parametrize("word", ["true", "yes", "1", "on", "True", "ON"])
def test_str_to_bool_withTruthyWord_returnsTrue(word):
    assert speak.str_to_bool(word) is True


@pytest.mark.parametrize("word", ["false", "no", "0", "off", "False", "OFF"])
def test_str_to_bool_withFalsyWord_returnsFalse(word):
    assert speak.str_to_bool(word) is False


def test_str_to_bool_withNonBooleanWord_raisesArgumentTypeError():
    # GIVEN a string that is not in the true/false word lists
    # WHEN parsed as a boolean
    # THEN argparse gets an ArgumentTypeError (which becomes exit 2)
    with pytest.raises(argparse.ArgumentTypeError):
        speak.str_to_bool("test")


def test_join_url_withTrailingAndLeadingSlashes_normalizesToSingleSlash():
    assert (
        speak.join_url("http://10.0.0.5:8000/", "/synthesize")
        == "http://10.0.0.5:8000/synthesize"
    )


def test_join_url_withNoSlashes_addsSeparator():
    assert speak.join_url("http://10.0.0.5:8000", "synthesize") == "http://10.0.0.5:8000/synthesize"


def test_load_audio_base64_roundTripsOriginalBytes(tmp_path):
    # GIVEN a binary audio file
    # WHEN base64-encoded by the script
    # THEN decoding yields the original bytes
    data = bytes(range(256)) * 10
    path = tmp_path / "ref.wav"
    path.write_bytes(data)
    assert base64.b64decode(speak.load_audio_base64(str(path))) == data


def test_load_transcript_withMessyWhitespace_collapsesToSingleSpaces(tmp_path):
    # GIVEN a transcript file with leading/trailing whitespace and newlines
    # WHEN loaded
    # THEN all whitespace runs collapse to single spaces
    path = tmp_path / "ref.txt"
    path.write_text("  Hello,   world.\n\nLine two.\n", encoding="utf-8")
    assert speak.load_transcript(str(path)) == "Hello, world. Line two."


def test_load_transcript_withNonUtf8File_raisesSpeakError(tmp_path):
    path = tmp_path / "ref.txt"
    path.write_bytes(b"\xff\xfe\x00binary")
    with pytest.raises(speak.SpeakError, match=r"could not be read as UTF-8"):
        speak.load_transcript(str(path))


def test_env_or_none_withUnsetVariable_returnsNone():
    # GIVEN TTS_SPEAK_SERVER unset (the conftest fixture guarantees this)
    # WHEN read via _env_or_none
    # THEN it yields None, the "unset" signal the callers check for
    assert speak._env_or_none(speak.SERVER_ENV_VAR) is None


@pytest.mark.parametrize("value", ["", "   "])
def test_env_or_none_withBlankVariable_returnsNone(monkeypatch, value):
    # GIVEN the variable set to an empty or whitespace-only string
    # WHEN read
    # THEN it is treated as unset (spec: blank values must not masquerade as
    #      a configured value and die later as 'Failed to connect')
    monkeypatch.setenv(speak.SERVER_ENV_VAR, value)
    assert speak._env_or_none(speak.SERVER_ENV_VAR) is None


def test_env_or_none_withValue_stripsSurroundingWhitespace(monkeypatch):
    # GIVEN the variable set to a value with accidental padding
    monkeypatch.setenv(speak.SERVER_ENV_VAR, "  http://10.0.0.5:8000  ")
    # WHEN read
    # THEN the value comes back stripped
    assert speak._env_or_none(speak.SERVER_ENV_VAR) == "http://10.0.0.5:8000"


@pytest.mark.parametrize("explicit", [None, "", "   "])
def test_resolve_server_withUnsetOrBlankFlag_fallsBackToEnvVar(monkeypatch, explicit):
    # GIVEN no --server (None) or a blank one, plus TTS_SPEAK_SERVER set
    monkeypatch.setenv(speak.SERVER_ENV_VAR, "http://env:8000")

    # WHEN resolved
    # THEN the env var's value wins: a blank flag is "as if not given"
    assert speak._resolve_server(explicit) == "http://env:8000"


@pytest.mark.parametrize("explicit", ["", "   "])
def test_resolve_server_withBlankFlagAndBlankEnvVar_returnsNone(monkeypatch, explicit):
    # GIVEN a blank --server and a blank TTS_SPEAK_SERVER
    monkeypatch.setenv(speak.SERVER_ENV_VAR, "  ")

    # WHEN resolved
    # THEN None, so main() can raise the exit-2 usage error — and, crucially,
    #      nothing schemeless can reach urlopen (regression: uncaught
    #      ValueError 'unknown url type: /capabilities')
    assert speak._resolve_server(explicit) is None


def test_resolve_server_withNonBlankFlag_ignoresEnvVar(monkeypatch):
    # GIVEN both the flag and the env var with different values
    monkeypatch.setenv(speak.SERVER_ENV_VAR, "http://env:8000")

    # WHEN resolved
    # THEN the command line wins (spec: "the command line arg wins")
    assert speak._resolve_server("http://flag:8000") == "http://flag:8000"


def test_resolve_server_withPaddedFlag_stripsWhitespace():
    # GIVEN a --server value with accidental surrounding whitespace
    # WHEN resolved
    # THEN it comes back stripped, exactly like the env var's value
    assert speak._resolve_server("  http://flag:8000  ") == "http://flag:8000"


# ---------------------------------------------------------------------------
# Stage-1 parser: --timeout, --server env-var fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "-5", "abc"])
def test_build_base_parser_withBadTimeout_exits2(value):
    # GIVEN a zero, negative, or non-numeric timeout
    # WHEN parsed
    # THEN argparse rejects it (exit 2) instead of letting urlopen do
    #      something degenerate with it (0 = non-blocking, < 0 = block
    #      indefinitely)
    parser = speak.build_base_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            ["Hi", "--server", "http://x", "--ref-audio", "a.wav", "--timeout", value]
        )
    assert excinfo.value.code == 2


def test_build_base_parser_withValidTimeout_parsesIt():
    parser = speak.build_base_parser()
    args = parser.parse_args(
        ["Hi", "--server", "http://x", "--ref-audio", "a.wav", "--timeout", "30"]
    )
    assert args.timeout == 30


def test_build_base_parser_withoutServerFlag_defaultsToEnvVar(monkeypatch):
    # GIVEN TTS_SPEAK_SERVER set and no --server on the command line
    monkeypatch.setenv(speak.SERVER_ENV_VAR, "http://env:8000")
    parser = speak.build_base_parser()

    # WHEN stage-1 parsing runs
    # THEN the env var value becomes the argument's value
    args, _ = parser.parse_known_args(["Hello"])
    assert args.server == "http://env:8000"


def test_build_base_parser_withServerFlagAndEnvVar_flagWins(monkeypatch):
    # GIVEN both the env var and an explicit --server
    monkeypatch.setenv(speak.SERVER_ENV_VAR, "http://env:8000")
    parser = speak.build_base_parser()

    # WHEN the command line supplies --server
    # THEN the command line wins (spec: "the command line arg wins")
    args, _ = parser.parse_known_args(["Hello", "--server", "http://flag:8000"])
    assert args.server == "http://flag:8000"


@pytest.mark.parametrize(
    ("env_value", "expected_annotation"),
    [("http://10.0.0.5:8000", "http://10.0.0.5:8000"), ("   ", "not set")],
)
def test_build_base_parser_withServerEnvVar_helpShowsCurrentValue(
    monkeypatch, capsys, env_value, expected_annotation
):
    # GIVEN the env var set (or blank, which the spec treats as unset)
    monkeypatch.setenv(speak.SERVER_ENV_VAR, env_value)
    parser = speak.build_base_parser()

    # WHEN --help is rendered
    # THEN the custom help names the variable and shows its current value
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "TTS_SPEAK_SERVER" in out
    assert expected_annotation in out


# ---------------------------------------------------------------------------
# Capabilities handling
# ---------------------------------------------------------------------------


def test_check_schema_version_withSupportedVersion_doesNotRaise():
    speak.check_schema_version({"schema_version": speak.SUPPORTED_SCHEMA_VERSION})


def test_check_schema_version_withFutureVersion_raisesWithExactMessage():
    # GIVEN a capabilities document with a newer schema version
    # WHEN checked
    # THEN the spec's exact error message is produced
    with pytest.raises(speak.SpeakError, match=r"unknown schema version 3, expected 2; aborting\."):
        speak.check_schema_version({"schema_version": 3})


def test_check_schema_version_withListDocument_raisesWithCleanMessage():
    # GIVEN a server that answered with valid JSON that is not an object
    #      (e.g. a misrouted URL hitting a chatty gateway)
    # WHEN checked
    # THEN a clean SpeakError is raised, not an AttributeError from .get()
    with pytest.raises(speak.SpeakError, match=r"non-object capabilities document \(list\)"):
        speak.check_schema_version(["not", "an", "object"])


def test_parameter_specs_withMalformedEntries_skipsThemWithWarnings(capsys):
    # GIVEN a parameters list containing non-dict entries
    # WHEN narrowed
    # THEN only well-formed entries survive, each skip is warned about
    specs = speak._parameter_specs({"parameters": [{"name": "ok", "type": "string"}, "junk", 42]})
    assert [s.get("name") for s in specs] == ["ok"]
    out = capsys.readouterr().out
    assert "Skipping malformed capabilities parameter entry: 'junk'" in out
    assert "Skipping malformed capabilities parameter entry: 42" in out


def test_parameter_specs_withNonListParameters_warnsAndReturnsEmpty(capsys):
    # GIVEN a capabilities document whose 'parameters' is a dict, not a list
    # WHEN narrowed
    # THEN nothing is registered and the user is warned
    assert speak._parameter_specs({"parameters": {"name": "oops"}}) == []
    assert "is not a list" in capsys.readouterr().out


def test_validate_response_withMissingAudioBase64_raises():
    with pytest.raises(speak.SpeakError, match=r"Invalid response from server\."):
        speak.validate_response({"seed": 1})


def test_validate_response_withEmptyAudioBase64_raises():
    with pytest.raises(speak.SpeakError, match=r"Invalid response from server\."):
        speak.validate_response({"audio_base64": ""})


def test_validate_response_withNonObjectBody_raises():
    with pytest.raises(speak.SpeakError, match=r"Invalid response from server\."):
        speak.validate_response(["not", "an", "object"])


def test_validate_response_withValidBody_returnsBody():
    body = {"audio_base64": "AAAA", "seed": 1, "time_used": 1.0, "rtf": None}
    assert speak.validate_response(body) is body


# ---------------------------------------------------------------------------
# Capabilities -> argparse mapping (stage 2)
# ---------------------------------------------------------------------------


def test_add_dynamic_arguments_withDotsCapabilities_addsExactlyTheNonFileDrivenParams():
    # GIVEN the real dots.tts capabilities document
    # WHEN dynamic arguments are registered
    # THEN text/audio_base64/reference_text are skipped and the rest, in
    #      declaration order, become CLI flags
    _, names = _dots_parser()
    assert names == ["language", "seed", "num_steps", "guidance_scale", "speaker_scale", "ode_method"]


def test_add_dynamic_arguments_withReservedName_warnsAndSkips(capsys):
    # GIVEN a server advertising a parameter named like a reserved flag
    # WHEN dynamic arguments are registered
    # THEN it is skipped with the spec's exact warning
    parser = speak.build_base_parser()
    caps = {"parameters": [{"name": "output_file", "type": "string", "required": False}]}
    names = speak.add_dynamic_arguments(parser, caps)
    assert names == []
    assert "Warning: Server returned a reserved parameter name output_file; ignoring it." in capsys.readouterr().out


def test_add_dynamic_arguments_withHelpParam_warnsAndSkipsInsteadOfCrashing(capsys):
    # GIVEN a server advertising a parameter named 'help' — a legal Pydantic
    #      field name that would collide with argparse's built-in
    # WHEN dynamic arguments are registered
    # THEN it is skipped with the reserved-name warning, not an
    #      unhandled ArgumentError ("conflicting option string")
    parser = speak.build_base_parser()
    caps = {"parameters": [{"name": "help", "type": "string", "required": False, "enum": None}]}
    names = speak.add_dynamic_arguments(parser, caps)
    assert names == []
    assert "Warning: Server returned a reserved parameter name help; ignoring it." in capsys.readouterr().out


def test_add_dynamic_arguments_withEnum_acceptsValidChoice():
    # GIVEN the dots.tts ode_method enum (euler/midpoint/rk4)
    # WHEN a valid choice is supplied
    # THEN it is accepted
    parser, _ = _dots_parser()
    args = _stage_two_args(parser, ["--ode-method", "rk4"])
    assert args.ode_method == "rk4"


def test_add_dynamic_arguments_withEnum_rejectsInvalidChoiceWithExit2():
    parser, _ = _dots_parser()
    with pytest.raises(SystemExit) as excinfo:
        _stage_two_args(parser, ["--ode-method", "bogus"])
    assert excinfo.value.code == 2


def test_add_dynamic_arguments_withRequiredParam_omittedExits2():
    # GIVEN a required engine parameter
    # WHEN the user omits it
    # THEN argparse enforces it with exit 2
    parser = speak.build_base_parser()
    caps = {"parameters": [{"name": "mandatory", "type": "integer", "required": True}]}
    speak.add_dynamic_arguments(parser, caps)
    with pytest.raises(SystemExit) as excinfo:
        _stage_two_args(parser)
    assert excinfo.value.code == 2


def test_add_dynamic_arguments_withUnrecognizedArg_exits2():
    parser, _ = _dots_parser()
    with pytest.raises(SystemExit) as excinfo:
        _stage_two_args(parser, ["--does-not-exist", "1"])
    assert excinfo.value.code == 2


def test_add_dynamic_arguments_withInvalidIntValue_exits2():
    # GIVEN the integer parameter num_steps
    # WHEN given a non-numeric string
    # THEN argparse's type check fails with exit 2
    parser, _ = _dots_parser()
    with pytest.raises(SystemExit) as excinfo:
        _stage_two_args(parser, ["--num-steps", "test"])
    assert excinfo.value.code == 2


def test_add_dynamic_arguments_withBooleanParam_parsesWordValues():
    # GIVEN a boolean engine parameter
    # WHEN given a word-list value
    # THEN it parses to a real bool
    parser = speak.build_base_parser()
    caps = {"parameters": [{"name": "loud", "type": "boolean", "required": False}]}
    speak.add_dynamic_arguments(parser, caps)
    args = _stage_two_args(parser, ["--loud", "yes"])
    assert args.loud is True


def test_add_dynamic_arguments_withBooleanParam_rejectsArbitraryString():
    # GIVEN a boolean engine parameter
    # WHEN given a string outside the word lists
    # THEN it is rejected with exit 2 (argparse's bool would have said True!)
    parser = speak.build_base_parser()
    caps = {"parameters": [{"name": "loud", "type": "boolean", "required": False}]}
    speak.add_dynamic_arguments(parser, caps)
    with pytest.raises(SystemExit) as excinfo:
        _stage_two_args(parser, ["--loud", "maybe"])
    assert excinfo.value.code == 2


def test_add_dynamic_arguments_withNumberParam_acceptsFloatValue():
    parser, _ = _dots_parser()
    args = _stage_two_args(parser, ["--guidance-scale", "2.5"])
    assert args.guidance_scale == 2.5


def test_add_dynamic_arguments_withIntegerEnum_coercesChoicesAndAcceptsValue():
    # GIVEN an integer parameter with a stringified enum (capabilities always
    #      carry enum values as strings)
    # WHEN a value within the enum is supplied
    # THEN argparse's type conversion yields a value that actually matches
    #      the coerced choices (raw string choices could never match an int)
    parser = speak.build_base_parser()
    caps = {"parameters": [{"name": "mode", "type": "integer", "required": False, "enum": ["1", "2"]}]}
    speak.add_dynamic_arguments(parser, caps)
    args = _stage_two_args(parser, ["--mode", "1"])
    assert args.mode == 1


def test_add_dynamic_arguments_withIntegerEnum_rejectsValueOutsideEnum():
    parser = speak.build_base_parser()
    caps = {"parameters": [{"name": "mode", "type": "integer", "required": False, "enum": ["1", "2"]}]}
    speak.add_dynamic_arguments(parser, caps)
    with pytest.raises(SystemExit) as excinfo:
        _stage_two_args(parser, ["--mode", "3"])
    assert excinfo.value.code == 2


def test_add_dynamic_arguments_withInconsistentEnum_raisesSpeakError():
    # GIVEN an integer parameter advertising a non-numeric enum value:
    #      the document contradicts itself
    # WHEN the enum is coerced
    # THEN a loud SpeakError is raised rather than a silent misparse
    parser = speak.build_base_parser()
    caps = {"parameters": [{"name": "mode", "type": "integer", "required": False, "enum": ["1", "banana"]}]}
    with pytest.raises(speak.SpeakError, match=r"enum value 'banana'"):
        speak.add_dynamic_arguments(parser, caps)


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------


def test_build_payload_withOnlyCoreArgs_containsExactlyCoreFields():
    args = argparse.Namespace(text="hi")
    payload = speak.build_payload(args, [], "AQID", None)
    assert payload == {"text": "hi", "audio_base64": "AQID"}


def test_build_payload_withSuppliedDynamicArg_includesIt():
    args = argparse.Namespace(text="hi", seed=42)
    payload = speak.build_payload(args, ["seed"], "AQID", None)
    assert payload["seed"] == 42


def test_build_payload_withUnsuppliedDynamicArg_omitsIt():
    # GIVEN an optional dynamic arg the user never supplied (no namespace
    # attribute, thanks to default=SUPPRESS)
    # WHEN the payload is assembled
    # THEN it is omitted rather than filled with its default
    args = argparse.Namespace(text="hi")
    payload = speak.build_payload(args, ["seed"], "AQID", None)
    assert "seed" not in payload


def test_build_payload_withTranscript_includesReferenceText():
    args = argparse.Namespace(text="hi")
    payload = speak.build_payload(args, [], "AQID", "hello there")
    assert payload["reference_text"] == "hello there"


def test_build_payload_withoutTranscript_omitsReferenceText():
    args = argparse.Namespace(text="hi")
    payload = speak.build_payload(args, [], "AQID", None)
    assert "reference_text" not in payload


# ---------------------------------------------------------------------------
# Reference source resolution (flags + persona dir)
# ---------------------------------------------------------------------------


def test_resolve_persona_dir_withExplicitFlag_ignoresRefAudioAndEnvVar():
    # GIVEN --persona-dir explicitly given, plus --ref-audio and the env var
    # WHEN resolved
    # THEN the flag wins (spec precedence rule 1)
    assert speak.resolve_persona_dir("/flag", "/ref.wav", "/env") == "/flag"


def test_resolve_persona_dir_withRefAudioOnly_ignoresEnvVar():
    # GIVEN no explicit --persona-dir, but --ref-audio and the env var
    # WHEN resolved
    # THEN no persona dir is used and the env var is silently ignored
    #      (spec rule 2 — the case a naive argparse default gets wrong)
    assert speak.resolve_persona_dir(None, "/ref.wav", "/env") is None


def test_resolve_persona_dir_withOnlyEnvVar_returnsIt():
    # GIVEN neither reference flag, but the env var is set
    # WHEN resolved
    # THEN it is used "as though it had been given to --persona-dir" (rule 3)
    assert speak.resolve_persona_dir(None, None, "/env") == "/env"


def test_resolve_persona_dir_withNoSourceAtAll_returnsNone():
    # GIVEN nothing from flag or env var
    # WHEN resolved
    # THEN None, so the caller can raise the spec's exit-2 usage error
    assert speak.resolve_persona_dir(None, None, None) is None


def _make_persona(directory: Path, audio: bytes = b"WAV", transcript: str = "ref words") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ref.wav").write_bytes(audio)
    (directory / "ref.txt").write_text(transcript, encoding="utf-8")
    return directory


def test_resolve_reference_sources_withPersonaDirAndBothFlags_personaWinsAndWarns(tmp_path, capsys):
    # GIVEN both the individual flags and a persona dir
    # WHEN resolved
    # THEN the persona dir wins and both flags are ignored with warnings
    persona = _make_persona(tmp_path / "alice")
    audio, transcript = speak.resolve_reference_sources("a.wav", "t.txt", str(persona))
    assert audio == str(persona / "ref.wav")
    assert transcript == str(persona / "ref.txt")
    out = capsys.readouterr().out
    assert "Warning: --ref-audio ignored because --persona-dir was given." in out
    assert "Warning: --ref-audio-transcript ignored because --persona-dir was given." in out


def test_resolve_reference_sources_withMissingPersonaDir_raises(tmp_path):
    with pytest.raises(speak.SpeakError, match=r"Persona directory does not exist"):
        speak.resolve_reference_sources(None, None, str(tmp_path / "nope"))


def test_resolve_reference_sources_withPersonaDirMissingRefWav_raises(tmp_path):
    (tmp_path / "ref.txt").write_text("x", encoding="utf-8")
    with pytest.raises(speak.SpeakError, match=r"ref\.wav"):
        speak.resolve_reference_sources(None, None, str(tmp_path))


def test_resolve_reference_sources_withPersonaDirMissingRefTxt_raises(tmp_path):
    (tmp_path / "ref.wav").write_bytes(b"x")
    with pytest.raises(speak.SpeakError, match=r"ref\.txt"):
        speak.resolve_reference_sources(None, None, str(tmp_path))


def test_resolve_reference_sources_withRefAudioOnly_resolvesAudioAndNoTranscript(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"WAV")
    audio, transcript = speak.resolve_reference_sources(str(tmp_path / "a.wav"), None, None)
    assert audio == str(tmp_path / "a.wav")
    assert transcript is None


def test_resolve_reference_sources_withMissingRefAudioFile_raises(tmp_path):
    with pytest.raises(speak.SpeakError, match=r"not found or not readable"):
        speak.resolve_reference_sources(str(tmp_path / "missing.wav"), None, None)


def test_resolve_reference_sources_withMissingRefTranscriptFile_raises(tmp_path):
    with pytest.raises(speak.SpeakError, match=r"not found or not readable"):
        speak.resolve_reference_sources(None, str(tmp_path / "missing.txt"), None)


# ---------------------------------------------------------------------------
# End-to-end main() with network + aplay stubbed
# ---------------------------------------------------------------------------


def _stub_network(monkeypatch, caps, response, seen=None, urls=None):
    if seen is None:
        seen = {}
    if urls is None:
        urls = []
    monkeypatch.setattr(
        speak, "fetch_json", lambda url, timeout: (urls.append(url), caps)[1]
    )
    monkeypatch.setattr(
        speak, "post_json",
        lambda url, payload, timeout: (urls.append(url), seen.update(payload), response)[2],
    )


def _ok_response() -> dict:
    return {
        "audio_base64": base64.b64encode(b"WAVDATA").decode("ascii"),
        "sample_rate": 24000,
        "seed": 7,
        "time_used": 1.5,
        "rtf": 0.3,
    }


def _aplay_spy(monkeypatch) -> list[bytes]:
    """Stub run_aplay, recording every call — used to prove the silent path."""
    calls: list[bytes] = []

    def spy(audio: bytes) -> bool:
        calls.append(audio)
        return True

    monkeypatch.setattr(speak, "run_aplay", spy)
    return calls


def test_main_happyPath_savesFileLogsStatsAndReturnsZero(tmp_path, monkeypatch, capsys):
    # GIVEN a chatterbox-style server (no reference_text param) and a real ref file
    caps = _caps("chatterbox_capabilities.json")
    seen, urls = {}, []
    _stub_network(monkeypatch, caps, _ok_response(), seen, urls)
    aplay_calls = _aplay_spy(monkeypatch)
    (tmp_path / "ref.wav").write_bytes(b"REF")
    out_file = tmp_path / "out.wav"

    # WHEN the user synthesizes with a trailing-slash server URL and --seed
    code = speak.main(
        [
            "Hello",
            "--server", "http://10.0.0.5:8000/",
            "--ref-audio", str(tmp_path / "ref.wav"),
            "--seed", "7",
            "--output-file", str(out_file),
        ]
    )

    # THEN exit 0, the payload carries the core fields, the URLs use the
    #      advertised endpoint with normalized slashes, and stats are logged
    assert code == 0
    assert out_file.read_bytes() == b"WAVDATA"
    assert aplay_calls == []  # --output-file is implied "silent": never plays
    assert seen == {"text": "Hello", "audio_base64": base64.b64encode(b"REF").decode(), "seed": 7}
    assert urls == [
        "http://10.0.0.5:8000/capabilities",
        "http://10.0.0.5:8000/synthesize",
    ]
    out = capsys.readouterr().out
    assert "seed=7" in out
    assert "time_used=1.5s" in out
    assert "rtf=0.3" in out
    assert "Saved audio to" in out


def test_main_withPersonaDir_usesRefWavAndRefTxt(tmp_path, monkeypatch):
    caps = _caps("dots_capabilities.json")  # advertises reference_text
    seen = {}
    _stub_network(monkeypatch, caps, _ok_response(), seen)
    monkeypatch.setattr(speak, "run_aplay", lambda audio: True)
    persona = _make_persona(tmp_path / "alice", audio=b"PERSONA", transcript="alice words")

    code = speak.main(["Hello", "--server", "http://x", "--persona-dir", str(persona)])

    assert code == 0
    assert base64.b64decode(seen["audio_base64"]) == b"PERSONA"
    assert seen["reference_text"] == "alice words"


def test_main_withTranscriptOnServerWithoutReferenceText_warnsAndOmitsField(tmp_path, monkeypatch, capsys):
    # GIVEN a server that does not advertise reference_text (chatterbox)
    caps = _caps("chatterbox_capabilities.json")
    seen = {}
    _stub_network(monkeypatch, caps, _ok_response(), seen)
    monkeypatch.setattr(speak, "run_aplay", lambda audio: True)
    (tmp_path / "a.wav").write_bytes(b"REF")
    (tmp_path / "t.txt").write_text("  words  here\n", encoding="utf-8")

    # WHEN the user still supplies a transcript
    code = speak.main(
        ["Hello", "--server", "http://x", "--ref-audio", str(tmp_path / "a.wav"),
         "--ref-audio-transcript", str(tmp_path / "t.txt")]
    )

    # THEN it is ignored with the spec's warning, and the field is not sent
    assert code == 0
    assert "reference_text" not in seen
    assert "Warning: Reference text ignored as this server does not accept it." in capsys.readouterr().out


def test_main_withTranscriptOnServerWithReferenceText_sendsCollapsedText(tmp_path, monkeypatch):
    caps = _caps("dots_capabilities.json")  # advertises reference_text
    seen = {}
    _stub_network(monkeypatch, caps, _ok_response(), seen)
    monkeypatch.setattr(speak, "run_aplay", lambda audio: True)
    (tmp_path / "a.wav").write_bytes(b"REF")
    (tmp_path / "t.txt").write_text("  words  here\n", encoding="utf-8")

    code = speak.main(
        ["Hello", "--server", "http://x", "--ref-audio", str(tmp_path / "a.wav"),
         "--ref-audio-transcript", str(tmp_path / "t.txt")]
    )

    assert code == 0
    assert seen["reference_text"] == "words here"


def test_main_withNoReferenceAudioAtAll_exits2BeforeAnyNetworkCall(monkeypatch, capsys):
    # GIVEN no --ref-audio, no --persona-dir, and no TTS_SPEAK_PERSONA_DIR
    #      (the conftest fixture strips any ambient value)
    called = []
    monkeypatch.setattr(speak, "fetch_json", lambda url, timeout: called.append(url))

    # WHEN main runs
    # THEN it fails locally (exit 2 per spec, argparse framing) without
    #      touching the network
    with pytest.raises(SystemExit) as excinfo:
        speak.main(["Hello", "--server", "http://x"])
    assert excinfo.value.code == 2
    assert called == []
    assert "You must supply reference audio or a persona directory!" in capsys.readouterr().err


def test_main_withoutText_exits1BeforeAnyNetworkCall(tmp_path, monkeypatch, capsys):
    # GIVEN no text positional (the reference file is fine)
    called = []
    monkeypatch.setattr(speak, "fetch_json", lambda url, timeout: called.append(url))
    (tmp_path / "a.wav").write_bytes(b"REF")

    # WHEN main runs without a text argument
    code = speak.main(["--server", "http://x", "--ref-audio", str(tmp_path / "a.wav")])

    # THEN it fails locally (exit 1) without touching the network
    assert code == 1
    assert called == []
    assert "You must supply the text to synthesize!" in capsys.readouterr().err


def test_main_withUnknownSchemaVersion_exits1WithMessage(tmp_path, monkeypatch, capsys):
    caps = _caps("dots_capabilities.json")
    caps["schema_version"] = 99
    monkeypatch.setattr(speak, "fetch_json", lambda url, timeout: caps)

    code = speak.main(["Hello", "--server", "http://x", "--ref-audio", str(tmp_path / "a.wav")])

    assert code == 1
    assert "Server returned unknown schema version 99, expected 2; aborting." in capsys.readouterr().err


def test_main_withUnreachableServer_exits1(tmp_path, monkeypatch, capsys):
    def boom(url, timeout):
        raise speak.SpeakError(f"Failed to connect to server: {url}")

    monkeypatch.setattr(speak, "fetch_json", boom)
    (tmp_path / "a.wav").write_bytes(b"REF")

    code = speak.main(["Hello", "--server", "http://x", "--ref-audio", str(tmp_path / "a.wav")])

    assert code == 1
    assert "Failed to connect to server" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# End-to-end main(): --server / TTS_SPEAK_SERVER
# ---------------------------------------------------------------------------


def test_main_withNoServerAnywhere_exits2WithArgparseMessage(monkeypatch, capsys):
    # GIVEN no --server and no TTS_SPEAK_SERVER (fixture strips ambient values)
    # WHEN main runs
    # THEN argparse reports the missing requirement (exit 2, spec's message)
    with pytest.raises(SystemExit) as excinfo:
        speak.main(["Hello", "--ref-audio", "a.wav"])
    assert excinfo.value.code == 2
    assert "the following arguments are required: --server" in capsys.readouterr().err


def test_main_withNoServerAndNoRefAudio_reportsMissingServerFirst(monkeypatch, capsys):
    # GIVEN text, but neither a server nor any reference source
    # WHEN main runs
    # THEN the server error precedes the reference-audio error (spec:
    #      'Error precedence')
    with pytest.raises(SystemExit) as excinfo:
        speak.main(["Hello"])
    assert excinfo.value.code == 2
    assert "the following arguments are required: --server" in capsys.readouterr().err


def test_main_withNoServerAndNoText_reportsMissingServerFirst(monkeypatch, capsys):
    # GIVEN reference audio, but neither a server nor text
    # WHEN main runs
    # THEN the server error precedes the missing-text error
    with pytest.raises(SystemExit) as excinfo:
        speak.main(["--ref-audio", "a.wav"])
    assert excinfo.value.code == 2
    assert "the following arguments are required: --server" in capsys.readouterr().err


def test_main_withBlankServerEnvVar_treatsItAsUnset(monkeypatch, capsys):
    # GIVEN TTS_SPEAK_SERVER set to whitespace only
    monkeypatch.setenv(speak.SERVER_ENV_VAR, "   ")

    # WHEN main runs with no --server
    # THEN the blank value is treated as unset: the spec's exit-2 usage error
    with pytest.raises(SystemExit) as excinfo:
        speak.main(["Hello", "--ref-audio", "a.wav"])
    assert excinfo.value.code == 2
    assert "the following arguments are required: --server" in capsys.readouterr().err


@pytest.mark.parametrize("blank", ["", "   "])
def test_main_withBlankServerFlag_treatsItAsUnset(monkeypatch, capsys, blank):
    # GIVEN --server supplied with an empty/whitespace-only value and no
    #      TTS_SPEAK_SERVER (the conftest fixture strips ambient values)
    # WHEN main runs
    # THEN the blank value is treated as unset: the spec's exit-2 usage error,
    #      not an uncaught ValueError from urlopen (regression: 'unknown url
    #      type: /capabilities')
    with pytest.raises(SystemExit) as excinfo:
        speak.main(["Hello", "--server", blank, "--ref-audio", "a.wav"])
    assert excinfo.value.code == 2
    assert "the following arguments are required: --server" in capsys.readouterr().err


@pytest.mark.parametrize(
    "garbage",
    [
        "192.168.1.50",  # bare IP: the classic 'forgot http://' typo
        "localhost:9",   # a colon, but no scheme
        "relative/path",
        "/capabilities",
    ],
)
def test_main_withSchemelessServerValue_exits2WithSchemeHint(monkeypatch, capsys, garbage):
    # GIVEN a --server value with no scheme: without the check, the no-colon
    #      values die deep inside urllib with an uncaught ValueError and the
    #      colon value with the baffling 'Failed to connect to server:
    #      unknown url type: localhost'
    # WHEN main runs
    # THEN an exit-2 usage error naming the expected shape, before any
    #      network traffic
    with pytest.raises(SystemExit) as excinfo:
        speak.main(["Hello", "--server", garbage, "--ref-audio", "a.wav"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "server URL must include a scheme" in err
    assert "http://host:port" in err


def test_main_withBlankServerFlagAndEnvVar_fallsBackToEnvVar(tmp_path, monkeypatch):
    # GIVEN --server blank (treated as unset, i.e. as if the flag were not
    #      given at all) and TTS_SPEAK_SERVER set
    caps = _caps("chatterbox_capabilities.json")
    urls: list[str] = []
    _stub_network(monkeypatch, caps, _ok_response(), {}, urls)
    monkeypatch.setenv(speak.SERVER_ENV_VAR, "http://env.example:8000")
    (tmp_path / "a.wav").write_bytes(b"REF")

    # WHEN synthesizing with the blank flag
    code = speak.main(
        ["Hello", "--server", "   ", "--ref-audio", str(tmp_path / "a.wav"),
         "--output-file", str(tmp_path / "o.wav")]
    )

    # THEN the env var's server is used, exactly as if the flag were absent
    assert code == 0
    assert urls == [
        "http://env.example:8000/capabilities",
        "http://env.example:8000/synthesize",
    ]


def test_main_withBlankServerFlagAndListServerParams_queriesEnvVar(monkeypatch):
    # GIVEN a blank --server and TTS_SPEAK_SERVER set
    caps = _caps("dots_capabilities.json")
    urls: list[str] = []
    monkeypatch.setenv(speak.SERVER_ENV_VAR, "http://env.example:8000")
    monkeypatch.setattr(speak, "fetch_json", lambda url, timeout: (urls.append(url), caps)[1])

    # WHEN the user asks for the parameter listing
    # THEN the listing works and queries the env var's URL (spec: listing must
    #      work with either source)
    code = speak.main(["--server", "", "--list-server-params"])

    assert code == 0
    assert urls == ["http://env.example:8000/capabilities"]


def test_main_withPaddedServerFlag_stripsWhitespace(tmp_path, monkeypatch):
    # GIVEN a --server value with accidental surrounding whitespace (the same
    #      case _env_or_none absorbs for the env var)
    caps = _caps("chatterbox_capabilities.json")
    urls: list[str] = []
    _stub_network(monkeypatch, caps, _ok_response(), {}, urls)
    (tmp_path / "a.wav").write_bytes(b"REF")

    # WHEN synthesizing
    code = speak.main(
        ["Hello", "--server", "  http://flag.example:8000  ", "--ref-audio", str(tmp_path / "a.wav"),
         "--output-file", str(tmp_path / "o.wav")]
    )

    # THEN both requests go to the stripped URL — the second one in particular
    #      proves stage-2 re-parsing did not resurrect the padded value for
    #      join_url (it would have died in urlopen with an uncaught ValueError)
    assert code == 0
    assert urls == [
        "http://flag.example:8000/capabilities",
        "http://flag.example:8000/synthesize",
    ]


def test_main_withOnlyServerEnvVar_andListServerParams_queriesEnvValue(monkeypatch):
    # GIVEN the server only via the env var
    caps = _caps("dots_capabilities.json")
    urls: list[str] = []
    monkeypatch.setenv(speak.SERVER_ENV_VAR, "http://env.example:8000")
    monkeypatch.setattr(speak, "fetch_json", lambda url, timeout: (urls.append(url), caps)[1])

    # WHEN the user asks for the parameter listing (no --server on the CLI)
    # THEN the listing works and queries the env var's URL (spec: listing must
    #      work with either source)
    code = speak.main(["--list-server-params"])

    assert code == 0
    assert urls == ["http://env.example:8000/capabilities"]


def test_main_withOnlyServerEnvVar_sendsRequestToEnvValue(tmp_path, monkeypatch):
    caps = _caps("chatterbox_capabilities.json")
    seen, urls = {}, []
    _stub_network(monkeypatch, caps, _ok_response(), seen, urls)
    monkeypatch.setenv(speak.SERVER_ENV_VAR, "http://env.example:8000")
    (tmp_path / "a.wav").write_bytes(b"REF")

    # WHEN synthesizing with --server absent from the command line
    code = speak.main(
        ["Hello", "--ref-audio", str(tmp_path / "a.wav"), "--output-file", str(tmp_path / "o.wav")]
    )

    # THEN both requests go to the env var's server
    assert code == 0
    assert urls == [
        "http://env.example:8000/capabilities",
        "http://env.example:8000/synthesize",
    ]


def test_main_withServerEnvVarAndFlag_prefersFlag(tmp_path, monkeypatch):
    caps = _caps("chatterbox_capabilities.json")
    urls: list[str] = []
    _stub_network(monkeypatch, caps, _ok_response(), {}, urls)
    monkeypatch.setenv(speak.SERVER_ENV_VAR, "http://env.example:8000")
    (tmp_path / "a.wav").write_bytes(b"REF")

    # WHEN both the env var and the command line supply a server
    code = speak.main(
        [
            "Hello",
            "--server", "http://flag.example:8000",
            "--ref-audio", str(tmp_path / "a.wav"),
            "--output-file", str(tmp_path / "o.wav"),
        ]
    )

    # THEN the command line wins and the env var's server is never contacted
    assert code == 0
    assert urls == [
        "http://flag.example:8000/capabilities",
        "http://flag.example:8000/synthesize",
    ]


def test_main_withOnlyServerFlag_usesFlagValue(tmp_path, monkeypatch):
    # GIVEN no env var (the conftest fixture strips it): the flag is the sole
    #      source
    caps = _caps("chatterbox_capabilities.json")
    urls: list[str] = []
    _stub_network(monkeypatch, caps, _ok_response(), {}, urls)
    (tmp_path / "a.wav").write_bytes(b"REF")

    code = speak.main(
        [
            "Hello",
            "--server", "http://flag.example:8000",
            "--ref-audio", str(tmp_path / "a.wav"),
            "--output-file", str(tmp_path / "o.wav"),
        ]
    )

    assert code == 0
    assert urls == [
        "http://flag.example:8000/capabilities",
        "http://flag.example:8000/synthesize",
    ]


def test_main_withUnrecognizedEngineArg_exits2(tmp_path, monkeypatch):
    caps = _caps("dots_capabilities.json")
    monkeypatch.setattr(speak, "fetch_json", lambda url, timeout: caps)
    (tmp_path / "a.wav").write_bytes(b"REF")

    with pytest.raises(SystemExit) as excinfo:
        speak.main(["Hello", "--server", "http://x", "--ref-audio", str(tmp_path / "a.wav"),
                    "--does-not-exist", "1"])
    assert excinfo.value.code == 2


def test_main_withMissingRequiredEngineArg_exits2(tmp_path, monkeypatch):
    caps = {
        "schema_version": 2,  # required: the script rejects unknown versions pre-parse
        "parameters": [
            {"name": "text", "type": "string", "required": True},
            {"name": "audio_base64", "type": "string", "required": True},
            {"name": "mandatory", "type": "integer", "required": True},
        ],
    }
    monkeypatch.setattr(speak, "fetch_json", lambda url, timeout: caps)
    (tmp_path / "a.wav").write_bytes(b"REF")

    with pytest.raises(SystemExit) as excinfo:
        speak.main(["Hello", "--server", "http://x", "--ref-audio", str(tmp_path / "a.wav")])
    assert excinfo.value.code == 2


def test_main_withEmptyAudioResponse_exits1WithInvalidResponseMessage(tmp_path, monkeypatch, capsys):
    caps = _caps("dots_capabilities.json")
    _stub_network(monkeypatch, caps, {"seed": 1})
    monkeypatch.setattr(speak, "run_aplay", lambda audio: True)
    (tmp_path / "a.wav").write_bytes(b"REF")

    code = speak.main(["Hello", "--server", "http://x", "--ref-audio", str(tmp_path / "a.wav")])

    assert code == 1
    assert "Invalid response from server." in capsys.readouterr().err


def test_main_whenAplayMissing_exits1AfterLoggingStats(tmp_path, monkeypatch, capsys):
    caps = _caps("dots_capabilities.json")
    _stub_network(monkeypatch, caps, _ok_response())

    def no_aplay(audio):
        raise speak.SpeakError("'aplay' not found. Please install the ALSA utilities.")

    monkeypatch.setattr(speak, "run_aplay", no_aplay)
    (tmp_path / "a.wav").write_bytes(b"REF")

    # WHEN aplay is missing, synthesis stats were already logged (spec: log
    #      before play/save), then exit 1
    code = speak.main(["Hello", "--server", "http://x", "--ref-audio", str(tmp_path / "a.wav")])

    captured = capsys.readouterr()
    assert code == 1
    assert "seed=7" in captured.out  # stats logged before the playback attempt failed
    assert "'aplay' not found" in captured.err


def test_main_whenAplayFailsWithoutOutputFile_warnsAndExits0(tmp_path, monkeypatch, capsys):
    # GIVEN the play path (no --output-file) and aplay that exits non-zero
    caps = _caps("dots_capabilities.json")
    _stub_network(monkeypatch, caps, _ok_response())
    monkeypatch.setattr(speak, "run_aplay", lambda audio: False)
    (tmp_path / "a.wav").write_bytes(b"REF")

    code = speak.main(["Hello", "--server", "http://x", "--ref-audio", str(tmp_path / "a.wav")])

    # THEN aplay's failure is a warning, not an error: audio discarded, exit 0
    assert code == 0
    assert "Warning: aplay returned a non-zero exit code" in capsys.readouterr().out


def test_main_withExistingOutputFileAndDeclinedOverwrite_discardsAudioSilently(tmp_path, monkeypatch):
    # GIVEN an existing output file and a "n" answer at the overwrite prompt
    caps = _caps("dots_capabilities.json")
    _stub_network(monkeypatch, caps, _ok_response())
    aplay_calls = _aplay_spy(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    (tmp_path / "a.wav").write_bytes(b"REF")
    out_file = tmp_path / "o.wav"
    out_file.write_bytes(b"OLD")

    code = speak.main(
        ["Hello", "--server", "http://x", "--ref-audio", str(tmp_path / "a.wav"),
         "--output-file", str(out_file)]
    )

    # THEN declining is not an error, the old file is untouched, and nothing
    #      is played: the silent path never falls through to aplay
    assert code == 0
    assert out_file.read_bytes() == b"OLD"
    assert aplay_calls == []


def test_main_withExistingOutputFileAndAcceptedOverwrite_replacesFileWithoutPlaying(tmp_path, monkeypatch):
    caps = _caps("dots_capabilities.json")
    _stub_network(monkeypatch, caps, _ok_response())
    aplay_calls = _aplay_spy(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    (tmp_path / "a.wav").write_bytes(b"REF")
    out_file = tmp_path / "o.wav"
    out_file.write_bytes(b"OLD")

    code = speak.main(
        ["Hello", "--server", "http://x", "--ref-audio", str(tmp_path / "a.wav"),
         "--output-file", str(out_file)]
    )

    # THEN the file is replaced and the silent path never touches aplay
    assert code == 0
    assert out_file.read_bytes() == b"WAVDATA"
    assert aplay_calls == []


def test_main_withPersonaDirAndIndividualFlags_warnsAboutIgnoredFlags(tmp_path, monkeypatch, capsys):
    caps = _caps("dots_capabilities.json")
    _stub_network(monkeypatch, caps, _ok_response())
    monkeypatch.setattr(speak, "run_aplay", lambda audio: True)
    persona = _make_persona(tmp_path / "alice")
    (tmp_path / "a.wav").write_bytes(b"IGNORED")

    code = speak.main(
        ["Hello", "--server", "http://x", "--persona-dir", str(persona),
         "--ref-audio", str(tmp_path / "a.wav"),
         "--ref-audio-transcript", str(tmp_path / "a.txt")]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "Warning: --ref-audio ignored because --persona-dir was given." in out
    assert "Warning: --ref-audio-transcript ignored because --persona-dir was given." in out


# ---------------------------------------------------------------------------
# End-to-end main(): --persona-dir / TTS_SPEAK_PERSONA_DIR
# ---------------------------------------------------------------------------


def test_main_withPersonaDirFlagAndEnvVar_prefersFlag(tmp_path, monkeypatch):
    # GIVEN both an explicit --persona-dir and TTS_SPEAK_PERSONA_DIR, each
    #      pointing at a persona with different content
    caps = _caps("dots_capabilities.json")
    seen = {}
    _stub_network(monkeypatch, caps, _ok_response(), seen)
    monkeypatch.setattr(speak, "run_aplay", lambda audio: True)
    flag_persona = _make_persona(tmp_path / "flag", audio=b"FLAG", transcript="flag words")
    env_persona = _make_persona(tmp_path / "env", audio=b"ENV", transcript="env words")
    monkeypatch.setenv(speak.PERSONA_DIR_ENV_VAR, str(env_persona))

    # WHEN main runs with the explicit flag
    # THEN the flag's persona is used and the env var is silently ignored
    #      (spec rule 1)
    code = speak.main(["Hello", "--server", "http://x", "--persona-dir", str(flag_persona)])

    assert code == 0
    assert base64.b64decode(seen["audio_base64"]) == b"FLAG"
    assert seen["reference_text"] == "flag words"


def test_main_withRefAudioAndPersonaEnvVar_usesRefAudioWithoutWarning(tmp_path, monkeypatch, capsys):
    # GIVEN TTS_SPEAK_PERSONA_DIR set, but the user supplied only --ref-audio
    caps = _caps("chatterbox_capabilities.json")  # no reference_text: simple payload
    seen = {}
    _stub_network(monkeypatch, caps, _ok_response(), seen)
    monkeypatch.setattr(speak, "run_aplay", lambda audio: True)
    env_persona = _make_persona(tmp_path / "env", audio=b"ENV")
    monkeypatch.setenv(speak.PERSONA_DIR_ENV_VAR, str(env_persona))
    (tmp_path / "a.wav").write_bytes(b"REF")

    # WHEN main runs
    # THEN --ref-audio is used and the env var is ignored *silently* — no
    #      "ignored because --persona-dir was given" warning (spec rule 2;
    #      this is the case a naive argparse default gets wrong)
    code = speak.main(["Hello", "--server", "http://x", "--ref-audio", str(tmp_path / "a.wav")])

    assert code == 0
    assert base64.b64decode(seen["audio_base64"]) == b"REF"
    assert "ignored" not in capsys.readouterr().out


def test_main_withOnlyPersonaEnvVar_resolvesLikeExplicitFlag(tmp_path, monkeypatch):
    # GIVEN TTS_SPEAK_PERSONA_DIR set and no reference flags at all
    caps = _caps("dots_capabilities.json")  # advertises reference_text
    seen = {}
    _stub_network(monkeypatch, caps, _ok_response(), seen)
    monkeypatch.setattr(speak, "run_aplay", lambda audio: True)
    persona = _make_persona(tmp_path / "env", audio=b"PERSONA", transcript="env words")
    monkeypatch.setenv(speak.PERSONA_DIR_ENV_VAR, str(persona))

    # WHEN main runs
    # THEN the persona dir is resolved exactly as if the flag had been given
    #      (spec rule 3)
    code = speak.main(["Hello", "--server", "http://x"])

    assert code == 0
    assert base64.b64decode(seen["audio_base64"]) == b"PERSONA"
    assert seen["reference_text"] == "env words"


def test_main_withPersonaEnvVarPointingAtMissingDir_exits1LikeExplicitFlag(
    tmp_path, monkeypatch, capsys
):
    # GIVEN TTS_SPEAK_PERSONA_DIR naming a directory that does not exist
    caps = _caps("dots_capabilities.json")
    monkeypatch.setattr(speak, "fetch_json", lambda url, timeout: caps)
    monkeypatch.setenv(speak.PERSONA_DIR_ENV_VAR, str(tmp_path / "nope"))

    # WHEN main runs with no reference flags
    # THEN the env-var persona goes through the same validation as an
    #      explicit --persona-dir: exit 1, same message (spec rule 3,
    #      "as though it had been given")
    code = speak.main(["Hello", "--server", "http://x"])

    assert code == 1
    assert "Persona directory does not exist or is not a directory" in capsys.readouterr().err


def test_main_withBlankPersonaEnvVar_treatsItAsUnset(tmp_path, monkeypatch, capsys):
    # GIVEN TTS_SPEAK_PERSONA_DIR set to whitespace only and no reference flags
    monkeypatch.setenv(speak.PERSONA_DIR_ENV_VAR, "   ")
    monkeypatch.setattr(speak, "fetch_json", lambda url, timeout: None)

    # WHEN main runs
    # THEN the blank value is treated as unset: the spec's exit-2 usage error
    with pytest.raises(SystemExit) as excinfo:
        speak.main(["Hello", "--server", "http://x"])
    assert excinfo.value.code == 2
    assert "You must supply reference audio or a persona directory!" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# run_aplay: temp-file lifecycle
# ---------------------------------------------------------------------------


def test_run_aplay_withTempfileCreationFailure_propagatesOriginalError(monkeypatch):
    # GIVEN temp-file creation itself fails (e.g. temp filesystem full)
    def boom(*_args, **_kwargs):
        raise OSError("temp filesystem full")

    monkeypatch.setattr(speak.tempfile, "NamedTemporaryFile", boom)

    # WHEN run_aplay runs
    # THEN the original OSError propagates — the cleanup path must not raise
    #      a NameError for an unbound temp file and mask it
    with pytest.raises(OSError, match="temp filesystem full"):
        speak.run_aplay(b"WAVDATA")


def test_run_aplay_onSuccess_leavesNoTempFileBehind(monkeypatch):
    seen = {}

    def fake_run(cmd, **_kwargs):
        seen["path"] = cmd[-1]
        seen["existed_while_running"] = os.path.exists(cmd[-1])
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(speak.subprocess, "run", fake_run)

    # WHEN run_aplay plays successfully
    # THEN the file existed while aplay ran and is gone afterwards
    assert speak.run_aplay(b"WAVDATA") is True
    assert seen["existed_while_running"] is True
    assert os.path.exists(seen["path"]) is False


# ---------------------------------------------------------------------------
# Listing server parameters (--list-server-params)
# ---------------------------------------------------------------------------


def test_main_withListServerParams_printsSummaryAndSkipsSynthesis(monkeypatch, capsys):
    # GIVEN a real dots.tts capabilities document and spies on all network
    #      calls
    caps = _caps("dots_capabilities.json")
    urls: list[str] = []
    posted: list[dict] = []
    monkeypatch.setattr(speak, "fetch_json", lambda url, timeout: (urls.append(url), caps)[1])
    monkeypatch.setattr(
        speak, "post_json", lambda url, payload, timeout: posted.append(payload) or {}
    )

    # WHEN the user asks for the parameter listing (note: no --ref-audio!)
    code = speak.main(["--server", "http://10.0.0.5:8000/", "--list-server-params"])

    # THEN exit 0, only /capabilities was hit, nothing was POSTed, and the
    #      summary shows the metadata and the flags this script exposes
    assert code == 0
    assert urls == ["http://10.0.0.5:8000/capabilities"]
    assert posted == []
    out = capsys.readouterr().out
    assert "Endpoint:    /synthesize" in out
    assert "--num-steps" in out
    assert "--ode-method" in out
    assert "choices: euler, midpoint, rk4" in out
    assert "range: 1..64" in out
    assert "--ref-audio" in out  # file-driven fields shown with their provider flag


def test_main_withListServerParamsAndUnknownSchemaVersion_exits1(tmp_path, monkeypatch, capsys):
    caps = _caps("dots_capabilities.json")
    caps["schema_version"] = 99
    monkeypatch.setattr(speak, "fetch_json", lambda url, timeout: caps)

    code = speak.main(["--server", "http://x", "--list-server-params"])

    assert code == 1
    assert "Server returned unknown schema version 99, expected 2; aborting." in capsys.readouterr().err


def test_format_server_params_withReservedParam_marksItIgnored():
    # GIVEN a server advertising a parameter whose flag is reserved
    # WHEN the summary is rendered
    # THEN the flag is shown but marked as ignored by this script
    caps = {
        "engine": "x",
        "model": "m",
        "device": "cpu",
        "sample_rate": 24000,
        "watermarked": False,
        "endpoint": "/synthesize",
        "languages": None,
        "reference_audio": None,
        "parameters": [
            {"name": "timeout", "type": "integer", "required": False, "default": 5, "description": ""}
        ],
    }
    text = speak.format_server_params(caps)
    assert "--timeout" in text
    assert "[ignored by this script: reserved flag name]" in text


# ---------------------------------------------------------------------------
# save_or_play: save and play are mutually exclusive (spec: 'Output')
# ---------------------------------------------------------------------------


def test_save_or_play_withOutputFile_writesFileWithoutCallingAplay(tmp_path, monkeypatch):
    # GIVEN the silent path: an output file is requested
    aplay_calls = _aplay_spy(monkeypatch)
    out_file = tmp_path / "o.wav"

    # WHEN save_or_play runs
    speak.save_or_play(b"WAVDATA", str(out_file))

    # THEN the file is written and aplay is never invoked
    assert out_file.read_bytes() == b"WAVDATA"
    assert aplay_calls == []


def test_save_or_play_withoutOutputFile_playsAudio(monkeypatch):
    # GIVEN the play path: no output file
    aplay_calls = _aplay_spy(monkeypatch)

    # WHEN save_or_play runs
    speak.save_or_play(b"WAVDATA", None)

    # THEN the audio is handed to aplay exactly once
    assert aplay_calls == [b"WAVDATA"]


def test_save_or_play_withoutOutputFile_whenAplayFails_warnsAndReturnsNormally(monkeypatch, capsys):
    # GIVEN the play path and aplay that exits non-zero
    monkeypatch.setattr(speak, "run_aplay", lambda audio: False)

    # WHEN save_or_play runs, THEN the failure is a warning, not an exception
    speak.save_or_play(b"WAVDATA", None)

    assert "Warning: aplay returned a non-zero exit code" in capsys.readouterr().out


def test_save_or_play_withExistingFileAndDeclinedOverwrite_discardsWithoutCallingAplay(tmp_path, monkeypatch):
    # GIVEN an existing output file and a "n" answer at the overwrite prompt
    aplay_calls = _aplay_spy(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    out_file = tmp_path / "o.wav"
    out_file.write_bytes(b"OLD")

    # WHEN save_or_play runs
    speak.save_or_play(b"NEW", str(out_file))

    # THEN the old file is untouched, nothing is written, nothing is played
    assert out_file.read_bytes() == b"OLD"
    assert aplay_calls == []


def test_save_or_play_withExistingFileAndEofAtPrompt_discardsWithoutCallingAplay(tmp_path, monkeypatch):
    # GIVEN an existing output file and stdin at EOF (piped, non-interactive):
    #      the prompt must be treated as a decline
    aplay_calls = _aplay_spy(monkeypatch)

    def eof(*_args):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    out_file = tmp_path / "o.wav"
    out_file.write_bytes(b"OLD")

    speak.save_or_play(b"NEW", str(out_file))

    assert out_file.read_bytes() == b"OLD"
    assert aplay_calls == []


def test_save_or_play_withUnwritablePath_raisesSpeakError(tmp_path, monkeypatch):
    # GIVEN an output path whose parent directory does not exist
    aplay_calls = _aplay_spy(monkeypatch)

    # WHEN save_or_play tries to write it, THEN a SpeakError (exit-1 family)
    #      is raised and aplay was still never invoked
    with pytest.raises(speak.SpeakError, match=r"Could not write output file"):
        speak.save_or_play(b"WAVDATA", str(tmp_path / "does-not-exist" / "o.wav"))
    assert aplay_calls == []
