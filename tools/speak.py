#!/usr/bin/env python3
"""speak.py — command-line testing tool for tts-serve engine servers.

Synthesizes speech by POSTing to any tts-serve implementation.  The tunable
parameters are not hard-coded: the script first queries ``GET {server}/capabilities``,
builds the rest of its CLI from the returned document, and only then validates
the user's remaining arguments.  This is the client-side half of design
decision D4 (docs/01-server-generification.md): the server's request model is
the single source of truth, so the CLI can never drift from it.

Usage:
    python tools/speak.py "Hello, world" --server http://10.0.0.5:8000 --ref-audio my_voice.wav
    python tools/speak.py "Bonjour le monde" --server http://10.0.0.5:8000 \
        --persona-dir ./alice --language fr --output-file greeting.wav
    python tools/speak.py --server http://10.0.0.5:8000 --list-server-params

Standard library only, on purpose: this must run on any client box, even
without tts-engine-common (or torch) installed.

Full behavioral spec: docs/03-speak-script.md.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import urllib.error
import urllib.request

# Kept in sync with tts-engine-common's SCHEMA_VERSION.  This script must not
# import tts-engine-common (stdlib-only), so the constant is deliberately
# duplicated here: a mismatch against a future server is a loud, deliberate
# failure, not a silent misparse of an unknown document shape.
SUPPORTED_SCHEMA_VERSION = 2

# Request fields handled by dedicated CLI machinery (positional / file-backed),
# so they are never exposed as dynamic arguments.  The values name the CLI
# surface that supplies each field (used by --list-server-params).
_FILE_DRIVEN_PROVIDERS = {
    "text": "positional 'text'",
    "audio_base64": "--ref-audio",
    "reference_text": "--ref-audio-transcript",
}
_FILE_DRIVEN_FIELDS = frozenset(_FILE_DRIVEN_PROVIDERS)

# CLI flags owned by this script (plus argparse's built-in ``help``, which a
# parameter named "help" would otherwise blow up with "conflicting option
# string").  A server parameter whose kebab-case name collides with one of
# these is ignored with a warning instead of silently shadowing a flag.
_RESERVED_FLAGS = frozenset(
    (
        "server",
        "help",
        "ref-audio",
        "ref-audio-transcript",
        "persona-dir",
        "output-file",
        "timeout",
        "list-server-params",
    )
)

_PERSONA_AUDIO_NAME = "ref.wav"
_PERSONA_TRANSCRIPT_NAME = "ref.txt"

_TRUE_WORDS = frozenset(("true", "yes", "1", "on"))
_FALSE_WORDS = frozenset(("false", "no", "0", "off"))


class SpeakError(Exception):
    """A user- or server-facing failure that must exit with status 1."""


def kebab(name: str) -> str:
    """Map a server parameter name to its CLI flag (underscores to dashes)."""
    return name.replace("_", "-")


def str_to_bool(value: str) -> bool:
    """argparse type for boolean parameters.

    argparse's built-in bool coercion treats *any* non-empty string as True,
    so boolean values are matched against explicit word lists instead.
    """
    word = value.strip().lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    raise argparse.ArgumentTypeError(
        f"invalid boolean value: {value!r} (expected one of true/false, yes/no, on/off, 1/0)"
    )


def positive_timeout(value: str) -> int:
    """argparse type for --timeout: whole seconds, minimum 1.

    urlopen treats 0 as non-blocking and negative values as 'block
    indefinitely', so a degenerate timeout is rejected at parse time (exit 2)
    instead of being discovered mid-request.
    """
    try:
        seconds = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from exc
    if seconds < 1:
        raise argparse.ArgumentTypeError(f"timeout must be at least 1 second (got {seconds})")
    return seconds


def join_url(base: str, path: str) -> str:
    """Join a server base URL and a path, normalizing duplicate slashes."""
    return base.rstrip("/") + "/" + path.lstrip("/")


def load_audio_base64(path: str) -> str:
    """Read an audio file and return its base64 encoding (the wire format)."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except OSError as exc:
        raise SpeakError(f"Reference audio file could not be read: {path} ({exc})") from exc


def load_transcript(path: str) -> str:
    """Read a transcript file, collapsing all whitespace to single spaces.

    Transcript files are about words, not line layout, so newlines and runs of
    spaces are folded exactly like the old ai-playground script did.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise SpeakError(f"Reference transcript file could not be read as UTF-8 text: {path} ({exc})") from exc
    return " ".join(raw.split())


def resolve_reference_sources(
    ref_audio: str | None, ref_audio_transcript: str | None, persona_dir: str | None
) -> tuple[str, str | None]:
    """Resolve the reference flags to concrete file paths.

    Returns ``(audio_path, transcript_path_or_None)``.  ``persona_dir``
    supersedes both individual flags (with a warning per the spec).  Raises
    SpeakError for missing or unreadable inputs.
    """
    if persona_dir:
        if ref_audio:
            print("Warning: --ref-audio ignored because --persona-dir was given.")
        if ref_audio_transcript:
            print("Warning: --ref-audio-transcript ignored because --persona-dir was given.")
        if not os.path.isdir(persona_dir):
            raise SpeakError(f"Persona directory does not exist or is not a directory: {persona_dir}")
        if not os.access(persona_dir, os.R_OK):
            raise SpeakError(f"Persona directory is not readable: {persona_dir}")
        audio_path = os.path.join(persona_dir, _PERSONA_AUDIO_NAME)
        transcript_path = os.path.join(persona_dir, _PERSONA_TRANSCRIPT_NAME)
        _require_readable_file(audio_path, "Persona directory reference audio (ref.wav)")
        _require_readable_file(transcript_path, "Persona directory reference transcript (ref.txt)")
        return audio_path, transcript_path

    if ref_audio:
        _require_readable_file(ref_audio, "Reference audio")
    if ref_audio_transcript:
        _require_readable_file(ref_audio_transcript, "Reference transcript")
    return ref_audio, ref_audio_transcript


def _require_readable_file(path: str, label: str) -> None:
    if not os.path.isfile(path) or not os.access(path, os.R_OK):
        raise SpeakError(f"{label} not found or not readable: {path}")


def check_schema_version(capabilities: dict) -> None:
    """Reject capabilities documents we do not understand (spec: exit 1)."""
    if not isinstance(capabilities, dict):
        # A misrouted URL or chatty gateway can return perfectly valid JSON
        # that is *not* an object (e.g. []).  That is an unknown document
        # shape and must fail with the same clean message as everything else,
        # not with an AttributeError traceback from .get().
        raise SpeakError(
            f"Server returned a non-object capabilities document "
            f"({type(capabilities).__name__}); aborting."
        )
    version = capabilities.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise SpeakError(
            f"Server returned unknown schema version {version}, "
            f"expected {SUPPORTED_SCHEMA_VERSION}; aborting."
        )


def validate_response(body: object) -> dict:
    """Enforce the frozen response core: JSON object with non-empty audio_base64."""
    if not isinstance(body, dict) or not body.get("audio_base64"):
        raise SpeakError("Invalid response from server.")
    return body


def _http(url: str, data: bytes | None, timeout: int) -> tuple[int, bytes]:
    """Perform one HTTP request; return (status, body) without raising on HTTP errors."""
    request = urllib.request.Request(
        url,
        data=data,
        method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        raise SpeakError(f"Failed to connect to server: {exc.reason}") from exc


def _snippet(body: bytes, limit: int = 500) -> str:
    text = body.decode("utf-8", errors="replace").strip()
    return text[:limit] if text else "(empty body)"


def fetch_json(url: str, timeout: int) -> dict:
    """GET a JSON document; any failure is a SpeakError (spec: exit 1)."""
    status, body = _http(url, None, timeout)
    if not 200 <= status < 300:
        raise SpeakError(f"GET {url} returned HTTP {status}: {_snippet(body)}")
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpeakError(f"GET {url} did not return valid JSON: {exc}") from exc


def fetch_capabilities(server: str, timeout: int) -> dict:
    """Fetch GET {server}/capabilities and reject an unknown schema version."""
    capabilities = fetch_json(join_url(server, "capabilities"), timeout)
    check_schema_version(capabilities)
    return capabilities


def post_json(url: str, payload: dict, timeout: int) -> object:
    """POST a JSON payload; any failure is a SpeakError (spec: exit 1)."""
    status, body = _http(url, json.dumps(payload).encode("utf-8"), timeout)
    if not 200 <= status < 300:
        raise SpeakError(f"POST {url} returned HTTP {status}: {_snippet(body)}")
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        # The spec assigns one exact message to "not valid JSON or missing audio".
        raise SpeakError("Invalid response from server.") from exc


def build_base_parser() -> argparse.ArgumentParser:
    """Stage-1 parser: only the arguments known in advance (spec: 'Proposed new usage')."""
    parser = argparse.ArgumentParser(
        prog="speak.py",
        description=(
            "Synthesize speech via any tts-serve engine.  Engine-specific parameters "
            "are discovered from the server's /capabilities endpoint."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s \"Hello, world\" --server http://10.0.0.5:8000 --ref-audio my_voice.wav\n"
            "  %(prog)s \"Bonjour le monde\" --server http://10.0.0.5:8000 "
            "--persona-dir ./alice --language fr\n"
            "  %(prog)s \"Test\" --server http://10.0.0.5:8000 --ref-audio v.wav --output-file out.wav\n"
            "  %(prog)s --server http://10.0.0.5:8000 --list-server-params\n"
        ),
    )
    # Optional at the argparse level on purpose: --list-server-params needs
    # no text, and main() turns a missing text into a usage error (exit 1)
    # for the synthesis path only, before any network traffic.
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Text string to synthesize (not needed with --list-server-params)",
    )
    parser.add_argument(
        "--server",
        required=True,
        metavar="URL",
        help="Server base URL, e.g. http://10.0.0.5:8000. "
        "The synthesis endpoint is discovered via /capabilities.",
    )
    parser.add_argument(
        "--ref-audio",
        default=None,
        metavar="PATH",
        help="Path to reference audio file (voice cloning). "
        "At least one of --ref-audio or --persona-dir is required.",
    )
    parser.add_argument(
        "--ref-audio-transcript",
        default=None,
        metavar="PATH",
        help="Path to a text file with the reference audio transcript. "
        "Ignored (with a warning) by engines that do not accept reference_text.",
    )
    parser.add_argument(
        "--persona-dir",
        default=None,
        metavar="DIR",
        help="Directory containing ref.wav and ref.txt; "
        "supersedes --ref-audio and --ref-audio-transcript.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        metavar="PATH",
        help="Path to save the output WAV.  Implies 'silent': the audio is "
        "saved, not played via aplay.  An existing file prompts for "
        "overwrite confirmation.",
    )
    parser.add_argument(
        "--timeout",
        type=positive_timeout,
        default=120,
        help="Request timeout in seconds, minimum 1 (default: 120).",
    )
    parser.add_argument(
        "--list-server-params",
        action="store_true",
        help="Print the engine's /capabilities summary (metadata and the "
        "flags this script will expose) and exit.  Needs no text or "
        "reference audio; nothing is synthesized.",
    )
    return parser


def _parameter_specs(capabilities: dict) -> list[dict]:
    """The capabilities' parameter entries, narrowed to well-formed dicts.

    A broken document must not take down the stage-2 parser or the
    --list-server-params printer, so malformed entries are skipped with a
    warning rather than raising AttributeError from .get().
    """
    params = capabilities.get("parameters")
    if params is None:
        return []
    if not isinstance(params, list):
        print("Warning: 'parameters' in the capabilities document is not a list; ignoring it.")
        return []
    specs: list[dict] = []
    for spec in params:
        if isinstance(spec, dict):
            specs.append(spec)
        else:
            print(f"Warning: Skipping malformed capabilities parameter entry: {spec!r}")
    return specs


def _coerce_enum(enum: list[object], spec_type: str) -> list[object]:
    """Coerce the document's enum values into the parameter's own type.

    The capabilities document always carries enum values as strings, while
    argparse checks ``choices`` *after* applying the flag's type.  Without
    this, an integer enum like ["1", "2"] could never match the int value
    the user actually typed.  A value that does not coerce means the
    document contradicts itself, which is a loud error (R1: never
    silently misdescribe).
    """

    def convert(value: object) -> object:
        if spec_type == "integer":
            return int(str(value))
        if spec_type == "number":
            return float(str(value))
        if spec_type == "boolean":
            return str_to_bool(str(value))
        return value

    coerced: list[object] = []
    for value in enum:
        try:
            coerced.append(convert(value))
        except (ValueError, argparse.ArgumentTypeError) as exc:
            raise SpeakError(
                f"Server advertises enum value {value!r} for a {spec_type} parameter; "
                f"the capabilities document is inconsistent."
            ) from exc
    return coerced


def add_dynamic_arguments(parser: argparse.ArgumentParser, capabilities: dict) -> list[str]:
    """Register one CLI flag per server parameter (stage 2 of the two-stage parse).

    Skips the file-driven fields, warns-and-skips reserved-name collisions,
    and maps the capabilities types onto argparse (enum to type-coerced
    choices, boolean to str_to_bool; min/max are deliberately left to the
    server as source of truth).  Returns the snake_case names added, so the
    payload builder knows which namespace attributes correspond to server
    parameters.
    """
    added: list[str] = []
    for spec in _parameter_specs(capabilities):
        name = spec.get("name", "")
        if not name or name in _FILE_DRIVEN_FIELDS:
            continue
        flag = kebab(name)
        if flag in _RESERVED_FLAGS:
            print(f"Warning: Server returned a reserved parameter name {name}; ignoring it.")
            continue
        kwargs: dict = {
            # SUPPRESS => an unspecified optional arg has no namespace attribute
            # at all, which is how build_payload() knows to omit it.
            "default": argparse.SUPPRESS,
            "help": spec.get("description") or None,
        }
        if spec.get("required"):
            kwargs["required"] = True
        spec_type = spec.get("type", "string")
        if spec_type == "integer":
            kwargs["type"] = int
        elif spec_type == "number":
            kwargs["type"] = float
        elif spec_type == "boolean":
            kwargs["type"] = str_to_bool
        enum = spec.get("enum")
        if enum:
            kwargs["choices"] = _coerce_enum(list(enum), spec_type)
        parser.add_argument(f"--{flag}", **kwargs)
        added.append(name)
    return added


def build_payload(
    args: argparse.Namespace,
    dynamic_names: list[str],
    audio_base64: str,
    reference_text: str | None,
) -> dict:
    """Assemble the POST body from user-supplied arguments only.

    Unspecified optional parameters are omitted entirely rather than filled
    with the server's defaults: a null default (e.g. seed) carries meaning
    ("pick randomly"), and echoing defaults would only bloat the request.
    """
    payload: dict = {"text": args.text, "audio_base64": audio_base64}
    if reference_text is not None:
        payload["reference_text"] = reference_text
    for name in dynamic_names:
        value = getattr(args, name, None)
        if value is not None:
            payload[name] = value
    return payload


def run_aplay(audio_bytes: bytes) -> bool:
    """Play a WAV buffer via aplay; return True on a clean exit.

    Raises SpeakError when aplay is not installed at all (spec: exit 1).  A
    non-zero aplay exit is *not* fatal (spec: warn and continue).
    """
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        result = subprocess.run(["aplay", "-q", tmp_path], check=False)
        return result.returncode == 0
    except FileNotFoundError as exc:
        raise SpeakError(
            "'aplay' not found. Please install the ALSA utilities (e.g. 'sudo apt install alsa-utils')."
        ) from exc
    finally:
        # Guarded: if temp-file creation itself fails, an unguarded
        # os.unlink(tmp.name) would raise NameError and mask the real error.
        if tmp_path is not None:
            os.unlink(tmp_path)


def _confirm_overwrite(path: str) -> bool:
    try:
        answer = input(f"File '{path}' already exists. Overwrite? [y/N] ").strip().lower()
    except EOFError:
        # No interactive stdin (piped invocation): treat as a decline.
        return False
    return answer == "y"


def save_or_play(audio_bytes: bytes, output_file: str | None) -> None:
    """Either save the audio to output_file or play it via aplay — never both.

    Per the spec ('Output'), --output-file is an implied "silent" option: it
    lets the user keep the returned audio on boxes with no aplay at all.
    Declining the overwrite prompt silently discards the audio (not an error).
    """
    if output_file:
        if os.path.exists(output_file) and not _confirm_overwrite(output_file):
            return
        try:
            with open(output_file, "wb") as f:
                f.write(audio_bytes)
        except OSError as exc:
            raise SpeakError(f"Could not write output file {output_file}: {exc}") from exc
        print(f"Saved audio to {output_file}")
        return
    if not run_aplay(audio_bytes):
        print("Warning: aplay returned a non-zero exit code; audio may not have played.")


def _format_stats(response: dict) -> str:
    rtf = response.get("rtf")
    return (
        f"seed={response.get('seed')} "
        f"time_used={response.get('time_used')}s "
        f"rtf={'null' if rtf is None else rtf}"
    )


def _format_number(value: object) -> str:
    """Render a JSON number without a spurious .0 (64.0 -> '64')."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _format_default(value: object) -> str:
    return "null" if value is None else _format_number(value)


def format_server_params(capabilities: dict) -> str:
    """Render the --list-server-params summary.

    Engine metadata first, then the exact flags this script will expose for
    the engine's parameters (file-driven fields are shown with the CLI flag
    that supplies them, reserved collisions are marked as ignored).
    """
    lines: list[str] = [
        f"Engine:      {capabilities.get('engine', '?')}",
        f"Model:       {capabilities.get('model', '?')}",
        f"Device:      {capabilities.get('device', '?')}",
        f"Sample rate: {capabilities.get('sample_rate', '?')} Hz",
        f"Watermarked: {'yes' if capabilities.get('watermarked') else 'no'}",
        f"Endpoint:    {capabilities.get('endpoint') or '/synthesize'}",
    ]
    languages = capabilities.get("languages")
    lines.append(f"Languages:   {', '.join(languages) if languages else '(none advertised)'}")

    reference = capabilities.get("reference_audio")
    if isinstance(reference, dict):
        bits = ["required" if reference.get("required") else "optional"]
        if reference.get("formats"):
            bits.append("formats: " + ", ".join(reference["formats"]))
        max_duration = reference.get("max_duration_s")
        if max_duration is not None:
            bits.append(f"max {_format_number(max_duration)}s")
        lines.append("Reference:   " + ", ".join(bits))
        if reference.get("note"):
            lines.append(textwrap.indent(textwrap.fill(str(reference["note"]), width=68), "              "))
    else:
        lines.append("Reference:   not supported by this engine")

    lines.append("")
    lines.append("Parameters (this script's flags for them):")
    for spec in _parameter_specs(capabilities):
        name = spec.get("name", "")
        if not name:
            continue
        spec_type = spec.get("type", "string")
        required = bool(spec.get("required"))
        if name in _FILE_DRIVEN_PROVIDERS:
            flag = _FILE_DRIVEN_PROVIDERS[name]
        else:
            flag = f"--{kebab(name)}"
        default = "" if required else f"  (default: {_format_default(spec.get('default'))})"
        lines.append(f"  {flag:<28} {spec_type:<8} {'required' if required else 'optional'}{default}")
        if flag.startswith("--") and kebab(name) in _RESERVED_FLAGS:
            lines.append("      [ignored by this script: reserved flag name]")
        description = str(spec.get("description") or "").strip()
        if description:
            lines.append(textwrap.indent(textwrap.fill(description, width=68), "    "))
        enum = spec.get("enum")
        if enum:
            lines.append(f"    choices: {', '.join(str(v) for v in enum)}")
        if spec.get("min") is not None or spec.get("max") is not None:
            lo = spec.get("min")
            hi = spec.get("max")
            lo_text = "" if lo is None else _format_number(lo)
            hi_text = "" if hi is None else _format_number(hi)
            lines.append(f"    range: {lo_text}..{hi_text}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the tool; return the process exit code (0 ok, 1 error; argparse exits 2)."""
    # Stage 1: parse only the arguments known in advance.  Unknown tokens pass
    # through unvalidated; argparse itself still exits 2 for missing --server
    # or -h.  The text positional and the reference flags are checked below,
    # after --list-server-params has had a chance to short-circuit, because
    # the listing needs neither.
    parser = build_base_parser()
    args, _unknown = parser.parse_known_args(argv)

    # Discovery only: no text, no reference audio, no synthesis.
    if args.list_server_params:
        try:
            capabilities = fetch_capabilities(args.server, args.timeout)
        except SpeakError as exc:
            return _fail(str(exc))
        print(format_server_params(capabilities))
        return 0

    # Fail fast, before any network traffic: every tts-serve engine requires
    # text and a reference clip, so missing sources are usage errors, not
    # server errors.
    if args.text is None:
        return _fail("You must supply the text to synthesize!")
    if not args.ref_audio and not args.persona_dir:
        return _fail("You must supply reference audio or a persona directory!")

    try:
        capabilities = fetch_capabilities(args.server, args.timeout)
    except SpeakError as exc:
        return _fail(str(exc))

    try:
        dynamic_names = add_dynamic_arguments(parser, capabilities)
    except SpeakError as exc:
        return _fail(str(exc))
    except argparse.ArgumentError as exc:
        return _fail(f"Cannot register server parameter: {exc}")
    # Stage 2, same parser object: re-parses the entire command line strictly,
    # so engine-specific typos and type errors now exit 2.
    args = parser.parse_args(argv)

    try:
        audio_path, transcript_path = resolve_reference_sources(
            args.ref_audio, args.ref_audio_transcript, args.persona_dir
        )
        accepts_reference_text = any(
            p.get("name") == "reference_text" for p in _parameter_specs(capabilities)
        )
        reference_text = None
        if transcript_path is not None:
            if accepts_reference_text:
                reference_text = load_transcript(transcript_path)
            else:
                print("Warning: Reference text ignored as this server does not accept it.")
        audio_base64 = load_audio_base64(audio_path)
    except SpeakError as exc:
        return _fail(str(exc))

    payload = build_payload(args, dynamic_names, audio_base64, reference_text)

    try:
        endpoint = capabilities.get("endpoint") or "/synthesize"
        response = post_json(join_url(args.server, endpoint), payload, args.timeout)
        validate_response(response)
    except SpeakError as exc:
        return _fail(str(exc))

    # Log the response stats before playing/saving: if audio output fails, the
    # user still sees what the engine produced (spec: 'Output', new behaviour).
    print(_format_stats(response))

    try:
        audio_bytes = base64.b64decode(response["audio_base64"])
    except (binascii.Error, KeyError, TypeError) as exc:
        return _fail("Invalid response from server.")
    try:
        save_or_play(audio_bytes, args.output_file)
    except SpeakError as exc:
        return _fail(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
