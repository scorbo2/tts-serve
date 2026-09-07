# Speak script (command-line testing tool)

The project should include a `speak.py` script for testing purposes,
similar to the old `ai-playground/TTS` repo, but updated for use with
any of the impl scripts in `tts-serve`.

## Current state

This project replaced the older TTS server scripts in `ai-playground`.
The `speak.py` script from that repo was not ported to this repo
during development.

## Proposed state

Create a new `tools` directory in `tts-serve`. Its contents
will be a script `speak.py`, based loosely on the original `speak.py`,
and tests for this script (capabilities-parser mapping, payload assembly,
persona-dir resolution, base64 handling, reference text whitespace collapsing, etc).

The original script assumed a hard-coded "lowest common denominator"
set of input parameters. The new script will interrogate the user-supplied
server via `GET /capabilities` to determine the list of required and
optional input parameters.

The script uses only Python standard library dependencies (urllib/json/argparse/etc), 
so it should run on any client machine, even without tts-engine-common installed.

### Old speak.py usage

```
$ speak.py -h
usage: speak.py [-h] [--language LANGUAGE] [--server SERVER] [--ref-audio REF_AUDIO]
                [--ref-audio-transcript REF_AUDIO_TRANSCRIPT] [--seed SEED] [--steps [1-30]]
                [--output-file OUTPUT_FILE]
                text

Synthesize speech using the dots.tts REST API.

positional arguments:
  text                  Text string to synthesize

options:
  -h, --help            show this help message and exit
  --language LANGUAGE   Language code (default: en)
  --server SERVER       API server URL
  --ref-audio REF_AUDIO
                        Path to reference audio WAV file
  --ref-audio-transcript REF_AUDIO_TRANSCRIPT
                        Path to reference transcript text file
  --seed SEED           Random seed for generation (default: null)
  --steps [1-30]        Number of generation steps (default: 12)
  --output-file OUTPUT_FILE
                        Path to save output WAV. If empty, plays via aplay and discards

Examples:
  # Use all defaults (backward compatible with speak.sh)
  speak.py "Hello, how are you?"

  # Custom language, steps, and save to file
  speak.py "Bonjour le monde" --language fr --steps 20 --output-file greeting.wav

  # Override reference files and server
  speak.py "Test" --ref-audio my_voice.wav --ref-audio-transcript my_voice.txt --server http://localhost:7500/tts
```

### Proposed new usage

Use `argparse` for command-line argument handling.

The new script's only known-in-advance parameters (`parse_known_args()`):

- `--server`: (optional, but see Environment Variables section) The server URL (example: `http://10.0.0.5:7500`)
  **Note**: the old script required the full URL, with endpoint (example: `http://10.0.0.5:7500/synthesize`).
  This new script requires just the server name/IP and port. The endpoint is discovered from `/capabilities`.
- `text` (required positional argument): any text to be synthesized.
  Not needed when `--list-server-params` is given (the listing performs no synthesis).
- `--ref-audio`: (optional) any audio file (voice cloning reference audio)
- `--ref-audio-transcript`: (optional) any text file (voice cloning reference transcript)
- `--persona-dir` (optional): see Persona Directory section.
- `--output-file` (optional): see Output section.
- `--timeout` (optional, default 120): request timeout in seconds. Must be a
  positive integer; zero or negative values are rejected at parse time (exit 2).
- `--list-server-params` (optional flag): fetch `/capabilities`, print a
  formatted summary (engine metadata plus the flags this script will expose
  for the engine's parameters, with defaults, choices, and ranges), and exit
  0. Needs no `text` and no reference audio; nothing is synthesized. The
  reserved-name and file-driven conventions above are reflected in the output.

These should be parsed first to ensure that they are present. Additional arguments should **not**
be parsed at this time, because the script does not yet know if they are valid.

The script must query `{server}/capabilities` endpoint and parse the returned Json.
If the server can't be reached, returns any non-2xx code, or does not return a valid
Json *object*, log error and exit with status 1. (Valid JSON that is not an object —
an array or a bare string — is an unknown document shape and must produce the same
clean error, never a traceback.) Check the schema version in the return! If the script
is connecting to a server with an unknown (future) version, log error and exit with
status 1: "Server returned unknown schema version {returnedVersion}, expected {currentVersion}; aborting."

Based on the returned Json document, the script can then use `add_argument()` as needed
for each parameter. This `add_argument()` loop must skip `text`, `audio_base64`, and
`reference_text` to avoid polluting the `-h` help text output.

Now, use strict `parse_args()` (not `parse_known_args()`) to check the remaining supplied arguments.
Any unrecognized/unexpected argument can trigger an `Unrecognized argument: ...` error
and exit status 2. Type checking comes for free with `argparse`. If a given parameter
is marked as numeric by the server, and the user has supplied a string like "test":
`invalid int value: 'test'`. Required enforcement is also free: if an argument is required
and the user omitted it: `the following arguments are required: ...` and exit status 2.

The two-stage argparse requires that `parse_known_args()` extend the **same** parser object,
since the final strict `parse_args()` re-parses the entire command line.

**Enum handling note**: Enum options returned from the server should be handled as argparse choices.
This gives us "invalid choice" for free.
**Gotcha**: the capabilities document carries enum values as *strings*, and argparse
checks `choices` *after* applying the flag's type — so the values must be coerced
through the parameter's type (an integer enum `["1", "2"]` must become choices
`[1, 2]`), or a numeric enum can never match the value the user typed.

**Range handling note**: The script is NOT responsible for handling min/max validation. The
server can already handle that and return a meaningful error for invalid values. Let the server
be the source of truth for min/max value handling instead of inventing our own range validators.

**Boolean handling note**: the script should use a string-to-bool function for boolean arguments
(explicit checks for "true", "yes", "1", "on", "false", "no", "0", "off") to avoid problems
with argparse's default handling of "any non-empty string is true".

The script should build the payload only from the arguments the user actually specified,
instead of filling them in with their default values. For example, if the user omits
the `seed` argument (optional, with `default: null`), it is better to omit it from
the payload rather than specify it with its default value.

**Kebab-case versus snake-case**: All underscores in server parameters should be
replaced with `-` to maintain the kebab-style argument convention. For example,
the `num_steps` argument should be given to this script as `--num-steps`, not `--num_steps`.

If the server returns a parameter whose name conflicts with one of the script's
reserved argument names, that server parameter should be ignored with a warning.
For example, if the server specifies a parameter named `output_file`, that would
conflict with this script's `--output-file` argument. Log warning "Server returned
a reserved parameter name output_file; ignoring it." and continue. The reserved set
also includes argparse's built-in `help`, since a parameter named `help` would
otherwise crash argparse with "conflicting option string".

The computed payload should be POSTed to `{server}/{endpoint}` using the server's
advertised endpoint (do not hard-code `/synthesize`). Normalize trailing slashes
on `{server}`. Any non-2xx return code: log error including the return code, and
exit with status 1.

The return body must be valid Json, and must contain a non-empty `audio_base64` field.
Otherwise, exit with status 1 and error "Invalid response from server."

### Handling reference audio and reference transcripts

All `tts-serve` implementation scripts expect `audio_base64`, but there is no `--audio-base64`
argument. Instead, the script should accept `--ref-audio` pointing to any audio file, and handle
the base64 encoding on behalf of the user. This matches the behavior of the old script.

Any `tts-serve` implementation script that requires a transcript of the reference audio
will expect `reference_text`, but there is no `--reference-text` argument. Instead, the script
should accept `--ref-audio-transcript` pointing to any text file, and handle loading text
from that file (collapsing contents with `" ".join(raw.split())`). 
This matches the behavior of the old script.

Both `--ref-audio` and `--ref-audio-transcript` are superseded by `--persona-dir` if present.

It is never an error to specify `--ref-audio-transcript`, even for `tts-serve` implementations
that do not ask for reference text (Chatterbox, for example). The script can simply ignore
the supplied reference text with a warning ("Reference text ignored as this server does not accept it").

It is always an error to omit `--ref-audio` (or `--persona-dir`, if `--ref-audio` is not specified).
ALL tts-serve implementation scripts require `audio_base64`. So, even though both `--ref-audio`
and `--persona-dir` are marked as optional arguments, at least one of them must be specified!
Exit with status 2 and message "You must supply reference audio or a persona directory!"
Note that `--persona-dir` can also be specified by an environment variable. See the Environment
Variables section for details.

## Persona Directory

Instead of `--ref-audio` and `--ref-audio-transcript`, the user can supply `--persona-dir`
(or the `TTS_SPEAK_PERSONA_DIR` env var), pointing to any directory. This directory should
contain `ref.wav` representing the reference audio, and `ref.txt` representing the reference
transcript. It is an error if either file is missing, or if the given directory does not
exist or can't be read.

If both `--ref-audio` and `--persona-dir` are given, `--ref-audio` is ignored with a warning.

If both `--ref-audio-transcript` and `--persona-dir` are given, `--ref-audio-transcript` is
ignored with a warning.

## Output

If `--output-file` is specified, the returned audio is base64-decoded and written
to the specified output file, with a `[y/N]` overwrite confirmation prompt if the
specified file already exists. If stdin is EOF (piped, non-interactive), "N" is
assumed. If "N" is selected, the output is silently discarded. This is not an error.

If `--output-file` is not specified, the returned audio is base64-decoded and
given to `aplay` for immediate output. If `aplay` is not installed, log error
and exit with status 1. If `aplay` returns a non-zero exit code, log warning and
continue.

Note that `--output-file` is an implied "silent" option - the script EITHER
saves the returned audio to a file OR outputs it to `aplay`, never both.

All `tts-serve` implementation scripts return `seed`, `time_used`, and `rtf`. These should
be logged regardless of the output destination of the audio. Log this before attempting
to play/save the audio! Otherwise, it may get skipped if audio output or save fails.

The stats should be logged in a human-readable block: a `Generation stats:` header line,
followed by one indented line per stat. `time_used` and `rtf` are rounded to two decimal
places (with the `s` unit suffix on `time_used`); the seed is an integer and is shown
whole. Absent or non-numeric values are shown as `null` / as-is rather than crashing the
run after a successful synthesis. For example:

```
Generation stats:
  seed: 42
  time_used: 3.69s
  rtf: 0.12
```

## Logging

Normal logging and warnings to stdout, errors to stderr. User can redirect as needed.

## Script exit codes

- 0: normal termination
- 1: catch-all error code
- 2: argparse error code

## Tests

`AGENTS.md` in this repo requires a full `python -m pytest tts-engine-common/tests impl/tests`
after every code change, to ensure all tests are green. That instruction should be updated to
include the tests for this script in the new `tools` directory.

## Environment variables

### Specifying server

The `--server` command line argument is technically required (we can't discover capabilities
without a TTS server), but must be marked as optional in argparse, as its value can come
from two sources:

- the `--server` command line argument itself
- a `TTS_SPEAK_SERVER` environment variable

At least one of these must be provided. The suggested approach is to check for the environment
variable first, then allow the command line argument to override it:

```
parser = argparse.ArgumentParser()
parser.add_argument(
    "--server",
    required=False,
    default=os.environ.get("TTS_SPEAK_SERVER"),
)

args = parser.parse_args()

if args.server is None:
    parser.error("the following arguments are required: --server")
```

To make this clear to the user, custom help text should be supplied, rather than relying
on argparse's auto-generated text. Something like this:

```
help=(
    "Server address. Required unless TTS_SPEAK_SERVER env var is set "
    f"(currently: {os.environ.get('TTS_SPEAK_SERVER', 'not set')}). "
    "Command-line value takes precedence. Note: supply base URL only. "
    "The endpoint is discovered from /capabilities."
),
```

If both the env var and the command line argument are specified, the command line arg wins.
If neither is set, exit with code 2 and message: "the following arguments are required: --server"

Note that the `--list-server-params` option should work regardless of whether the server
is supplied via `--server` or via `TTS_SPEAK_SERVER`, as long as at least one is valid.
Same precedence order applies: the command line arg wins if both are specified.

**Error precedence**: missing server is reported before missing reference audio / missing text.

### Specifying persona directory

The `--persona-dir` argument will also have an environment variable fallback
`TTS_SPEAK_PERSONA_DIR`. This env var will **only** apply when `--persona-dir` is not specified.
The precedence order for this is:

1. `--persona-dir` supersedes both `--ref-audio` and the env var.
2. `--ref-audio` supersedes the env var, if `--persona-dir` is not specified.
3. if neither `--persona-dir` nor `--ref-audio` is specified, use the value from
   the environment variable as though it had been given to `--persona-dir`.
4. if none of the above are specified, exit with code 2 and message
   "You must supply reference audio or a persona directory!"

If either env var is set to an empty/blank string, it should be treated as unset.

### Notes for testing

In order to keep the tests hermetic on a dev box that may have the environment variables specified,
the tests should have an autouse fixture that strips both `TTS_SPEAK_*` env vars.

In previous versions of this specification, `--server` was marked as required, so there are
currently no explicit tests for missing server (we got it for free from argparse itself).
Now that it is marked as optional, with custom rules around resolving a value of it,
there should be new tests to cover these scenarios: neither env var nor command line arg supplied
(should error), both env var and command line arg supplied (should ignore env var), empty
env var supplied (should be treated as unset), only env var supplied (should use the env var value),
only command line arg supplied (should use the command line arg).

The same scenarios should be covered for TTS_SPEAK_PERSONA_DIR: both --persona-dir and the env var
(use the flag, env var silently ignored); only --ref-audio with the env var set (use --ref-audio,
env var silently ignored with no warning — this distinguishes an explicit flag from the env-var fallback);
only the env var (resolved exactly as if given to --persona-dir, including the exit-1 validation errors
for a bad path); empty env var (treated as unset); and none of --ref-audio, --persona-dir, or the env
var (exit 2 — note the existing test currently asserts exit 1 and must be updated).

