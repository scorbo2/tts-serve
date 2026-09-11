# tts-serve

A wrapper framework that turns local, open-source TTS engines into **one
consistent REST API** with machine-discoverable configuration.

## Why?

There are a large (and growing) number of open-source TTS engines that can be
cloned and run locally. Some ship a demo web app, some a REST API — but each
engine exposes a different set of parameters, or similar parameters under
different names. An application that wants to support several engines is
stuck either offering a lowest-common-denominator experience or building a
separate UI per engine.

tts-serve inverts that. Every engine is wrapped by a small FastAPI server
that speaks one **common language** (a "core" request/response vocabulary)
and exposes everything engine-specific through a machine-readable
`GET /capabilities` endpoint. A client application only ever needs to support
tts-serve — which engine is on the other end becomes a server-side detail,
and the client can build its UI dynamically from `/capabilities`.

## Engines

Here are the TTS engines supported so far.
Suggest a new one on the [project issues page](https://github.com/scorbo2/tts-serve/issues)!

- [Chatterbox](impl/server_chatterbox.md)
- [OmniVoice](impl/server_omnivoice.md)
- [Qwen3-TTS](impl/server_qwen3TTS.md)
- [Faster Qwen3-TTS](impl/server_fasterQwen3TTS.md)
- [dots.tts](impl/server_dotsTTS.md)
- [Index-TTS](impl/server_indexTTS.md)

## Quickstart

Refer to the `Installation` section in one of the engine-specific documents above.

Note: you can install multiple engines on the same server, provided they
each have their own environment (venv, conda). Do not install multiple engines
in the same environment - their dependency trees will conflict.

Once up and running, you can use the supplied `speak.py` tool to talk to it:

```bash
# What parameters does this server accept?
python tools/speak.py --server http://localhost:7500 --list-server-params

# Synthesize from reference audio + transcript:
python tools/speak.py --server http://localhost:7500 \
  --ref-audio /path/to/reference.wav \
  --ref-audio-transcript /path/to/transcript.txt
```

You can also use `curl` to quickly verify server health:

```
curl http://localhost:7500/health
```

Every server has:
- `GET /` - informational landing page
- `GET /health` - server health
- `GET /docs` - Swagger docs
- `POST /synthesize` - synthesize speech

Configuration is via environment variables. Refer to the
engine-specific docs linked above for details.

## The common API

Core request fields (supported by every engine, some optional):

| Field | Meaning |
|---|---|
| `text` | Text to synthesize |
| `audio_base64` | Reference voice sample, base64 (WAV/MP3/OGG/FLAC — anything soundfile decodes) |
| `reference_text` | Transcript of the reference sample (optional where the engine can work without it) |
| `language` | Language hint; form differs per engine (code vs name vs free-form — see `/capabilities`) |
| `seed` | Random seed for reproducibility (echoed in the response) |

Core response fields:

| Field | Meaning |
|---|---|
| `audio_base64` | Generated audio, base64 WAV (PCM 16-bit) |
| `sample_rate` | Output sample rate in Hz |
| `seed` | The seed actually used |
| `time_used` | Wall-clock seconds for synthesis |
| `rtf` | Real-time factor (`time_used` / audio duration), or `null` if not computable |
| `fid` | Per-request id (engine extras follow, e.g. `num_steps`) |

Everything else an engine supports is advertised in `GET /capabilities` with
type, default, bounds, enum, and step info — see
[`docs/01-server-generification.md`](docs/01-server-generification.md) for the
full design, and [`tts-engine-common/README.md`](tts-engine-common/README.md)
for how the endpoint is generated.

## Repository layout

```
tts-engine-common/   Shared FastAPI/Pydantic package (no torch): capabilities
                     derivation, core models, /capabilities route, helpers.
impl/                The six engine servers + their (GPU-free) tests.
docs/                Design documents.
```

## Development

```bash
# Full test suite (works on a dev box with no torch/GPU — see impl/README.md)
python -m pytest tts-engine-common/tests/ impl/tests/ tools/tests/

# Regenerate the /capabilities snapshots after changing a request schema
python impl/tests/update_snapshots.py
```

## Testing

In addition to a full suite of unit tests, manual testing is possible via
REST calls using `curl` or some Postman-like tool against a running `tts-serve` instance:

```
curl http://localhost:7500/health # shows basic server information including server type
curl http://localhost:7500/capabilities # full capabilities list in Json format
```

Actual speech generation is better handled via the `speak.py` script,
available in the `tools` directory:

```
python3 speak.py -h # show general help
python3 speak.py --server http://localhost:7500 --list-server-params # inspect capabilities
```

## Documentation

This project is built with spec-driven development. The human comes up with a detailed spec,
and the LLM (Qwen 3.8 27B mostly) does the actual implementation. The specs should be kept
up to date with code changes, either by amending the spec doc in question, or superseding or
supplementing it with a newer/supplemental one. This avoids code/spec drift over time.

- [`docs/00-project-overview.md`](docs/00-project-overview.md) — project goals and the common vocabulary
- [`docs/01-server-generification.md`](docs/01-server-generification.md) — `/capabilities` design and open questions
- [`docs/02-language-handling.md`](docs/02-language-handling.md) — Amendments to `language` parameter handling
- [`docs/03-speak-script.md`](docs/03-speak-script.md) — addition of a handy command-line testing tool

## Release notes

- **2026-09-10** [v1.0] - initial release

