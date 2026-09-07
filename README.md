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

| Server file | Engine | Sample rate | Notes |
|---|---|---|---|
| `impl/server_chatterbox.py` | [Chatterbox](https://github.com/resemble-ai/chatterbox) (multilingual) | 24 kHz | 23 language codes; output is PerTh-watermarked by the library |
| `impl/server_omnivoice.py` | [OmniVoice](https://github.com/k2-fsa/OmniVoice) | 24 kHz | auto-transcribes the reference clip with Whisper when `reference_text` is omitted |
| `impl/server_qwen3TTS.py` | [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Base) | 24 kHz | 10 language names + `auto`; falls back to speaker-embedding-only cloning when the transcript is omitted |
| `impl/server_fasterQwen3TTS.py` | [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) (CUDA-graphs Qwen3-TTS fork) | 24 kHz | 10 language names + `auto`; ICL (advanced) mode only — `reference_text` is required; NVIDIA GPU required |
| `impl/server_dotsTTS.py` | [dots.tts](https://github.com/rednote-hilab/dots.tts) | 48 kHz | flow-matching knobs (`num_steps`, `ode_method`, guidance/speaker scales) |

## Quickstart

The servers in `impl/` are standalone scripts built on the shared
`tts-engine-common` package — both live in this repo, so start from a clone:

```bash
git clone https://github.com/scorbo2/tts-serve && cd tts-serve

# Choose one (don't install more than one in the same environment):
#   For Chatterbox: pip install chatterbox-tts
#   For Qwen3-TTS: pip install -U qwen-tts
#   For dots.tts: pip install dots.tts
#   For OmniVoice: pip install omnivoice
#   For faster-qwen3-tts: pip install faster-qwen3-tts

# Now set up the server wrapper:
pip install ./tts-engine-common fastapi uvicorn loguru soundfile

# Run it! (whichever one you installed above)
#  For Chatterbox: python impl/server_chatterbox.py
#  For Qwen3-TTS: python impl/server_qwen3TTS.py
#  For dots.tts: python impl/server_dotsTTS.py
#  For OmniVoice: python impl/server_omnivoice.py
#  For faster-qwen3-tts: python impl/server_fasterQwen3TTS.py
```

On first run, the model weights will be downloaded from HuggingFace.
After first run, no internet connection is required.

**Note**: The install commands assume an active virtual environment (e.g.
`python3 -m venv .venv && source .venv/bin/activate`), or your existing
conda setup. If you run more than one engine, give each one its own
environment — their dependency trees will conflict.

Then use the supplied `speak.py` tool to talk to it:

```bash
# What parameters does this server accept?
python tools/speak.py --server http://localhost:8000 --list-server-params

# Synthesize from reference audio + transcript:
python tools/speak.py --server http://localhost:8000 \
  --ref-audio /path/to/reference.wav \
  --ref-audio-transcript /path/to/transcript.txt
```

You can also use `curl` to quickly verify server health:

```
curl http://localhost:8000/health
```

Every server has:
- `GET /` - informational landing page
- `GET /health` - server health
- `GET /docs` - Swagger docs

Configuration is via environment variables — `*_DEVICE` (`cuda`, `mps`,
`cpu`), `*_HOST`, `*_PORT`, and a model-path variable where applicable.
Each script's module docstring lists them all. For example, to set a local
model path for Qwen3-TTS, set `QWEN3TTS_MODEL=/path/to/model` before startup.

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
impl/                The five engine servers + their (GPU-free) tests.
docs/                Design documents.
```

## Development

```bash
# Full test suite (works on a dev box with no torch/GPU — see impl/README.md)
python -m pytest tts-engine-common/tests/ impl/tests/

# Regenerate the /capabilities snapshots after changing a request schema
python impl/tests/update_snapshots.py
```

## Testing

In addition to a full suite of unit tests, manual testing is possible via
REST calls using `curl` or some Postman-like tool against a running `tts-serve` instance:

```
curl http://localhost:8000/health # shows basic server information including server type
curl http://localhost:8000/capabilities # full capabilities list in Json format
```

Actual speech generation is better handled via the `speak.py` script,
available in the `tools` directory:

```
python3 speak.py -h # show general help
python3 speak-py --server http://localhost:8000 --list-server-params # inspect capabilities
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

