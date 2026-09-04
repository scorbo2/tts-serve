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

| Server file | Engine | Sample rate | Notes |
|---|---|---|---|
| `impl/server_chatterbox.py` | [Chatterbox](https://github.com/resemble-ai/chatterbox) (multilingual) | 24 kHz | 23 language codes; output is PerTh-watermarked by the library |
| `impl/server_omnivoice.py` | [OmniVoice](https://github.com/k2-fsa/OmniVoice) | 24 kHz | auto-transcribes the reference clip with Whisper when `reference_text` is omitted |
| `impl/server_qwen3TTS.py` | [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Base) | 24 kHz | 10 language names + `auto`; falls back to speaker-embedding-only cloning when the transcript is omitted |
| `impl/server_dotsTTS.py` | [dots.tts](https://github.com/rednote-hilab/dots.tts) | 48 kHz | flow-matching knobs (`num_steps`, `ode_method`, guidance/speaker scales) |

## Quickstart

Each server is a standalone script. Install the engine package plus the
shared bits, then run:

```bash
pip install chatterbox-tts fastapi uvicorn loguru soundfile
pip install tts-engine-common   # or: pip install -e ./tts-engine-common
python impl/server_chatterbox.py
```

(Replace `chatterbox-tts` with `omnivoice`, `qwen-tts`, or `dots.tts` and
`server_chatterbox.py` with the matching file for the other engines. On first
start the model weights download from HuggingFace.)

Then talk to it:

```bash
# What parameters does this server accept? (machine-readable)
curl -s localhost:8000/capabilities | python -m json.tool

# Synthesize: text + reference voice sample (base64)
curl -s -X POST localhost:8000/synthesize \
  -H 'Content-Type: application/json' \
  -d "{\"text\": \"Hello there\", \"audio_base64\": \"$(base64 -w0 reference.wav)\"}"
```

Every server also has `GET /` (landing page), `GET /health`, and Swagger
docs at `GET /docs`.

Configuration is via environment variables — `*_DEVICE` (`cuda`, `mps`,
`cpu`), `*_HOST`, `*_PORT`, and a model-path variable where applicable.
Each script's module docstring lists them all.

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
impl/                The four engine servers + their (GPU-free) tests.
docs/                Design documents.
```

## Development

```bash
# Full test suite (works on a dev box with no torch/GPU — see impl/README.md)
python -m pytest tts-engine-common/tests/ impl/tests/

# Regenerate the /capabilities snapshots after changing a request schema
python impl/tests/update_snapshots.py
```

## Documentation

- [`docs/00-project-overview.md`](docs/00-project-overview.md) — goals and the common vocabulary
- [`docs/01-server-generification.md`](docs/01-server-generification.md) — `/capabilities` design and open questions
