# Implementations

One FastAPI server per TTS engine. Each server:

- loads its model once on startup (HuggingFace download on first run),
- exposes `GET /` (landing page), `GET /health`, `GET /capabilities`, and
  `POST /synthesize`,
- validates requests with a Pydantic model and derives `/capabilities` from
  that same model via [`tts-engine-common`](../tts-engine-common/README.md),
- rejects unknown fields (`422`) and bad reference audio (`400`) before
  touching the model.

| File | Engine | Output rate | Device env | Model env |
|---|---|---|---|---|
| `server_chatterbox.py` | Chatterbox multilingual | 24 kHz | `CHATTERBOX_DEVICE` (cuda/mps/cpu, default cuda) | `CHATTERBOX_T3_MODEL` (v2/v3 or a `.safetensors` name, default v3) |
| `server_omnivoice.py` | OmniVoice | 24 kHz | `OMNIVOICE_DEVICE` (default cuda) | `OMNIVOICE_MODEL` (HF id or path, default `k2-fsa/OmniVoice`) |
| `server_qwen3TTS.py` | Qwen3-TTS Base | 24 kHz | `QWEN3TTS_DEVICE` (default cuda) | `QWEN3TTS_MODEL` (default `Qwen/Qwen3-TTS-12Hz-1.7B-Base`) |
| `server_dotsTTS.py` | dots.tts | 48 kHz | — (the runtime auto-selects CUDA/CPU) | `DOTS_TTS_MODEL` (default `rednote-hilab/dots.tts-soar`) |

All servers also take `*_HOST` (default `0.0.0.0`) and `*_PORT`
(default `8000`).

## Running

```bash
pip install <engine-package> fastapi uvicorn loguru soundfile
pip install tts-engine-common        # or: pip install -e ../tts-engine-common
python server_chatterbox.py          # or: uvicorn server_chatterbox:app
```

Engine packages: `chatterbox-tts`, `omnivoice`, `qwen-tts`, `dots.tts`.
The full parameter list for each server is at its `GET /capabilities` —
don't trust this README over that endpoint, the schema is the source of truth.

## Engine-specific notes

- **Chatterbox** — no `reference_text` field: it conditions purely on the
  reference audio. Output is PerTh-watermarked by the library itself. Only the
  first 10 s of the reference clip are used.
- **OmniVoice** — omitting `reference_text` triggers an on-the-fly Whisper
  transcription; the first such request pays the ASR model load. References
  over 20 s are trimmed at the largest silence gap.
- **Qwen3-TTS** — `language` takes lowercase *names* (`english`, `chinese`,
  ...) or `auto`. Omitting `reference_text` transparently enables
  speaker-embedding-only mode (`x_vector_only_mode`), since the engine's
  in-context mode hard-requires a transcript.
- **dots.tts** — 48 kHz output (unlike the 24 kHz engines). The runtime
  demands a file path for the prompt audio, so the server writes a temp file;
  the reference transcript is optional.

## Tests (`tests/`)

```bash
python -m pytest tests/
```

The suite runs **without torch, NumPy, soundfile, or any engine installed** —
`tests/stubs/` provides import-only stand-ins so the servers can be imported
and their HTTP surface exercised. Stubs are appended to `sys.path` *last*, so
on a machine with the real packages installed (e.g. a GPU box) the real ones
always win and the tests run against them.

What is covered:

- `GET /capabilities` — full-document snapshot per engine
  (`tests/snapshots/*.json`) plus targeted assertions (core fields, enums,
  sample rates).
- `GET /health` and the `GET /` landing page.
- `POST /synthesize` request-body validation: `422` on unknown fields,
  missing/empty fields, out-of-range values, and bad enum members.
- `POST /synthesize` reference-audio pre-flight: `400` on undecodable audio
  and clips shorter than the engine's minimum (the stub `soundfile.info()`
  parses real WAV headers via the stdlib `wave` module).

What is *not* covered: actual synthesis (needs a real model + GPU) and the
success path of `/synthesize`.

### Snapshots

Regenerate after changing a server's request schema (or engine constants):

```bash
python tests/update_snapshots.py
```

Then review the diff. The dots.tts snapshot records the `device` observed on
the generating machine; its test compares everything except `device`
(machine-dependent: CUDA if available, else CPU) and only asserts the value
is `cuda` or `cpu`.

### Gotchas

- The environment is pinned to the servers' documented defaults in
  `tests/_bootstrap.py` (shared with `update_snapshots.py`) so snapshots are
  deterministic regardless of ambient `*_DEVICE` / `*_MODEL` variables.
- Pydantic v2's lax validation coerces `"yes"`/`"true"`/`"1"` to booleans —
  the boolean-rejection tests use values that are *not* coercible.
- The stub `numpy` module must satisfy pytest's own introspection
  (`isscalar`, `bool_`, `ndarray`, `asarray`); see the comment in
  `stubs/numpy/__init__.py` if cross-suite runs start failing with
  `AttributeError: module 'numpy' has no attribute ...`.
