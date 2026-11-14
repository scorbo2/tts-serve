# KittenTTS 2

Quick stats:

- **Server script**: [server_kittenTTS.py](server_kittenTTS.py)
- **Sample rate**: 24 kHz
- **Device**: `KITTENTTS_DEVICE` — `auto` (default: CUDA when available,
  else CPU), `cuda`, `mps` or `cpu`. Budget roughly 8 GB of free VRAM for
  the CUDA path (engine figure), or ~6 GB of system RAM for CPU inference.
  A 4 GB laptop card will **not** hold this model on CUDA — use `cpu`.
  - `cuda:1` is **not** accepted. The engine compares the device to the
    literal string `'cuda'` (`kittenml/kittentts2/speaker.py::_choose_device`
    starts `if self.device != "cuda": return "cpu"`), so an indexed value
    wouldn't select GPU 1 — it would quietly run the transcription/speaker
    models on the CPU. To pick which GPU the server gets, pin it upstream:
    `CUDA_VISIBLE_DEVICES=1 KITTENTTS_DEVICE=cuda`. The value itself is
    checked at server start, and an explicit `cuda` on a box without CUDA is
    refused before the ~1.7B checkpoint is loaded (`auto` is never refused —
    "no GPU" is a valid answer for it).
- **Notes**:
  - KittenTTS 2 is a 1.7B-parameter **1-bit speech language model** with
    in-context voice cloning. The Python backend is the reference
    implementation, not a speed-optimized one (engine docs). Measured on a
    laptop CPU (RTX 3050 4 GB box, too small for the CUDA path): RTF ≈ 8.5–9,
    so treat CPU as "works for testing", not a production voice. The engine
    project points at a llama.cpp-based fork when you need real-time CPU.
  - two ways to pick a **voice**: `voice` names one of the model's **115
    built-in voices** (default `Bruno`; several are non-English, named after
    the language — `German`, `Arabic`, ...), or `audio_base64` zero-shot
    clones a reference clip (5–30 s of one speaker). `voice` and
    `audio_base64` **together are a 422** — the engine treats them as
    mutually exclusive, so the server rejects the combination instead of
    silently picking one. Neither one means the checkpoint's default voice.
    `reference_audio` is therefore **optional** in capabilities.
  - `reference_text` is optional for cloning: without it the engine
    transcribes the clip with Whisper on the fly (a `whisper-large-v3`
    download and load on the first such request). Gotcha: that auto-pass
    runs in **English mode** — for a non-English reference clip, send the
    transcript yourself. `reference_text` without `audio_base64` is a 422.
  - unknown `voice`/`preset` names are a runtime `422` with the model's
    count and a sample: the voice table is checkpoint data (115 names for
    `kitten-tts-2`), unknown at import time, so it is not an enum in
    capabilities.
  - **expression control** travels inside `text`: a leading `[emotion]` tag,
    inline `<event>` tags and `(((emphasis)))` spans. The engine calls this
    area early beta — it steers delivery, it does not guarantee it. The
    `preset` field (`stable` default, `expressive`) and the raw sampler
    knobs (`temperature`, `top_p`, `top_k`, `min_p`) tune the same area;
    `max_new_tokens` caps generation length; `normalize` (default on) runs
    text normalization before synthesis.
  - **language**: no `language` parameter exists in the engine — the
    language is carried by the speaker/voice. The server accepts any
    well-formed two-letter code for API consistency and does not forward
    it, so `/capabilities` advertises `languages: null` (the docs/02
    no-support case, same as LuxTTS/VoxCPM/Breeze/Pocket).
  - `seed` is meaningful: the same seed, text and speaker reproduce each
    other. Bit-exact on CPU (verified live); on CUDA some kernels are
    non-deterministic, so expect near-exact there.
  - reference clips are staged as content-hashed (SHA-256) temp files that
    are **kept** — the engine caches the encoded clip and speaker embedding
    per path, so per-request deletion would throw the cache away (and race
    a queued request for the same clip). Wipe the staging directory
    (under `/tmp/kittentts_rest_api/`) to reclaim space.
  - synthesis is serialized with a lock: the model mutates a shared
    pending-speaker-embedding between `generate()` calls.
  - not watermarked (`watermarked: false` in capabilities).
  - this server speaks the **KittenTTS 2** API only. The same `kittenml`
    package can also load the legacy 0.8 ONNX models through the same
    constructor, but they take a different `generate()` signature; pointing
    `KITTENTTS_MODEL` at an ONNX repository fails fast at startup.

## Installation

Start by setting up a venv (or use your conda setup):

```
mkdir KittenTTS
cd KittenTTS
python3 -m venv .venv
source .venv/bin/activate
```

Install the engine. On Linux, if you have no GPU (or want the CPU build),
point pip at the CPU-only PyTorch index so it does not pull several GB of
CUDA wheels:

```
pip install kittenml --extra-index-url https://download.pytorch.org/whl/cpu
```

Clone `tts-serve` and install its shared layer plus the FastAPI server deps:

```
git clone https://github.com/scorbo2/tts-serve
cd tts-serve
pip install ./tts-engine-common fastapi uvicorn loguru soundfile
```

Start it up!

```
python impl/server_kittenTTS.py
```

Model weights are downloaded from HuggingFace on first start
(`KittenML/kitten-tts-2`, plus the s3gen vocoder weights from
`ResembleAI/chatterbox-turbo`).

### Alternative: from a local KittenTTS git checkout (uv)

The PyPI release currently matches git, so `pip install kittenml` works.
If you'd rather run from a checkout (a `git pull` in the engine takes
effect on the next server start), there is a ready-made uv environment:

```
git clone https://github.com/KittenML/KittenTTS   # next to tts-serve
python3 tools/serve.py kittentts
```

That syncs `envs/kittentts/` only when its inputs changed, then starts the
server. `--host`/`--port` override the bind address for that run. If your
engine checkout lives elsewhere, edit the `kittenml` path in
`envs/kittentts/pyproject.toml`. See
[`docs/04-engine-environments.md`](../docs/04-engine-environments.md).

### Changing host

By default, `0.0.0.0` is used. To force a local-only server:

```
export KITTENTTS_HOST=127.0.0.1
python impl/server_kittenTTS.py
```

### Changing port

By default, `7500` is used. To choose a different port:

```
export KITTENTTS_PORT=8500
python impl/server_kittenTTS.py
```

### Choosing the model

Default is `KittenML/kitten-tts-2`. `KITTENTTS_MODEL` also accepts a local
directory (e.g. a `huggingface-cli download KittenML/kitten-tts-2
--local-dir ...` result) — but **only** a KittenTTS 2 repository; a legacy
ONNX model directory fails at startup, by design.

```
export KITTENTTS_MODEL=/srv/models/kitten-tts-2
python impl/server_kittenTTS.py
```

### Shrinking the footprint

Two knobs trade memory for a quality hit (engine docs: `docs/decoders.md`
and the weight-pack guide):

```
export KITTENTTS_DECODER=student_w4   # distilled vocoder flow, saves ~420 MB
export KITTENTTS_WEIGHTS=emb4         # quantised embedding, half the download
python impl/server_kittenTTS.py
```

Unset (the default) keeps the repository settings: stock flow + `packed`
weights.

## Talking to it

Voice cloning via `speak.py` (reference clip + transcript; the transcript
also skips the Whisper auto-pass):

```
python3 tools/speak.py "Hello there, general Ken." \
  --server http://localhost:7500 \
  --ref-audio my_voice.wav --ref-audio-transcript my_voice.txt \
  --output-file out.wav
```

Built-in voices are a plain `curl` — `speak.py` always requires a reference
clip (per its spec), so the no-clip modes (built-in `voice`, or neither for
the checkpoint default) are reachable via the raw endpoint:

```
curl -s http://localhost:7500/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text": "[excited] Hello! This is a KittenTTS 2 test.", "voice": "Bruno", "seed": 7}'
```

Check what the server accepts at any time with
`GET /capabilities` — the schema is the source of truth, not this document.
