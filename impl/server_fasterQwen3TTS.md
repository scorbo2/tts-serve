# Faster Qwen3-TTS

Quick stats:

- **Server script**: [server_fasterQwen3TTS.py](server_fasterQwen3TTS.py)
- **Sample rate**: 24 kHz
- **Notes**:
  - CUDA-only fork of Qwen3-TTS
  - `xvec_only` (default `true`): x-vector mode — the reference audio supplies a
    speaker embedding, giving a consistent voice across requests; `reference_text`
    is ignored.
  - `xvec_only=false`: ICL mode — `reference_text` is **required** (422 without
    it); there is no automatic fallback to x-vector mode.
  - The server default (x-vector) deliberately differs from the engine's library
    default (ICL): in our testing the x-vector voice is the more consistent one,
    and that is what sentence-by-sentence streaming wants.

## Installation

Start by setting up a venv (or use your conda setup):

```
mkdir faster-Qwen3TTS
cd faster-Qwen3TTS
python3 -m venv .venv
source .venv/bin/activate
```

Now install Faster Qwen3-TTS in this environment:

```
pip install faster-qwen3-tts
```

Now clone `tts-serve` and install its dependencies:

```
git clone https://github.com/scorbo2/tts-serve
cd tts-serve
pip install ./tts-engine-common fastapi uvicorn loguru soundfile
```

Start it up!

```
python impl/server_fasterQwen3TTS.py
```

### Changing host

By default, `0.0.0.0` is used. To force a local-only server:

```
export FASTER_QWEN3TTS_HOST=127.0.0.1
python impl/server_fasterQwen3TTS.py
```

### Changing port

By default, `7500` is used. To choose a different port:

```
export FASTER_QWEN3TTS_PORT=8500
python impl/server_fasterQwen3TTS.py
```

### To use a local model path instead of huggingface

By default, the server script will download the needed model from huggingface on first run.
After that, the model should exist in your local huggingface cache dir (`~/.cache/huggingface/hub/`).
If you have downloaded the model yourself and want to force a local path:

```
export FASTER_QWEN3TTS_MODEL=/path/to/model/
python impl/server_fasterQwen3TTS.py
```

### Run on a different device

This engine only supports `cuda`! There is no CPU or MPS backend, so
`FASTER_QWEN3TTS_DEVICE` accepts `cuda` and an indexed `cuda:<index>` —
anything else is a startup error naming the variable, not a silent fallback:

```
export FASTER_QWEN3TTS_DEVICE=cuda:1
python impl/server_fasterQwen3TTS.py
```

An index this box does not have is refused when the model loads, before the
checkpoint is read (and before the CUDA-graph capture, which is the expensive
part of startup on this engine).



