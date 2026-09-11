# Index-TTS

Quick stats:

- **Server script**: [server_indexTTS.py](server_indexTTS.py)
- **Sample rate**: 22.05 kHz
- **Notes**:
  - reference audio transcript not used. Conditions on the reference audio alone.
  - only the first 15s of reference audio are used.
  - `seed` is best-effort (the engine exposes no seed parameter)

## Installation

Index-TTS has no `pip install`. Start by cloning the GitHub repo:

```
git clone https://github.com/index-tts/index-tts.git
cd index-tts
```

Now use `uv` to install all dependencies (this creates a venv):

```
uv sync --all-extras
```

Now clone `tts-serve` here inside the index-tts directory, and install its dependencies:

```
source .venv/bin/activate
git clone https://github.com/scorbo2/tts-serve && cd tts-serve
pip install ./tts-engine-common fastapi uvicorn loguru soundfile
```

Start it up!

```
python impl/server_indexTTS.py
```

### Changing host

By default, `0.0.0.0` is used. To force a local-only server:

```
export INDEXTTS_HOST=127.0.0.1
python impl/server_indexTTS.py
```

### Changing port

By default, `7500` is used. To choose a different port:

```
export INDEXTTS_PORT=8500
python impl/server_indexTTS.py
```

### Enable the `emotion_text` parameter

`emotion_text` is an experimental feature and is disabled by default. To enable it:

```
export INDEXTTS_USE_QWEN_EMO=1
python impl/server_indexTTS.py
```

### Reduce VRAM usage

This engine supports an option (disabled by default) to lower VRAM usage. To enable:

```
export INDEXTTS_USE_BF16=1
python impl/server_indexTTS.py
```

### To use a local model path instead of huggingface

On first run, Index-TTS downloads from huggingface into `INDEXTTS_MODEL_DIR` (default `checkpoints/`).
You can explicitly point to any local directory containing the needed model files to avoid
an internet connection:

```
export INDEXTTS_MODEL_DIR=/path/to/models/
python impl/server_indexTTS.py
```

### Run on a different device

By default the device is auto-selected (CUDA if available, otherwise XPU, MPS, or CPU). To force a specific device:

```
export INDEXTTS_DEVICE=cuda
python impl/server_indexTTS.py

export INDEXTTS_DEVICE=cpu
python impl/server_indexTTS.py

export INDEXTTS_DEVICE=xpu
python impl/server_indexTTS.py

export INDEXTTS_DEVICE=mps
python impl/server_indexTTS.py
```


