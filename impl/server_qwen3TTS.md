# Qwen3-TTS

Quick stats:

- **Server script**: [server_qwen3TTS.py](server_qwen3TTS.py)
- **Sample rate**: 24 kHz
- **Notes**:
  - reference audio transcript is optional; omitting it falls back to speaker-embedding-only mode.

## Installation

Start by setting up a venv (or use your conda setup):

```
mkdir Qwen3TTS
cd Qwen3TTS
python3 -m venv .venv
source .venv/bin/activate
```

Now install Qwen3-TTS in this environment:

```
pip install -U qwen-tts
```

Now clone `tts-serve` and install its dependencies:

```
git clone https://github.com/scorbo2/tts-serve
cd tts-serve
pip install ./tts-engine-common fastapi uvicorn loguru soundfile
```

Start it up!

```
python impl/server_qwen3TTS.py
```

### Changing host

By default, `0.0.0.0` is used. To force a local-only server:

```
export QWEN3TTS_HOST=127.0.0.1
python impl/server_qwen3TTS.py
```

### Changing port

By default, `7500` is used. To choose a different port:

```
export QWEN3TTS_PORT=8500
python impl/server_qwen3TTS.py
```

### To use a local model path instead of huggingface

By default, the server script will download the needed model from huggingface on first run.
After that, the model should exist in your local huggingface cache dir (`~/.cache/huggingface/hub/`).
If you have downloaded the model yourself and want to force a local path:

```
export QWEN3TTS_MODEL=/path/to/model/
python impl/server_qwen3TTS.py
```

### Run on a different device

By default, Qwen3-TTS will run on `cuda`. To force a different device:

```
export QWEN3TTS_DEVICE=cpu
python impl/server_qwen3TTS.py

export QWEN3TTS_DEVICE=mps
python impl/server_qwen3TTS.py
```


