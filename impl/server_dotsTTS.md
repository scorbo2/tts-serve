# dots.tts

Quick stats:

- **Server script**: [server_dotsTTS.py](server_dotsTTS.py)
- **Sample rate**: 48 kHz
- **Notes**:
  - reference audio transcript is optional.

## Installation

Start by setting up a venv (or use your conda setup):

```
mkdir dots.tts
cd dots.tts
python3 -m venv .venv
source .venv/bin/activate
```

Now install dots.tts in this environment:

```
pip install dots.tts
```

Now clone `tts-serve` and install its dependencies:

```
git clone https://github.com/scorbo2/tts-serve
cd tts-serve
pip install ./tts-engine-common fastapi uvicorn loguru soundfile
```

Start it up!

```
python impl/server_dotsTTS.py
```

### Changing host

By default, `0.0.0.0` is used. To force a local-only server:

```
export DOTS_TTS_HOST=127.0.0.1
python impl/server_dotsTTS.py
```

### Changing port

By default, `7500` is used. To choose a different port:

```
export DOTS_TTS_PORT=8500
python impl/server_dotsTTS.py
```

### To use a local model path instead of huggingface

By default, the server script will download the needed model from huggingface on first run.
After that, the model should exist in your local huggingface cache dir (`~/.cache/huggingface/hub/`).
If you have downloaded the model yourself and want to force a local path:

```
export DOTS_TTS_MODEL=/path/to/model/
python impl/server_dotsTTS.py
```

### Run on a different device

This runtime will automatically select `cuda` if available, with `cpu` as a fallback.
Currently, this behavior cannot be overridden.


