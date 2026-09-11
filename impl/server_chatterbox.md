# Chatterbox

Quick stats:

- **Server script**: [server_chatterbox.py](server_chatterbox.py)
- **Sample rate**: 24 kHz
- **Notes**: 
  - 23 supported language codes
  - output is PerTh-watermarked by the library
  - reference audio transcript not needed! Chatterbox conditions purely on the reference audio.
  - only the first 10s of the reference audio are used.

## Installation

Start by setting up a venv (or use your conda setup):

```
mkdir Chatterbox
cd Chatterbox
python3 -m venv .venv
source .venv/bin/activate
```

Now install Chatterbox in this environment:

```
pip install chatterbox-tts
```

Now clone `tts-serve` and install its dependencies:

```
git clone https://github.com/scorbo2/tts-serve
cd tts-serve
pip install ./tts-engine-common fastapi uvicorn loguru soundfile
```

Start it up!

```
python impl/server_chatterbox.py
```

### Changing host

By default, `0.0.0.0` is used. To force a local-only server:

```
export CHATTERBOX_HOST=127.0.0.1
python impl/server_chatterbox.py
```

### Changing port

By default, `7500` is used. To choose a different port:

```
export CHATTERBOX_PORT=8500
python impl/server_chatterbox.py
```

### To use a local model path instead of huggingface

By default, the server script will download the needed model from huggingface on first run.
After that, the model should exist in your local huggingface cache dir (`~/.cache/huggingface/hub/`).
If you have downloaded the model yourself and want to force a local path:

```
export CHATTERBOX_T3_MODEL=/path/to/model.safetensors
python impl/server_chatterbox.py
```

### Run on a different device

By default, Chatterbox will run on `cuda`. To force a different device:

```
export CHATTERBOX_DEVICE=cpu
python impl/server_chatterbox.py

export CHATTERBOX_DEVICE=mps
python impl/server_chatterbox.py
```


