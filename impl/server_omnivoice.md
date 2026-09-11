# OmniVoice

Quick stats:

- **Server script**: [server_omnivoice.py](server_omnivoice.py)
- **Sample rate**: 24 kHz
- **Notes**: 
  - reference audio transcript is optional! If omitted, whisper will be used (incurs additional overhead)

## Installation

Start by setting up a venv (or use your conda setup):

```
mkdir OmniVoice
cd OmniVoice
python3 -m venv .venv
source .venv/bin/activate
```

Now install OmniVoice in this environment:

```
pip install omnivoice
```

Now clone `tts-serve` and install its dependencies:

```
git clone https://github.com/scorbo2/tts-serve
cd tts-serve
pip install ./tts-engine-common fastapi uvicorn loguru soundfile
```

Start it up!

```
python impl/server_omnivoice.py
```

### Changing host

By default, `0.0.0.0` is used. To force a local-only server:

```
export OMNIVOICE_HOST=127.0.0.1
python impl/server_omnivoice.py
```

### Changing port

By default, `7500` is used. To choose a different port:

```
export OMNIVOICE_PORT=8500
python impl/server_omnivoice.py
```

### To use a local model path instead of huggingface

By default, the server script will download the needed model from huggingface on first run.
After that, the model should exist in your local huggingface cache dir (`~/.cache/huggingface/hub/`).
If you have downloaded the model yourself and want to force a local path:

```
export OMNIVOICE_MODEL=/path/to/model/
python impl/server_omnivoice.py
```

### Run on a different device

By default, OmniVoice will run on `cuda`. To force a different device:

```
OMNIVOICE_DEVICE=cpu
python impl/server_omnivoice.py

OMNIVOICE_DEVICE=mps
python impl/server_omnivoice.py
```


