# LuxTTS

Quick stats:

- **Server script**: [server_luxTTS.py](server_luxTTS.py)
- **Sample rate**: 48 kHz
- **Notes**:
  - reference audio transcript **not needed** — the engine always transcribes
    the reference clip itself with Whisper (openai/whisper-base on GPU,
    whisper-tiny on CPU), so there is no `reference_text` field and every
    request pays the ASR cost.
  - the engine has **no language parameter**: its tokenizer auto-detects
    English and Chinese per text segment (other scripts are dropped). The
    API accepts any two-letter `language` code for consistency but does not
    forward it.
  - `seed` is meaningful — the flow-matching solver's initial noise comes
    from the PyTorch RNG. The same seed and inputs give bit-identical audio
    on CPU; on CUDA the audio is near-identical but not bit-identical
    (residual ~1e-7 sample differences from non-deterministic GPU kernels).
  - the reference clip must yield an English or Chinese transcription: the
    engine conditions on Whisper's output, and a clip that produces none
    (e.g. pure silence, music, or speech in other languages) currently
    fails with HTTP 500 rather than a friendly 400.
  - the first request after startup also pays a one-time librosa
    initialisation (~10 s).

## Installation

Start by setting up a venv (or use your conda setup):

```
mkdir LuxTTS
cd LuxTTS
python3 -m venv .venv
source .venv/bin/activate
```

Now install LuxTTS in this environment. It is a git repo, not a PyPI package:

```
git clone https://github.com/ysharma3501/LuxTTS.git
cd LuxTTS
pip install -r requirements.txt
cd ..
```

Now clone `tts-serve` and install its dependencies:

```
git clone https://github.com/scorbo2/tts-serve
cd tts-serve
pip install ./tts-engine-common fastapi uvicorn loguru soundfile
```

Start it up!

```
python impl/server_luxTTS.py
```

### Changing host

By default, `0.0.0.0` is used. To force a local-only server:

```
export LUX_TTS_HOST=127.0.0.1
python impl/server_luxTTS.py
```

### Changing port

By default, `7500` is used. To choose a different port:

```
export LUX_TTS_PORT=8500
python impl/server_luxTTS.py
```

### To use a local model path instead of huggingface

By default, the server script will download the needed model from huggingface
(YatharthS/LuxTTS) on first run. After that, the model should exist in your
local huggingface cache dir (`~/.cache/huggingface/hub/`).
If you have downloaded the model yourself and want to force a local path:

```
export LUX_TTS_MODEL=/path/to/LuxTTS/
python impl/server_luxTTS.py
```

Two upstream engine quirks to be aware of (neither is a tts-serve bug):

- On the **GPU** path, the engine treats *any* non-default `LUX_TTS_MODEL`
  value as a local directory — it only ever downloads the default id. So a
  non-default HuggingFace id will not be fetched; you must pass a local path
  to already-extracted model files.
- On the **CPU** path, the engine ignores `LUX_TTS_MODEL` entirely and always
  downloads the default model (an upstream bug in `zipvoice`'s CPU loader).

### Run on a different device

By default, LuxTTS will run on `cuda`. To force a different device:

```
export LUX_TTS_DEVICE=cpu
python impl/server_luxTTS.py
```

Note: if `cuda` is requested but unavailable, the engine silently falls back
to MPS, then CPU. `GET /health` (and `GET /capabilities`) report the
*configured* device (`LUX_TTS_DEVICE`), not the one actually resolved —
check the startup log ("CUDA not available, switching to MPS/CPU") to see
what the server is really running on.

### CPU thread count

The CPU path is ONNX-based and takes a thread count:

```
export LUX_TTS_THREADS=8
python impl/server_luxTTS.py
```
