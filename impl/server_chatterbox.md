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

Now install Chatterbox in this environment.

> **Note:** the latest PyPI release (`chatterbox-tts` 0.1.7) predates the
> v3 multilingual API this server requires, so install from the pinned git
> commit below until upstream ships a newer release. (`pip show` cannot tell
> the two apart -- both report 0.1.7. If the server starts up and then dies
> with an `ImportError` naming `MULTILINGUAL_T3_MODELS`, you have the PyPI
> build; reinstall from git.)

```
pip install "git+https://github.com/resemble-ai/chatterbox.git@5de7a54aa4e5e2baadb0182dde554908b48b85c2"
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

### Alternative: from a local Chatterbox git checkout (uv)

[`uv`](https://docs.astral.sh/uv/) can build a venv dedicated to this server
that installs Chatterbox straight from a git checkout. Nothing is installed
globally. Clone Chatterbox next to tts-serve:

```
git clone https://github.com/resemble-ai/chatterbox
git clone https://github.com/scorbo2/tts-serve
cd tts-serve
python3 tools/serve.py chatterbox
```

The launcher (`tools/serve.py`, stdlib-only, Python 3.11+) builds
`envs/chatterbox/.venv` from
[`envs/chatterbox/pyproject.toml`](../envs/chatterbox/pyproject.toml) on first
use, then starts the server. On later starts it re-runs `uv sync` only when
the env's pyproject, its `.python-version`, or the Chatterbox checkout's
`pyproject.toml` has changed. Use `--sync` to force a sync, `--port N` to run
on a port other than 7500 (it sets `CHATTERBOX_PORT`), `--host 127.0.0.1` for a
local-only server (it sets `CHATTERBOX_HOST`), and `--list` to see each env's
status. By hand, it's `uv sync --project envs/chatterbox`, then
`envs/chatterbox/.venv/bin/python impl/server_chatterbox.py`.

- Chatterbox and `tts-engine-common` are installed editable, so after a
  `git pull` in either one you only need to restart the server.
- If your Chatterbox checkout isn't next to tts-serve, edit the `chatterbox-tts`
  path under `[tool.uv.sources]` in `envs/chatterbox/pyproject.toml`.
- The environment variables below (`CHATTERBOX_PORT`, `CHATTERBOX_DEVICE`, ...)
  work unchanged.
- Torch: Chatterbox pins torch 2.6.0. That pin is kept on x86_64 Linux, where
  PyPI's wheels include CUDA, and on macOS. Linux on aarch64 (e.g. NVIDIA GB10 /
  DGX Spark, Jetson Thor, GH200) has no CUDA build of 2.6.0, so there it gets
  CUDA torch 2.9.1 from the cu130 index instead.
- Besides the model weights, the first start also downloads a small
  spaCy/pkuseg segmentation model to `~/.pkuseg/`. The engine does this itself.
- The design behind `envs/` is in
  [`docs/04-engine-environments.md`](../docs/04-engine-environments.md).

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


