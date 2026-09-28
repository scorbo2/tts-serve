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

### Alternative: from a local OmniVoice git checkout (uv)

If you already have an [OmniVoice](https://github.com/k2-fsa/OmniVoice) git
checkout, or want to track its latest code, [`uv`](https://docs.astral.sh/uv/)
can build a venv dedicated to this server that installs OmniVoice straight
from the checkout. Nothing is installed globally, and OmniVoice's own venv is
left alone. Clone OmniVoice next to tts-serve:

```
git clone https://github.com/k2-fsa/OmniVoice
git clone https://github.com/scorbo2/tts-serve
cd tts-serve
python3 tools/serve.py omnivoice
```

The launcher (`tools/serve.py`, stdlib-only, Python 3.11+) builds
`envs/omnivoice/.venv` from
[`envs/omnivoice/pyproject.toml`](../envs/omnivoice/pyproject.toml) on first
use, then starts the server. On later starts it re-runs `uv sync` only when
the env's pyproject, its `.python-version`, or the OmniVoice checkout's
`pyproject.toml` has changed. Use `--sync` to force a sync, `--port N` to run
on a port other than 7500 (it sets `OMNIVOICE_PORT`), `--host 127.0.0.1` for a
local-only server (it sets `OMNIVOICE_HOST`), and `--list` to see each env's
status.

Without the launcher, the same steps by hand are:

```
uv sync --project envs/omnivoice
uv run --project envs/omnivoice python impl/server_omnivoice.py
```

`uv run` re-checks the environment on every start, which may need network
access. To skip that (e.g. offline), run the venv's Python directly:
`envs/omnivoice/.venv/bin/python impl/server_omnivoice.py`. The launcher does
this for you.

- OmniVoice and `tts-engine-common` are installed editable, so after a
  `git pull` in either one you only need to restart the server.
- If your OmniVoice checkout isn't next to tts-serve, edit the `omnivoice` path
  under `[tool.uv.sources]` in `envs/omnivoice/pyproject.toml`.
- The environment variables below (`OMNIVOICE_PORT`, `OMNIVOICE_DEVICE`, ...)
  work unchanged.
- Linux on aarch64 (e.g. NVIDIA GB10 / DGX Spark, Jetson Thor, GH200) gets CUDA
  torch 2.9.1 from the cu130 index, because there are no aarch64 cu128 wheels.
  Other Linux and Windows machines get torch 2.8.0 from cu128, the same as
  OmniVoice's own setup. This works with an unmodified OmniVoice checkout.
- The design behind `envs/` is in
  [`docs/04-engine-environments.md`](../docs/04-engine-environments.md).

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
export OMNIVOICE_DEVICE=cpu
python impl/server_omnivoice.py

export OMNIVOICE_DEVICE=mps
python impl/server_omnivoice.py
```


