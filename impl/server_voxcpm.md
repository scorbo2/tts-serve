# VoxCPM

Quick stats:

- **Server script**: [server_voxcpm.py](server_voxcpm.py)
- **Sample rate**: 48 kHz
- **HuggingFace checkpoint**: `openbmb/VoxCPM2` (2B parameters)
- **Notes**:
  - VoxCPM2 supports **three modes**:
    1. **Voice Design** — no reference audio needed; describe the desired voice
       in parentheses at the start of `text` (e.g. `"(A young woman, gentle voice)Hello there."`).
    2. **Controllable Cloning** — provide reference audio only; the model clones
       the timbre.
    3. **Ultimate Cloning** — provide reference audio *and* its exact transcript;
       the model treats the reference as a spoken prefix and continues from it,
       faithfully reproducing every vocal detail.
  - the engine **auto-detects language** from text content (30 languages
    supported internally). The API accepts any two-letter `language` code for
    consistency but does not forward it.
  - the engine requires a **file path** for reference audio, so the server
    stages the base64 clip to a temp file (UUID-named, deleted per-request).
  - `seed` is meaningful — the engine applies it via `torch.manual_seed()`
    before generation.
  - `load_denoiser=False` by default (the acoustic noise suppression model
    adds overhead without improving synthesis quality).
  - the engine uses `torch.compile` (`optimize=True`) by default for better
    throughput.
  - the engine's `resolve_runtime_device()` resolves `None`/`"auto"` to
    `cuda` → `mps` → `cpu` in that order.

## Installation

```
git clone https://github.com/OpenBMB/VoxCPM
cd VoxCPM
pip install .
```

VoxCPM requires Python ≥ 3.10 (< 3.13), PyTorch ≥ 2.5.0, and CUDA ≥ 12.0.

Now install tts-serve dependencies:

```
pip install fastapi uvicorn loguru soundfile
pip install ./tts-engine-common
```

Start it up!

```
python impl/server_voxcpm.py
```

### Changing host

```
export VOXCPM_HOST=127.0.0.1
python impl/server_voxcpm.py
```

### Changing port

```
export VOXCPM_PORT=8500
python impl/server_voxcpm.py
```

### To use a local model path instead of huggingface

```
export VOXCPM_MODEL=/path/to/VoxCPM2/
python impl/server_voxcpm.py
```

### Run on a different device

By default, the server resolves to `cuda`. To force a different device:

```
export VOXCPM_DEVICE=cpu
python impl/server_voxcpm.py
```

Supported values: `auto`, `cpu`, `mps`, `cuda`, `cuda:0`, etc.
