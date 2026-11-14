# VoxCPM

Quick stats:

- **Server script**: [server_voxcpm.py](server_voxcpm.py)
- **Sample rate**: 48 kHz
- **Notes**:
  - three synthesis modes via the same `generate()` call:
    - **Voice design** -- no reference audio needed; steer the generated voice
      with a `(control instruction)` prefix in `text`, e.g.
      "`(A young woman, gentle voice)Hello there!`".
    - **Controllable cloning** -- provide a roughly 3--10 s reference clip
      (base64); clone the timbre while style control still goes via a `(control)`
      prefix in `text`.
    - **Ultimate cloning** -- provide the reference clip plus its exact
      transcript (`reference_text`); the model treats the clip as a spoken
      prefix and continues from it, reproducing every vocal nuance. The engine
      loads the reference at 16 kHz (librosa) and conditions on its latent
      patches.
  - `reference_text` is only accepted together with `audio_base64`; a
    transcript with no clip is rejected with a `422` (it is a client error,
    not a mode switch).
  - `reference_audio` is **optional** in capabilities (voice design mode needs
    none), unlike most tts-serve engines where it is required.
  - the engine has **no language parameter**: its text encoder auto-detects the
    language from the input text (30 languages supported). The API accepts any
    two-letter `language` code for consistency but does not forward it.
  - `seed` is meaningful -- PyTorch's RNG is seeded before each generation. The
    same seed and inputs give bit-identical audio on CPU; on CUDA the audio is
    near-identical but not bit-identical (residual from non-deterministic GPU
    kernels).
  - reference clips must be at least 2 s long for stable cloning (checked at
    the request boundary with a 400).
  - synthesis is **serialized** (single shared model): the engine mutates KV
    caches in place, its `torch.compile`d functions are not re-entrant, and its
    `@torch.inference_mode()` contexts are thread-affine.

## Installation

Start by setting up a venv (or use your conda setup):

```
mkdir VoxCPM
cd VoxCPM
python3 -m venv .venv
source .venv/bin/activate
```

Now install VoxCPM in this environment:

```
pip install voxcpm
```

This pulls in PyTorch (CUDA ≥ 12.0 recommended), transformers, librosa,
soundfile, and the rest of the engine's dependencies.

Clone `tts-serve` and install its shared layer plus the FastAPI server deps:

```
git clone https://github.com/scorbo2/tts-serve
cd tts-serve
pip install ./tts-engine-common fastapi uvicorn loguru soundfile
```

Start it up!

```
python impl/server_voxcpm.py
```

### Changing host

By default, `0.0.0.0` is used. To force a local-only server:

```
export VOXCPM_HOST=127.0.0.1
python impl/server_voxcpm.py
```

### Changing port

By default, `7500` is used. To choose a different port:

```
export VOXCPM_PORT=8500
python impl/server_voxcpm.py
```

### To use a local model path instead of huggingface

By default, the server script will download the needed model from huggingface
(`openbmb/VoxCPM2`) on first run. After that, the model should exist in your
local huggingface cache dir (`~/.cache/huggingface/hub/`).
If you have downloaded the model yourself and want to force a local path:

```
export VOXCPM_MODEL=/path/to/VoxCPM2/
python impl/server_voxcpm.py
```

Note: cloning (any use of `reference_audio`) needs a VoxCPM2 checkpoint -- the
engine rejects reference audio on VoxCPM1 models. VoxCPM1 checkpoints also
output 16 kHz instead of 48 kHz; the response's `sample_rate` is read from the
loaded model, so it will report the true rate either way.

### Run on a different device

By default, VoxCPM will run on `cuda`. To force a different device:

```
export VOXCPM_DEVICE=cpu
python impl/server_voxcpm.py
```

`auto`, `cuda`, `cuda:<index>`, `mps`, and `cpu` are accepted — the same
grammar the engine documents, checked by the server at launch so a typo never
reaches the loader. `auto` lets the engine pick (CUDA preferred, then MPS, then
CPU); an explicit CUDA request this box cannot serve — `cuda` with no GPU, or
`cuda:9` on a one-GPU box — is refused at model load with a message naming
`VOXCPM_DEVICE`, the ordinal, and the GPU count, before the checkpoint is read.
On MPS the engine forces `float32` by default (bfloat16/float16 cause numerical
drift that breaks the diffusion loop) -- see the `VOXCPM_MPS_DTYPE` note below if
you want to test a different dtype.

### MPS dtype override

On Apple Silicon, bfloat16 or float16 produce numerical drift in the diffusion
loop, causing glitched output and infinite badcase retries. The engine forces
`float32` on MPS by default. If you want to test whether future engine versions
handle mixed precision better, you can override it:

```
export VOXCPM_MPS_DTYPE=bfloat16
python impl/server_voxcpm.py
```

Valid values: `float32`, `bfloat16`, `float16`. Leave unset in production.

### Reference audio preparation

For cloning modes, provide a clean, roughly 3--10 second clip of the target
voice (speech, not music). The engine resamples it to 16 kHz internally. Clips
shorter than 2 s are rejected with a `400` at the request boundary.

### Text normalization

The `normalize` flag runs `wetext` text normalization before generation
(disabled by default). Enable it if you hear issues with numbers,
abbreviations, or special characters in the input text. It is exposed as an
advanced parameter (not shown in the default UI) -- send it explicitly if needed.
