# Pocket TTS

Quick stats:

- **Server script**: [server_pocketTTS.py](server_pocketTTS.py)
- **Sample rate**: 24 kHz
- **Device**: CPU (about 100M parameters; the engine does not officially
  support GPUs, so there is no device env var and this server always loads
  the model on CPU). The engine's README has a
  [running on GPU](https://github.com/kyutai-labs/pocket-tts#running-on-gpu)
  section; read it if you want to try that. Getting it working with this
  server would mean changing `DEVICE` in `server_pocketTTS.py`.
- **Notes**:
  - the lightweight engine in the set: roughly 2--6x faster than real time
    on a few CPU cores, so it suits low-latency use on a machine with no GPU.
  - the **voice** comes from, in order: `audio_base64` (a reference clip,
    cloned zero-shot), `voice` (one of the engine's preset names), or the
    loaded language's default preset. `audio_base64` and `voice` together
    are a `422`. `reference_audio` is therefore **optional** in
    capabilities.
  - **cloning needs the gated weights**: accept the terms at
    <https://huggingface.co/kyutai/pocket-tts> and run `hf auth login` on
    the server machine. Without them the server still starts and the presets
    work, but a request with `audio_base64` gets a clear `400`.
  - reference audio transcript **not needed** — there is no `reference_text`
    field (a request carrying one is a `422`). Clips are truncated to 30 s.
  - one server instance speaks **one language**, chosen at startup with
    `POCKET_TTS_MODEL` (default `english`). The API accepts any well-formed
    two-letter `language` code for consistency but does not forward it, so
    `/capabilities` advertises `languages: null` (the docs/02 no-support
    case, same as LuxTTS/VoxCPM/Breeze) — which checkpoint is loaded is
    visible in the `model` label instead. Preset voices belong to the
    language of their recordings (e.g. `daan` is Dutch).
  - community checkpoints and fully-local weights load via
    `POCKET_TTS_CONFIG` (a local YAML path, an `https://` URL, or an
    `hf://` path) — mutually exclusive with `POCKET_TTS_MODEL`. Preset
    `voice` names are unavailable with a custom config (`422`): the engine
    precomputes them per shipped language checkpoint.
  - `seed` is meaningful and reproducible: the same seed, text and voice give
    identical audio on CPU. The voice itself is fixed by the clip or preset;
    the seed varies delivery.
  - conditioned voice states are cached (LRU of 8, keyed by clip hash /
    preset name), so repeat requests with the same voice skip the
    (expensive) clip encoding.

## Installation

Start by setting up a venv (or use your conda setup):

```
mkdir PocketTTS
cd PocketTTS
python3 -m venv .venv
source .venv/bin/activate
```

Install the engine. On Linux, point pip at the CPU-only PyTorch build so it
does not pull several GB of CUDA wheels:

```
pip install pocket-tts --extra-index-url https://download.pytorch.org/whl/cpu
```

Clone `tts-serve` and install its shared layer plus the FastAPI server deps:

```
git clone https://github.com/scorbo2/tts-serve
cd tts-serve
pip install ./tts-engine-common fastapi uvicorn loguru soundfile
```

Start it up!

```
python impl/server_pocketTTS.py
```

The model weights are downloaded from huggingface on first start.

### Changing host

By default, `0.0.0.0` is used. To force a local-only server:

```
export POCKET_TTS_HOST=127.0.0.1
python impl/server_pocketTTS.py
```

### Changing port

By default, `7500` is used. To choose a different port:

```
export POCKET_TTS_PORT=8500
python impl/server_pocketTTS.py
```

### Choosing a language

By default, the English model is loaded. To serve another language, name its
engine config (`french`, `german`, `portuguese`, `italian`, `spanish`,
`dutch`, optionally with a `_24l` suffix, or an `english_*` release):

```
export POCKET_TTS_MODEL=french
python impl/server_pocketTTS.py
```

### Community models (custom config)

Community checkpoints (Czech, Hindi, Japanese, ...) are loaded via their
config URL instead — mutually exclusive with `POCKET_TTS_MODEL`:

```
export POCKET_TTS_CONFIG=hf://vvolhejn/pocket-tts-czech/czech.yaml@7b7760dd0fe994a0800f2fdbc837dc4b8f219d1c
python impl/server_pocketTTS.py
```

With a custom config the server reports `pocket-tts-custom-config` as the
`model` label — it cannot name a checkpoint it did not pick by language —
and preset voices are rejected with a `422` (they are precomputed states
tied to the released language weights).

### Fully local weights (no HuggingFace access at startup)

Yes — you can pre-download a checkpoint and run the server with **zero**
HF traffic, via the same custom-config mechanism. The engine treats any
`weights_path` that doesn't start with `hf://` or `http(s)://` as a plain
local file.

Two files per language, not one — a per-language `tokenizer.model`
(sentencepiece) alongside `model.safetensors`:

```
languages/english_2m_2026-10/model.safetensors
languages/english_2m_2026-10/tokenizer.model
```

Copy the shipped config YAML (from the installed package under
`pocket_tts/config/english_2m_2026-10.yaml`) and replace its `hf://`
lines with absolute local paths — keep everything else byte-identical,
the architecture sections are strict-loaded against the safetensors:

```yaml
weights_path: /srv/models/pocket-tts/english_2m_2026-10/model.safetensors
weights_path_without_voice_cloning: /srv/models/pocket-tts/english_2m_2026-10/model.safetensors
# ... architecture + defaults unchanged ...
    tokenizer_path: /srv/models/pocket-tts/english_2m_2026-10/tokenizer.model
```

Then:

```
export POCKET_TTS_CONFIG=/srv/models/pocket-tts/english_2m_local.yaml
python impl/server_pocketTTS.py
```

Gotchas:

- Requests with **no** clip **and** no `voice` still reach the network:
  the no-voice fallback for a custom config is Kyutai's own *alba* clip,
  fetched from HuggingFace on first use (any model can clone it, unlike
  the precomputed preset states). Send `audio_base64` to stay fully
  offline.
- With local paths the engine's licence-fallback logic never triggers
  (`has_voice_cloning` stays True — the gate detection only fires on a
  *download* failure), so point `weights_path` at the file you actually
  downloaded from the gated repo if you want real cloning quality. The
  `weights_path_without_voice_cloning` line can simply point at the same
  file when nothing is gated locally.
- `POCKET_TTS_CONFIG` and an explicit `POCKET_TTS_MODEL` are mutually
  exclusive (startup fails fast).
- `load_model(checkpoint=...)` is **not** the knob for this: it expects a
  `.pt` *training* checkpoint, not the released safetensors.
- Alternative, if you'd rather keep the plain `POCKET_TTS_MODEL` name:
  pre-populate the HF cache (`huggingface-cli download kyutai/pocket-tts
  languages/english_2m_2026-10/model.safetensors --revision <pinned-sha>`
  plus the tokenizer from the tokenizer repo path) and set
  `HF_HUB_OFFLINE=1`. The revision pins in the shipped YAML must match
  what you cached, or the offline resolution fails.

### Faster on x86 CPUs

Dynamic int8 quantization is about 27% faster on x86 with no measurable
quality loss. For best speed also `pip install "pocket-tts[quantize]"`:

```
export POCKET_TTS_QUANTIZE=1
python impl/server_pocketTTS.py
```
