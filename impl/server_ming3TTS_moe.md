# Ming 3 TTS (MoE)

The big-checkpoint counterpart of [Ming 3 TTS (0.5b dense)](server_ming3TTS.md):
[Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts)'s **7.9B-total /
1.3B-active MoE** model on Ling V3, wrapped around the engine's `ming_tiny`
stack.

> **Naming warning**: the engine calls this checkpoint **"tiny"** (repo
> directory `ming_tiny/`, HuggingFace id
> `inclusionAI/Ming-3.0-tts-tiny`). It is not tiny — 7.9B total parameters.
> The name is the checkpoint's, not our editorial choice; the 0.5b model is
> the *dense* one.

Quick stats:

- **Server script**: [server_ming3TTS_moe.py](server_ming3TTS_moe.py)
- **Sample rate**: 44.1 kHz — the engine *enforces* 44100 as part of its
  release-contract validation (it refuses to load a checkpoint that claims
  anything else)
- **Default model**: `inclusionAI/Ming-3.0-tts-tiny` (~16 GB of weights in
  bf16; the vendor recommends an H100/H800-class GPU)
- **Watermarked**: no
- **Cloning mode**: ICL (in-context) only. `reference_text` (the clip's
  exact transcript) is **required** alongside the reference clip — the
  engine has no transcript-free clone mode for the exposed tasks, so
  omitting it is a 422, not a fallback (same convention as Qwen3-TTS MLX
  and Breeze TTS 2).
- **Languages**: `zh, en` (the engine's `seed` task) and `de, es, fr, ja, ko`
  (its `multilingual` task). Two-letter codes go straight to the engine;
  the server selects the task from the code. No `auto` — both tasks require
  an explicit target language.
- **Extra response field**: `stopped` — whether the engine's stop detector
  fired. `stopped: false` means the output may be truncated: try another
  `seed` or shorter text.

Not exposed (deliberate): voice design, built-in IP voices, podcast mode,
emotion / speed / dialect control, and the `speaker_only` conditioning that
only exists on the control tasks. This server is a voice-cloning server.

The two Ming servers pin conflicting dependencies, so keep them in separate
environments.

## Installation

The MoE runtime is CUDA-oriented with its own pins (the vendor's env is
Python 3.10.15 + torch 2.7.1+cu118). Like the dense model, the engine ships
as a plain git repo and must be on `PYTHONPATH`:

```
git clone https://github.com/inclusionAI/Ming-3.0-tts.git
cd Ming-3.0-tts
conda create -n ming-tiny python=3.10.15 -y && conda activate ming-tiny
pip install -r requirements-tiny.txt
pip install fastapi uvicorn loguru soundfile
pip install ../path/to/tts-serve/tts-engine-common   # or: pip install -e ...
```

Start the server:

```
cd Ming-3.0-tts
PYTHONPATH=. MING3_TTS_MOE_MODEL=/path/to/Ming-3.0-tts-tiny \
    python /path/to/tts-serve/impl/server_ming3TTS_moe.py
```

(`PYTHONPATH=.` is load-bearing: without it the server exits with an
actionable message rather than a stack trace.)

## Voice cloning with `speak.py`

```bash
# (the text is a positional argument; --ref-audio-transcript is a path to a
# text file containing the transcript)
python tools/speak.py --server http://localhost:7500 \
  --ref-audio reference.wav \
  --ref-audio-transcript reference_transcript.txt \
  --language ja \
  "こんにちは。これはテストです。"
```

`reference_text` is required, so `--ref-audio-transcript` is always
mandatory for this engine. Pick `--language` to match the *text* — it also
selects the engine task (`en`/`zh` → seed, `de/es/fr/ja/ko` → multilingual).

## Configuration (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `MING3_TTS_MOE_MODEL` | `inclusionAI/Ming-3.0-tts-tiny` | Local checkpoint dir or HuggingFace id |
| `MING3_TTS_MOE_DEVICE` | `cuda` | `cuda`, `cuda:N`, `cpu`, or `mps` (only CUDA is tested; the runtime env is CUDA-oriented) |
| `MING3_TTS_MOE_HOST` | `0.0.0.0` | Bind host for `python server_ming3TTS_moe.py` |
| `MING3_TTS_MOE_PORT` | `7500` | Bind port |

## Sampler defaults

`cfg_scale`, `flow_steps`, `sigma`, and `temperature` default to the
engine's own "formal" evaluation settings (`FORMAL_CFG=2.0`,
`FORMAL_FLOW_STEPS=10`, `FORMAL_SIGMA=0.25`, `FORMAL_TEMPERATURE=0.0`),
imported straight from `ming_tiny.task_protocol` — the server hardcodes no
magic numbers, so a vendor change flows through automatically.

## Reference audio handling

The engine loads the reference clip from a file path and re-encodes it on
every request (no path-keyed prompt cache). The server therefore writes a
UUID-named temp file and **deletes it when the request finishes** — nothing
accumulates on disk, unlike the 0.5b server's kept content-hashed staging.

## Reproducibility (`seed`)

The seed is passed to the engine, which seeds the whole pipeline itself at
the start of each synthesis (`seed_everything(seed)` inside `synthesize`).
The response echoes the seed used. On CUDA, reproducibility is near-exact,
not bit-exact (non-deterministic kernels).
