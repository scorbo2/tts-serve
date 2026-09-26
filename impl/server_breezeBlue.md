# Breeze TTS 2

Quick stats:

- **Server script**: [server_breezeBlue.py](server_breezeBlue.py)
- **Sample rate**: 24 kHz
- **Notes**:
  - two synthesis modes, both reference-based:
    - **Voice clone** — reference audio + its exact transcript
      (`reference_text`, **required** on this server). The engine preserves
      the speaker's timbre, rhythm, emotion, and style as-is.
    - **Voice direction** — the same, plus an `instruction` (natural
      language; e.g. "Speak slowly with a restrained, serious tone"). The
      speaker's identity is kept while the instruction steers tone,
      emotion, pace, and delivery. `cfg_scale` strengthens
      instruction-following (engine README recommends 4; the engine default
      is 1.0).
  - the engine's third mode, **voice design** (no reference audio; a
    natural-language voice description), is **deliberately not exposed** —
    the client application does not support it. `audio_base64` is therefore
    always required; omitting it is a `422`, not a mode switch.
  - `reference_text` must be the **exact** transcript of the clip: the
    engine conditions on the audio and transcript together, and a wrong
    transcript degrades cloning quality (include repetitions if the speech
    loops in the clip).
  - the engine has **no language parameter**: it is bilingual (English /
    Chinese) and auto-detects from the input text. The API accepts any
    well-formed two-letter `language` code for consistency but does not
    forward it, and capabilities advertise `languages: null`.
  - **CUDA only**: the streaming runtime hard-rejects non-CUDA devices at
    load time. ~7.7 GiB GPU memory for eager inference (12 GB GPU minimum);
    ~14.4 GiB with the fast path (24 GB GPU).
  - the checkpoint must be a **local directory**: the engine downloads the
    backbone and text tokenizer from HuggingFace when given an HF id, but it
    reads the checkpoint's bundled `audio_tokenizer/` subdirectory from
    local disk — so download the full checkpoint first (see below).
  - inline **vocal events** are supported in `text`: parentheses in English
    (`(laugh)`, `(sigh)`, `(clears throat)`), square brackets in Chinese
    (`[笑]`, `[叹气]`).
  - `seed` is meaningful: the process RNG is re-seeded per request and the
    engine re-seeds PyTorch at the start of sampling, so the same seed +
    inputs reproduce the same sampling start. On CUDA the audio is
    near-identical but not bit-identical (non-deterministic GPU kernels).
  - the engine loads reference audio from a file path; it re-encodes each
    request (no path-keyed cache), so the server writes a UUID temp file
    and deletes it per request.
  - synthesis is **serialized** (single shared model): the streaming
    runtime mutates shared KV/codec state in place and the engine's own API
    is single-request behind a lock.
  - model weights are **research / non-commercial** (BreezeBlue Research
    and Non-Commercial License; the Apache-2.0 source license does not
    grant commercial rights to the weights or self-hosted outputs).

## Installation

Start by setting up a venv (or use your conda setup):

```
mkdir BreezeTTS
cd BreezeTTS
python3 -m venv .venv
source .venv/bin/activate
```

Clone the engine and install its requirements (PyTorch 2.9.1, transformers,
qwen-tts, ...):

```
git clone https://github.com/breezeblue-ai/breeze-tts.git
cd breeze-tts
python -m pip install -r requirements.txt
```

Download the Breeze TTS 2 checkpoint to a local directory (it must contain
the bundled `audio_tokenizer/` subdirectory):

```
huggingface-cli download BreezeBlue/breeze-tts-2 --local-dir /models/breeze-tts-2
```

Clone `tts-serve` and install its shared layer plus the FastAPI server
deps:

```
git clone https://github.com/scorbo2/tts-serve
cd tts-serve
pip install ./tts-engine-common fastapi uvicorn loguru soundfile
```

Start it up from the breeze-tts repo root, with the repo on `PYTHONPATH`
(the server imports the engine's `breeze_infer` and `models` packages.  A
plain `python /path/to/script.py` puts only the script's *own* directory on
`sys.path` — not your working directory — so `PYTHONPATH=.` is mandatory):

```
cd breeze-tts
PYTHONPATH=. BREEZEBLUE_MODEL=/models/breeze-tts-2 python ../tts-serve/impl/server_breezeBlue.py
```

### Changing host

By default, `0.0.0.0` is used. To force a local-only server:

```
export BREEZEBLUE_HOST=127.0.0.1
```

### Changing port

By default, `7500` is used. To choose a different port:

```
export BREEZEBLUE_PORT=8500
```

### To use a local model path

By default, `BREEZEBLUE_MODEL` is `BreezeBlue/breeze-tts-2` (the HuggingFace
repo id). The engine *can* download the backbone and text tokenizer from
that id, but the bundled `audio_tokenizer/` must exist on local disk, so in
practice you always want an explicit local directory:

```
export BREEZEBLUE_MODEL=/models/breeze-tts-2
```

### Run on a specific GPU

By default, the model loads on `cuda` (device 0). To pin a specific GPU:

```
export BREEZEBLUE_DEVICE=cuda:1
```

`cuda` or `cuda:<index>` are accepted — the streaming runtime has no
CPU/MPS backend, and anything else is rejected at startup.

### Fast path (optional)

By default the server runs eager inference (~7.7 GiB GPU memory). To enable
the fast (CUDA-graph) path for every inference stage — better time-to-first
audio and RTF, at the cost of a one-time graph warmup at startup and
~14.4 GiB of GPU memory:

```
export BREEZEBLUE_FAST_ALL=1
```

The warmup reads the engine's own profile from the breeze-tts repo
(`configs/fast.json`), which is why the repo root must be importable. The
profile's `cfg_scales` (1.0 and 4.0) determine which branch-batch graphs
are frozen; the guidance scale itself is applied at replay time, so any
`cfg_scale` works without recapture. The prefill graphs only cover
512-token buckets, though — a reference clip whose audio frames, exact
transcript, and target text together exceed that budget will fail in fast
mode (the eager path tolerates prefills up to 2048 tokens).

### Reference audio preparation

For both modes, provide a clean, non-looping clip of the target voice
(speech, not music) with minimal background noise — roughly 3-10 s works
well. Its exact transcript goes in `reference_text` (required). Clips
shorter than 2 s are rejected with a `400` at the request boundary.
