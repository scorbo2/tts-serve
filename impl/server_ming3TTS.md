# Ming 3 TTS (0.5b dense)

[Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) is a
general-audio-generation model family from Ant Group's inclusionAI lab. It
ships **two checkpoints with two separate Python stacks**, so tts-serve has a
server for each:

- this document covers the **0.5b dense** checkpoint (`ming_tts.MingTTS`);
- the 7.9B-total / 1.3B-active MoE (engine name "tiny") has its own server:
  [Ming 3 TTS (MoE)](server_ming3TTS_moe.md).

The two stacks pin conflicting dependencies (torch 2.6.0 vs 2.7.1+cu118,
x-transformers 2.9.2 vs 2.7.4, Python 3.11 vs 3.10.15 — only
`transformers==4.46.3` is common to both), so **do not install both in one
environment** — one venv/conda env per server.

Quick stats:

- **Server script**: [server_ming3TTS.py](server_ming3TTS.py)
- **Sample rate**: 44.1 kHz (the released 0.5b checkpoint's audio-VAE rate)
- **Default model**: `inclusionAI/Ming-3.0-tts-0.5b` (~3 GB, not gated on
  HuggingFace)
- **Watermarked**: no
- **Cloning modes**: zero-shot. With `reference_text` the clone is
  **in-context** (the reference clip's acoustic continuation *and* its
  speaker embedding condition the output — closest match). Without it the
  server transparently falls back to **speaker-embedding-only** cloning
  (timbre closer, prosody freer) — the same convention as Qwen3-TTS, so
  omitting the transcript is a mode switch, not an error.
- **Languages**: `zh, en, ja, ko, de, fr, es` (two-letter codes, sent to
  the engine verbatim — unlike the Qwen3-TTS servers there is no
  name-mapping table). No `auto`: the engine's TTS task has no
  auto-detection sentinel.
- **Paralinguistic tags**: inline tags work in `text` — `[Breathing]`,
  `[Cough]`, `[Sigh]`, `[Laughter]`, `[Surprise-ah]`, `[Surprise-oh]`,
  `[Dissatisfaction-hnn]`, `[Uhm]`, `[Confirmation-en]`, `[Question-ei]`,
  `[Question-ah]`, `[Question-en]`, `[Question-oh]`, `[Question-huh]`.

Not exposed (deliberate): the engine's other task modes — voice design from
text, built-in IP voices, emotion / speed / dialect control, podcast. This
server is a voice-cloning server.

## Installation

The engine ships as a plain git repo (no PyPI package), so the server needs
the repo root on `PYTHONPATH`. Set up an environment for it:

```
git clone https://github.com/inclusionAI/Ming-3.0-tts.git
cd Ming-3.0-tts
conda create -n ming-tts python=3.11 -y && conda activate ming-tts
pip install -r requirements.txt
pip install fastapi uvicorn loguru soundfile pynini
pip install ../path/to/tts-serve/tts-engine-common   # or: pip install -e ...
```

The 0.5b checkpoint (~3 GB) downloads from HuggingFace on first start. To
pre-fetch it and run offline:

```
huggingface-cli download inclusionAI/Ming-3.0-tts-0.5b \
    --local-dir /path/to/Ming-3.0-tts-0.5b
```

Start the server (note the `PYTHONPATH=.` — running the script without it
turns into an actionable startup error, not a stack trace):

```
cd Ming-3.0-tts
PYTHONPATH=. MING3_TTS_MODEL=/path/to/Ming-3.0-tts-0.5b \
    python /path/to/tts-serve/impl/server_ming3TTS.py
```

## Voice cloning with `speak.py`

```bash
# In-context clone (best voice match): clip + its exact transcript
# (the text is a positional argument; --ref-audio-transcript is a path to a
# text file containing the transcript)
python tools/speak.py --server http://localhost:7500 \
  --ref-audio reference.wav \
  --ref-audio-transcript reference_transcript.txt \
  "Hello there, this is a test."

# Speaker-embedding-only clone: omit the transcript
python tools/speak.py --server http://localhost:7500 \
  --ref-audio reference.wav \
  "Hello there, this is a test."
```

A reference clip of at least 2 s is required; 3–10 s of clean speech is the
sweet spot.

## Configuration (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `MING3_TTS_MODEL` | `inclusionAI/Ming-3.0-tts-0.5b` | Local checkpoint dir or HuggingFace id |
| `MING3_TTS_DEVICE` | `cuda` | `cuda`, `cuda:N`, `cpu`, or `mps` |
| `MING3_TTS_TEXT_NORMALIZATION` | `1` | Set `0` to skip the FST text normalizer (numbers/dates/units are then spoken as written) |
| `MING3_TTS_HOST` | `0.0.0.0` | Bind host for `python server_ming3TTS.py` |
| `MING3_TTS_PORT` | `7500` | Bind port |

### Text normalization gotcha (`pynini`)

The engine's text normalizer builds FST grammars with `pynini`, which has no
universal wheel (conda-only on many platforms). If `pip install pynini`
fails on your box, start with `MING3_TTS_TEXT_NORMALIZATION=0` — everything
else runs fine, numbers just get read as written.

### VRAM

The checkpoint is ~3 GB of weights in bfloat16 plus the audio VAE. It fits
comfortably on ≥ 8 GB cards; 4 GB cards are tight (decode at 44.1 kHz may
OOM on long texts — try `MING3_TTS_DEVICE=cpu`, slow but functional, or
shorter texts).

## Reproducibility (`seed`)

`seed` is meaningful — the server re-seeds the whole pipeline (Python,
NumPy, PyTorch) inside the synthesis lock. But there is a caveat inherited
from the engine: **encoded reference prompts are cached in memory per
clip**, and the first request for a clip runs the VAE encoder (which
samples its posterior and consumes RNG) while later requests skip it.
Repeat (warm-cache) requests with the same seed therefore reproduce each
other exactly — but the very first (cold) request for a clip does *not*
reproduce against later ones: reproducing it requires a fresh server. Only
a **server restart** resets the prompt cache — wiping the staging directory
(`/tmp/ming3_tts_rest_api` on Linux) reclaims disk but leaves the
in-memory cache untouched. On CUDA, reproducibility is near-exact, not
bit-exact (non-deterministic kernels).

Because that cache is keyed by file path, reference clips are staged as
content-hashed (SHA-256) files that are **kept** on disk — repeat requests
for the same clip reuse the engine's cache. Wipe the staging directory to
reclaim space.
