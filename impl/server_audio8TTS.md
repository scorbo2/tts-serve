# Audio8 TTS (Realistic Air)

Quick stats:

- **Server script**: [server_audio8TTS.py](server_audio8TTS.py)
- **Engine**: [Audio8_TTS](https://github.com/Edge0-AI/Audio8_TTS) — a plain
  git repo that publishes **no Python package**. The checkpoint ships its own
  Transformers modelling/processor code (`modeling_arktts.py`,
  `processing_arktts.py`, …), loaded with `trust_remote_code=True`. This
  server therefore imports *nothing* from the engine repo: it needs the
  checkpoint plus torch/transformers/soundfile.
- **Checkpoint**: `Edge0/Audio8-TTS-Realistic-Air` — 0.6B DualAR language
  model (24 layers, `dim=896`, `model_type: arktts`) plus a separate 10-codebook
  audio codec (`codec.pth`). 1.2 GB + 1.3 GB of weights on disk.
- **Sample rate**: 44.1 kHz (the checkpoint's own `codec_sample_rate`)
- **Device**: CUDA by default. `AUDIO8_TTS_DEVICE` accepts the engine CLI's
  grammar (`cuda`, `cuda:<index>`, `cpu`, `auto`); CPU must run `float32`
  (the engine's own rule) and measured **RTF 6.6** on a 24-core x86 box —
  six times slower than real time, so this is a GPU model. There is no MPS
  path: the engine's CLI grammar is `auto|cpu|cuda[:N]` and the server mirrors
  it, so `mps` is a startup error rather than an untested promise.
- **VRAM**: about **6.4 GiB** at bf16, measured on an RTX 4070 (12 GiB card)
  after a few warm requests, allocator reservations included. `fp32` roughly
  doubles the weight memory.
- **Speed**: RTF **0.6–0.8** on an RTX 4070 at bf16 (one request at a time;
  synthesis is serialized). The first request after startup also pays the
  weight load.
- **Notes**:
  - **this is the multilingual "Realistic" checkpoint.** Language is detected
    from the input text; Thai, Chinese, Japanese, French, German, Arabic and
    Hindi all synthesize with no configuration (each verified against this
    server). The vendor lists 23 languages for this checkpoint (11 shared with
    Preview + 12 added here) and warns that quality varies across them — the
    older `Preview-0.6b` checkpoint shares this processor but covers only the
    11.
  - the engine has **no language parameter**, so the server accepts any
    well-formed two-letter `language` code for API consistency and does not
    forward it; `/capabilities` advertises `languages: null` (the docs/02
    no-support case, same as LuxTTS / Breeze / Pocket TTS). Since the server
    never advertises `auto`, sending `auto` is a `422` like any other
    malformed code — there is no engine-side sentinel to map it to.
  - **reference audio and its exact transcript are one unit.**
    `audio_base64` without `reference_text` is a `422` — the bundled processor
    refuses a clip with no transcript ("reference_text is required when a
    reference voice is provided"). The reverse half-pair is rejected too,
    because dropping a field the client explicitly sent is worse than a
    clear 422.
  - **no reference at all is legal** and gives the checkpoint's own voice.
    There is no preset list and no voice-design mode, so "default voice" is
    the only non-cloning mode.
  - **no temp files anywhere.** Unlike most engines here, the processor takes
    in-memory `(samples, rate)` rather than a file path — no staging
    directory, nothing to clean up, no path-keyed prompt cache.
  - `seed` is meaningful; see the reproducibility note below.
  - the response carries an extra **`finished`** flag: `false` means
    generation ran out of its frame budget and the clip is probably cut off
    mid-word.

## Installation

You do **not** need the engine checkout, and there is nothing to `pip install`
from it. Only the checkpoint and its loader matter.

```
mkdir audio8 && cd audio8
python3 -m venv .venv && source .venv/bin/activate
pip install "torch>=2.5" "torchaudio>=2.5" "transformers>=4.57,<5" \
            numpy soundfile safetensors
```

Then the server's own deps, from a `tts-serve` checkout:

```
pip install -e /path/to/tts-serve/tts-engine-common fastapi uvicorn loguru
```

Run it from wherever you like (the server has no engine-path requirement):

```
python /path/to/tts-serve/impl/server_audio8TTS.py
```

Measured on this box: Python 3.12, `torch 2.15.0+cu132`, `transformers
4.57.6`. The checkpoint's `config.json` records `transformers_version:
4.57.3`, and the engine's `requirements.txt` pins `transformers>=4.57,<5` —
the 5.x line works too (it is what the repo's SGLang guide pins), but 4.57.x
is what this server was validated against.

`torchaudio` is not optional in practice: the processor resamples any
reference clip that is not 44.1 kHz, which is most clips.

The checkpoint (~2.5 GB: `model.safetensors` + `codec.pth`) downloads from
HuggingFace on first start.

### Pointing at a local checkpoint

```
huggingface-cli download Edge0/Audio8-TTS-Realistic-Air \
    --local-dir /srv/models/Audio8-TTS-Realistic-Air
export AUDIO8_TTS_MODEL=/srv/models/Audio8-TTS-Realistic-Air
python impl/server_audio8TTS.py
```

`AUDIO8_TTS_MODEL` is passed straight to `from_pretrained`, so it takes a
HuggingFace repo id or a local directory — and that directory must hold
**both** `model.safetensors` and `codec.pth`, because the modelling code
resolves the codec relative to the checkpoint directory. A half-downloaded
directory fails when the model loads at startup (the model is pre-loaded in
the FastAPI lifespan, so you see it in the startup log, not in a request).
Set `HF_TOKEN` only if you mirror the checkpoint into a private repo.

### Trust: `AUDIO8_TTS_MODEL` runs code

The checkpoint carries its own modelling and processor code and the server
loads it with `trust_remote_code=True` — there is no other way to run this
model, since the engine publishes no package. That is a real trust decision:
pointing `AUDIO8_TTS_MODEL` at an arbitrary repo id or directory makes the
server **execute** whatever `modeling_arktts.py` / `processing_arktts.py` that
directory names, with the server's own permissions, at startup. Point it at
`Edge0/Audio8-TTS-Realistic-Air` (or a mirror of it you verified), not at
"some TTS repo I found", and do not expose `/synthesize` to untrusted clients
in the hope that only the weights matter — the weights are not all there is.

One upside of loading through `from_pretrained`: the rates and budgets are read
back off the loaded config (`codec_sample_rate`, `max_seq_len`), not hardcoded
per request. The sibling `Preview-0.6b` checkpoint shares this processor but
ships a different context window; a checkpoint like that changes what the
server enforces, rather than silently overflowing a constant the server
assumed. `/capabilities` is derived at import time and still reports the
module-level defaults, so if you do swap checkpoints, regenerate the snapshot.

### Host and port

Defaults to `0.0.0.0:7500`:

```
export AUDIO8_TTS_HOST=127.0.0.1 AUDIO8_TTS_PORT=8500
```

### Precision

```
export AUDIO8_TTS_DTYPE=bfloat16   # default on CUDA (auto = bf16 on CUDA, fp32 on CPU)
export AUDIO8_TTS_DTYPE=float16    # slightly less memory, same speed
export AUDIO8_TTS_DTYPE=float32    # required when AUDIO8_TTS_DEVICE=cpu
```

## Using it

```bash
curl http://localhost:7500/health
# {"status":"ok","serverType":"Audio8 TTS","model":"Edge0/Audio8-TTS-Realistic-Air",
#  "device":"cuda","version":"1.5"}

curl http://localhost:7500/capabilities      # the full parameter contract
```

Clone a voice (clip **plus** its exact transcript). The text is a positional
argument and `--ref-audio-transcript` is a path to a text file:

```bash
python tools/speak.py --server http://localhost:7500 \
  --ref-audio speaker.wav \
  --ref-audio-transcript speaker_transcript.txt \
  --output-file clone.wav \
  "This is a cloned take, in the same voice."
```

Engine-specific knobs show up as flags because `speak.py` reads them from
`/capabilities`:

```bash
python tools/speak.py --server http://localhost:7500 \
  --ref-audio speaker.wav --ref-audio-transcript speaker_transcript.txt \
  --output-file greedy.wav --seed 42 --max-new-tokens 300 --greedy \
  "Deterministic take."
```

Note that `speak.py` insists on `--ref-audio` (or `--persona-dir`) — per
docs/03 it is a required input — so the **no-reference** mode has to go
through the raw endpoint:

```bash
curl -s http://localhost:7500/synthesize -H 'Content-Type: application/json' \
  -d '{"text":"No reference clip, so this is the checkpoint's own voice.","seed":3}' \
  | python3 -c 'import base64,json,sys; sys.stdout.buffer.write(base64.b64decode(json.load(sys.stdin)["audio_base64"]))' \
  > default-voice.wav
```

Same server, different languages, no restart (the language comes from the
text):

```bash
python tools/speak.py --server http://localhost:7500 --ref-audio speaker.wav \
  --ref-audio-transcript speaker_transcript.txt --output-file th.wav \
  "สวัสดีครับ ยินดีที่ได้รู้จักนะครับ"
python tools/speak.py --server http://localhost:7500 --ref-audio speaker.wav \
  --ref-audio-transcript speaker_transcript.txt --output-file zh.wav \
  "你好，很高兴认识你。"
python tools/speak.py --server http://localhost:7500 --ref-audio speaker.wav \
  --ref-audio-transcript speaker_transcript.txt --output-file fr.wav \
  "Bonjour, comment allez-vous ?"
```

## Parameters

The full, authoritative list is `GET /capabilities`; this is the short
version.

| Field | Meaning |
| --- | --- |
| `text` | required, non-blank, ≤ 1500 characters (see "context budget") |
| `audio_base64` | optional reference clip (WAV/MP3/OGG/FLAC, 2–20 s) |
| `reference_text` | **required with `audio_base64`**: what the clip actually says |
| `language` | accepted (two-letter code only — `auto` is a `422`) but **not forwarded**; capabilities advertise `languages: null` |
| `seed` | 1–1000; `null` = random, echoed back in the response |
| `max_new_tokens` | 1–2048, default **1024**. Budget in audio *frames*: one frame = 2048 codec samples ≈ 21.5 frames/second of audio, so 1024 ≈ 47 s. Clamped to whatever the prompt leaves over (see "context budget") |
| `retry_max_new_tokens` | 1–2048, default **2000**. One retry with this budget when the first pass never emitted end-of-speech; ≤ `max_new_tokens` disables it, and so does a prompt so wide that both budgets clamp to the same ceiling |
| `temperature` | (0, 2], default 0.8. Must be positive — the model rejects 0; `greedy` is the deterministic switch |
| `top_p` | (0, 1], default 0.95 |
| `top_k` | 1–4096, default 50. Must be positive — the model's own filter raises on 0 |
| `greedy` | default `false`; `true` = argmax decoding (seed-independent, flatter prosody) |

Response: `audio_base64` (WAV, 44100 Hz), `sample_rate`, `seed`, `fid`,
`time_used`, `rtf`, `finished`.

## Reference audio

The clip and its transcript are a continuation pair: the model is shown
"this text was spoken like this", and it imitates the delivery.

- **Transcribe the clip yourself, exactly.** There is no ASR anywhere in this
  pipeline — no Whisper, no on-the-fly transcription (unlike OmniVoice or
  LuxTTS). The vendor's rule is the same: "Use your own recording at
  `reference.wav`. Its transcript should match the spoken content exactly."
  Keep the disfluencies of the take; the transcript is part of the condition,
  not metadata.
- A **wrong** transcript is not an error. Measured: cloning a clip while
  claiming a longer transcript than it actually speaks still returns `200`
  with plausible audio — just a weaker match and a higher chance of an early
  cutoff. There is nothing here to validate against, so exactness is the
  client's job.
- **Clips are never truncated, and the length that matters is the decoded
  one.** The 2–20 s window is not a quality heuristic, it is arithmetic: every
  second of clip takes ~21.5 of the model's 2048 context positions, so an
  over-long clip is an over-budget request rather than a degraded one. Over 20 s
  is a `400` that names the clip's actual duration; under 2 s is a `400` too
  (measured: a sub-second clip still generates, it just does not carry a voice).
  The container header is only an advertisement, so the window is enforced
  again on the samples that actually decoded — and a clip whose header and
  audio disagree by more than 0.5 s is a `400` naming both numbers, rather than
  a take conditioned on a mystery slice of the reference.
- **An MP3 reference must carry a length table (`Xing`/`Info`/`VBRI`).** Without
  one there is no duration to measure: a decoder can only guess from the first
  frame's bitrate and stops at the guess. Measured: a 40.05 s VBR MP3 written by
  `ffmpeg -write_xing 0` reads back as **6.73 s** — 33 seconds of the speaker
  silently vanish, and every downstream check (including the 2–20 s window)
  then passes on the truncated remainder. Such a clip is refused with a `400`
  that says to re-encode; WAV/FLAC/OGG carry honest lengths, and libmp3lame
  writes the table unless asked not to.
- 3–10 s of clean single-speaker audio is the sweet spot.
- Stereo and any sample rate are fine: the server mixes down to mono and the
  processor resamples to 44.1 kHz. Samples that are not finite (NaN, inf) are a
  `400` — such a clip cannot carry a voice, only a crash.

## Long text, and the one shared context budget

`max_seq_len = 2048` positions carry the prompt (transcript + reference
clip) **and** everything the model generates. The server keeps that budget
explicit instead of letting the engine fail:

- The vendor's own quality advice is tighter than the schema: "keep each
  input within 150 characters" and split anything longer. Follow it — the
  schema's 1500-character ceiling is about *not crashing*, not about sounding
  good.

- `text` ≤ **1500 characters** (`422` beyond that) — about 950 tokens of
  English, which leaves room for the default 1024-frame budget.
- `max_new_tokens` ≤ **2048**, the whole context (the model silently clamps
  it to `max_seq_len - prompt_width` anyway; the schema just refuses
  obviously-impossible values).  The server does the same arithmetic and asks
  for the clamped number, so what is logged, retried and answered is what the
  model was actually allowed to generate.
- The tokenized prompt is **measured** after the processor runs; if it does
  not fit, the request is a `400` that names both the text and the clip and
  says which one to shorten.  That measurement needs the processor, which lives
  behind the synthesis lock — so this particular `400` arrives in queue order,
  after whatever generation is already in flight (the schema and audio checks
  still answer immediately).
- A prompt that fits but leaves **less than 256 frames (~12 s) of room** is
  also a `400`, with the full breakdown (text tokens, clip frames, positions
  left over). That case is not theoretical: 1500 Devanagari characters plus a
  19.4 s clip cost 1897 of 2048 positions, leaving 151 frames — and the server
  answered `200` with **one frame** of audio (0.0 s, `finished: true`, RTF 60).
  Refusing beats answering with a sneeze. The floor brackets two runs of that
  same shape: 151 frames left over produced a single frame of audio, 623 left
  over produced 3.2 s of clean speech — which is still not the whole 1500-character
  script, because the model stops early on long inputs no matter what the budget
  says.
- A wide prompt can also starve the model with room to spare (measured: the same
  script with 651 frames left over still collapsed to a **0.05 s** take at RTF
  5.2). Nothing server-side catches that one — the model chose to stop — which
  is why the vendor's "keep each input under 150 characters" is the rule that
  actually protects you, and why `finished` is a truncation flag rather than a
  quality guarantee. A `200` here means "the model spoke and stopped"; listen to
  it.
- If generation merely *hits* its budget, the server retries **once** with
  `retry_max_new_tokens` and returns the better take. Set
  `retry_max_new_tokens` equal to `max_new_tokens` to turn the retry off
  (it doubles the latency of the affected request, which is the price of not
  returning a mid-word clip). The retry is also skipped on its own when the
  prompt is so wide that both budgets clamp to the same ceiling: a second pass
  could not add a single frame, it could only re-roll the same length and make
  the client wait twice for it.
- After that, `finished` tells the truth: `false` = the audio ran out of
  budget.

A ~1200-character English paragraph (≈ 30 s of audio) fits in a single
request. Much longer than that: split it client-side and concatenate the
audio (the engine repo's own `split_sentences` helper exists for exactly
this).

## Reproducibility

`seed_everything()` (Python / NumPy / PyTorch) runs inside the synthesis lock,
and sampling uses a dedicated `torch.Generator` — the same shape as the
engine CLI. Measured on a real GPU:

- two **warm** requests with the same text, seed and clip → **byte-identical**
  WAVs;
- the **first** request in a fresh process differs from later ones with the
  same seed (kernel/backend selection settles during that first run), so
  cold-start output is not reproducible;
- `greedy: true` ignores the seed entirely: two greedy requests with
  different seeds give identical audio (also measured);
- different seeds with sampling genuinely differ — that is the knob for
  "give me another take".

There is no server-side prompt cache to wipe (the model re-encodes the clip
every request), so nothing about the on-disk state affects reproducibility.

## Gotchas

- **`model.generate()` mutates state on the module** — `_setup_generation_caches`
  writes the decode caches onto the model instance, so two concurrent
  generations clobber each other. All generation (and the processor call that
  feeds it) is behind one lock. Request validation and the audio pre-flight
  deliberately happen *before* the lock, so a client's `400`/`422` never
  queues behind somebody else's speech — except the two prompt-budget `400`s,
  which can only be measured from the processor's own output and so answer in
  queue order.
- **The over-budget failure modes are ugly and this server translates them.**
  Natively: `ValueError: Prompt length N must be smaller than 2048` (raised by
  the model), or a `RuntimeError` out of the attention kernel
  (`The current input length (N) exceeds the maximum context length of 2048 in
  FlashAttentionFunction`) plus a CUDA `vectorized_elementwise_kernel` assert.
  Both become an actionable `400`.
- **`temperature: 0` and `top_k: 0` are `422`s here, not "disable
  sampling".** The bundled `ArkttsLegacyTopKTopPLogitsProcessor` raises on
  non-positive `top_k`, and `generate()` raises
  `ValueError: temperature must be positive`. Use `greedy: true`.
- **Truncated output is not an error** — the model returns what it got, so the
  server cannot tell a complete sentence from a clipped one except by the
  `finished` flag. Hence the retry, and hence the flag, rather than a silent
  cut or a 500. Note the other direction too: `finished: true` means the model
  *said* it was done, not that the take is usable — which is why an
  under-budgeted prompt is refused before generation rather than answered.
- **Out-of-range samples never ship as full-scale PCM.** The codec can hand
  back a waveform that is not finite, and `np.clip` does nothing to NaN/inf
  (`np.clip(nan)` is still NaN) while libsndfile writes them at digital full
  scale (measured: `nan → -32768`, `+inf → +32767`). The encoder runs
  `np.nan_to_num` before the clamp, so a NaN becomes silence instead of a
  crack that the client blames on the model.
- **The engine CLI's `--retry-badcase-layers` is deliberately not exposed.**
  The vendor's own inference guide says those layers "do not match Realistic
  Air", so the server offers no env var for them. Same reasoning for the CLI's
  `--cache-by-prompt`: that cache lives in the CLI script, not in the model,
  and this server does not use the CLI.
- **The engine repo and the checkpoint carry two different copies of the
  runtime.** `audio8_tts_infer.py` in the repo is a batch CLI with its own
  prompt cache keyed by *text only*; the checkpoint's bundled
  `modeling_arktts.py` is what transformers actually runs. This server goes
  straight to `AutoModel`/`AutoProcessor`, so the CLI's text-only cache (which
  would reuse a previous clip's encoding for the same text) never applies.
  Do not "fix" this by importing the engine CLI.
- **`watermarked: false`** — nothing in this engine marks its output, and
  capabilities say so.
