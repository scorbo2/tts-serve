# Implementations

One FastAPI server per TTS engine. Each server:

- loads its model once on startup (HuggingFace download on first run,
  can be explicitly pointed at a locally-downloaded model via env vars
  to run 100% offline),
- exposes `GET /` (landing page), `GET /health`, `GET /capabilities`,
  `GET /docs`, and `POST /synthesize`,
- validates requests with a Pydantic model and derives `/capabilities` from
  that same model via [`tts-engine-common`](../tts-engine-common/README.md),
- rejects unknown fields (`422`) and bad reference audio (`400`) before
  touching the model.

More information:

- [Chatterbox](server_chatterbox.md)
- [OmniVoice](server_omnivoice.md)
- [Qwen3-TTS](server_qwen3TTS.md)
- [Qwen3-TTS (MLX)](server_qwen3TTS_mlx.md)
- [Faster Qwen3-TTS](server_fasterQwen3TTS.md)
- [dots.tts](server_dotsTTS.md)
- [Index-TTS](server_indexTTS.md)
- [LuxTTS](server_luxTTS.md)
- [VoxCPM](server_voxcpm.md)


## Running

The full parameter list for each server is at its `GET /capabilities` —
don't trust this README over that endpoint, the schema is the source of truth.

## Language handling

Per [docs/02-language-handling.md](../docs/02-language-handling.md), the
`language` contract is a **two-letter lowercase code** (`en`, `fr`, `de`,
...); omitted or empty values are normalized to `en` at the request boundary
(shared helpers in `tts_engine_common.language`). Engines whose internal
format differs map it at their own server — the client never sees it:

- **Qwen3-TTS** maps codes to lowercase names (`en` → `english`);
- **Qwen3-TTS (MLX)** maps codes to lowercase names (`en` → `english`),
  identically to Qwen3-TTS;
- **faster-qwen3-tts** maps codes to lowercase names (`en` → `english`);
- **dots.tts** maps codes to uppercase (`en` → `EN`);
- **IndexTTS-2.5** maps codes to uppercase (`en` → `EN`) and accepts only
  the five advertised codes (`ar`, `en`, `es`, `ja`, `zh`) — anything else
  422s, because the engine would silently degrade to its `common` vocabulary.

Engines with auto-detection expose it as the special value `auto`
(Qwen3-TTS, Qwen3-TTS (MLX), faster-qwen3-tts, dots.tts).

**LuxTTS** is the no-support case: the engine has no language parameter at
all (its tokenizer auto-detects English and Chinese per text segment), so the
server accepts any well-formed two-letter code for API consistency, does not
forward it, and advertises `languages: null` in capabilities.

## Engine-specific notes

- **Chatterbox** — no `reference_text` field: it conditions purely on the
  reference audio. Output is PerTh-watermarked by the library itself. Only the
  first 10 s of the reference clip are used.
- **OmniVoice** — omitting `reference_text` triggers an on-the-fly Whisper
  transcription; the first such request pays the ASR model load. References
  over 20 s are trimmed at the largest silence gap.
- **Qwen3-TTS** — `language` takes two-letter codes (`en`, `zh`, ...) or
  `auto` for auto-detection (mapped to the engine's lowercase names
  internally). Omitting `reference_text` transparently enables
  speaker-embedding-only mode (`x_vector_only_mode`), since the engine's
  in-context mode hard-requires a transcript.
- **Qwen3-TTS (MLX)** — Apple-Silicon-native counterpart to Qwen3-TTS, run
  through `mlx-audio` instead of `qwen_tts`/PyTorch, for a controlled A/B
  comparison between the two backends. ICL cloning only, so `reference_text`
  is **required** (no speaker-embedding fallback). No device env var: MLX has
  no device-selection knob, so `device` is always reported as `mlx`. `seed`
  is meaningful (seeds MLX's global RNG via `mx.random.seed()`, which drives
  the talker's token sampling). Does not (yet) expose `temperature` /
  `top_p` / `repetition_penalty`, long-text chunking, streaming, or
  voice-library profiles — see
  [server_qwen3TTS_mlx.md](server_qwen3TTS_mlx.md) for the full list.
- **faster-qwen3-tts** — CUDA-only fork of Qwen3-TTS (its CUDA-graph backend
  rejects non-CUDA devices at load time; PyTorch ≥ 2.5.1 required).
  `xvec_only` (default `true`) selects x-vector mode: the reference audio
  supplies a speaker embedding, the voice stays consistent across requests
  (recommended for sentence-by-sentence streaming), and `reference_text` is
  ignored. `xvec_only=false` switches to ICL mode: `reference_text` becomes
  **required** (no automatic fallback to x-vector mode) and the voice can vary
  between requests — pass the same `seed` for each sentence when streaming.
  `language` takes two-letter codes or `auto` (mapped to the engine's
  lowercase names).
- **dots.tts** — 48 kHz output (unlike the 24 kHz engines). The runtime
  demands a file path for the prompt audio, so the server writes a temp file;
  the reference transcript is optional. `language` takes two-letter codes or
  `auto` (mapped to the engine's uppercase codes / `auto_detect`).
- **IndexTTS-2.5** — 22.05 kHz output (third distinct rate). No
  `reference_text` field: it conditions on the reference audio alone, and
  only its first 15 s are used (hard cut). The engine wants file paths for
  prompt audio, so reference clips are staged as content-hashed (SHA-256)
  temp files that are *kept* — the model caches conditioning per path, so
  per-request deletion would race a queued request for the same clip; wipe
  the staging dir to reclaim space. Emotion is steered with at most one of
  `emotion_audio_base64` (a second clip; `emotion_alpha` blends it with the
  speaker's own emotion),   `emotion_vector` (8 components
   `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]` —
   the same names the capabilities doc advertises in `item_labels`), or
   `emotion_text` (free text; requires starting the server with
   `INDEXTTS_USE_QWEN_EMO=1`, else 400). `duration_factor` stretches
   (>1) / compresses (<1) output length. `seed` is best-effort (the engine's
   inference API has no seed parameter).
- **LuxTTS** — 48 kHz output (with dots.tts). No `reference_text` field:
  the engine *always* transcribes the reference clip with Whisper
  (openai/whisper-base on GPU, whisper-tiny on CPU) and conditions on that —
  every request pays the ASR cost, and the first request after startup also
  pays a one-time librosa initialisation (~10 s). No `language` forwarding:
  the engine's tokenizer auto-detects English and Chinese per segment
  (other scripts are dropped). `seed` is meaningful (solver initial noise
  from the PyTorch RNG): bit-identical output on CPU, near-identical on
  CUDA (non-deterministic GPU kernels). The runtime demands a file path for
  prompt audio,
  so the server writes a temp file; the reference is one-shot (no engine
  prompt cache), so the file is deleted per request. Synthesis is serialized
  with a lock: `generate_speech()` mutates the shared vocoder's `return_48k`
  flag in place. `return_smooth` selects the 24 kHz vocoder head
  (upsampled to 48 kHz) instead of the full-band 48 kHz head — same rate,
  different artifact profile; try it if you hear metallic artifacts.

- **VoxCPM** — 48 kHz output. VoxCPM2 is a 2B parameter tokenizer-free TTS
  model with three modes: **voice design** (no reference audio; describe the
  voice in parentheses at the start of `text`), **controllable cloning**
  (reference audio only), and **ultimate cloning** (reference audio +
  transcript for audio-continuation cloning). No `reference_text` forwarding:
  the engine auto-detects language from text content (30 languages internally),
  so the server accepts any well-formed two-letter `language` code for API
  consistency but does not forward it. Reference audio is staged to a
  UUID-named temp file (deleted per-request). `seed` is meaningful — the
  engine applies it via `torch.manual_seed()` before generation.

## Tests (`tests/`)

```bash
python -m pytest tests/
```

The suite runs **without torch, NumPy, soundfile, or any engine installed** —
`tests/stubs/` provides import-only stand-ins so the servers can be imported
and their HTTP surface exercised. Stubs are appended to `sys.path` *last*, so
on a machine with the real packages installed (e.g. a GPU box) the real ones
always win and the tests run against them.

What is covered:

- `GET /capabilities` — full-document snapshot per engine
  (`tests/snapshots/*.json`) plus targeted assertions (core fields, enums,
  sample rates).
- `GET /health` and the `GET /` landing page.
- `POST /synthesize` request-body validation: `422` on unknown fields,
  missing/empty fields, out-of-range values, and bad enum members.
- `POST /synthesize` reference-audio pre-flight: `400` on undecodable audio
  and clips shorter than the engine's minimum (the stub `soundfile.info()`
  parses real WAV headers via the stdlib `wave` module).
- faster-qwen3-tts only: the `/synthesize` success path with a fake model —
  pins what the server forwards to the engine (the `xvec_only` mode flag and
  transcript handling); real audio generation is still out of scope.

What is *not* covered: actual synthesis (needs a real model + GPU).  The
`/synthesize` success path is exercised only where a test installs a fake
model (currently faster-qwen3-tts).

### Snapshots

Regenerate after changing a server's request schema (or engine constants):

```bash
python tests/update_snapshots.py
```

Then review the diff. The dots.tts and IndexTTS-2.5 snapshots record the
`device` observed on the generating machine; their tests compare everything
except `device` (machine-dependent — the runtime auto-selects) and only
assert the value is a sane one (`cuda`/`cpu` for dots.tts;
`cuda`/`cpu`/`mps`/`xpu` for IndexTTS-2.5).

### Gotchas

- The environment is pinned to the servers' documented defaults in
  `tests/_bootstrap.py` (shared with `update_snapshots.py`) so snapshots are
  deterministic regardless of ambient `*_DEVICE` / `*_MODEL` variables.
- Pydantic v2's lax validation coerces `"yes"`/`"true"`/`"1"` to booleans —
  the boolean-rejection tests use values that are *not* coercible.
- The stub `numpy` module must satisfy pytest's own introspection
  (`isscalar`, `bool_`, `ndarray`, `asarray`); see the comment in
  `stubs/numpy/__init__.py` if cross-suite runs start failing with
  `AttributeError: module 'numpy' has no attribute ...`.
