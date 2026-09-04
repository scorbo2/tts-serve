---
name: new-tts-engine
description: Use when adding a new TTS engine to tts-serve — writing a new impl/server_<name>.py, its import-only stub in impl/tests/stubs/, its test file, and the snapshot/_bootstrap/README registration. Covers the engine-research checklist and the exact file structure to follow.
---

# Adding a new TTS engine

Workflow for wrapping a new open-source TTS engine as a tts-serve FastAPI server
(design: `docs/01-server-generification.md` decisions D1–D7; `docs/00-project-overview.md` Goal 3).

Canonical references — copy their structure exactly, adapt only where the new engine
forces you to: `impl/server_chatterbox.py` (standard case), `impl/server_dotsTTS.py`
(auto-device, free-form language, 48 kHz variant), `impl/tests/test_server_chatterbox.py`,
`impl/tests/stubs/chatterbox/`. General repo conventions live in `AGENTS.md` — read it first.

## Phase 1 — Research the new engine

Clone the engine; read its README and the source around its `generate()`/`synthesize()`
entry point (and any demo app). Extract:

| Question | Why it matters |
|---|---|
| PyPI package name **and** import name | `dots.tts` imports as `dots_tts` — the stub directory must match the *import* name |
| Model class + constructor signature | Device arg? HF id/path arg? What constants it exposes at module level |
| Default HuggingFace checkpoint | Becomes the `<NAME>_MODEL` env default and the `model` field in capabilities |
| `generate()` inputs | text; prompt/reference audio (bytes, file path, or array?); reference transcript (required / optional / absent?); language (codes? names? free-form?); seed (argument, or set RNG manually?) |
| Output format | tensor shape (`(1, N)` vs `(N,)`), float range, and **sample rate** — read the engine's own constants, never assume 24 kHz |
| Watermarking? | drives the `watermarked` flag (Chatterbox applies PerTh inside the library) |
| Language table | build `Literal[tuple(...)]` from it (dict insertion order = stable enum); free-form or none → `str \| None` field and `languages=None` in capabilities |
| Audio constraints | minimum usable duration (→ `MIN_PROMPT_DURATION_S`), truncation behavior (e.g. Chatterbox's first-10 s), does it demand a **file path** for prompt audio? |
| Thread safety | shared mutable model state between calls? → needs a `_synthesis_lock` (see the comment in `server_chatterbox.py`) |
| Device support | explicit cuda/mps/cpu arg (→ `<NAME>_DEVICE` env var) vs auto-select (→ mirror `torch.cuda.is_available()` like dots.tts; no env var, and the snapshot test must skip `device`) |

## Phase 2 — Server script: `impl/server_<name>.py`

Follow the section layout of `server_chatterbox.py` exactly:

1. **Module docstring** — what it does, every env var with its default, the
   `pip install <engine-pkg> fastapi uvicorn loguru soundfile` + tts-engine-common line, usage.
2. **Config** — `os.getenv` at **import time** (module level). Fail-fast `_validate_config()`
   for enumerated values (device, model name).
3. **Constants** — `SEED_MIN`/`SEED_MAX` (all current servers use 1–1000),
   `MIN_PROMPT_DURATION_S`, `MAX_AUDIO_B64_LEN = 10_000_000`.
4. **`SynthesisRequest`** — `model_config = ConfigDict(extra="forbid")` (non-negotiable, D5).
   Core fields first (`text` with a whitespace-rejecting validator, `audio_base64`,
   `reference_text` *if* the engine uses a transcript, `language`, `seed`), then
   engine-specific knobs whose defaults mirror the engine's own defaults.
   A core field the engine can't use is **omitted entirely** (Chatterbox has no
   `reference_text`) — never stubbed with a dead `None`.
5. **`SynthesisResponse(CoreSynthesisResponse)`** — add `fid: str` plus any engine extras you actually report back.
6. **`HealthResponse`** — `status`, `serverType`, `model` (label string), `device`.
7. **`CAPABILITIES = build_capabilities(...)`** — stable engine slug (e.g. `"dots.tts"`),
   model label, device, the **real** sample rate, `watermarked`, `reference_audio` spec
   (required/formats/min_duration_s/note), `languages` (list or `None`), `overrides`
   (UI sugar — `step`, `advanced` — only; overrides that contradict validation are rejected at import).
8. **`app` + `lifespan`** — load the model at startup; on shutdown `del` the model and
   `torch.cuda.empty_cache()`.
9. **`/capabilities` route** — `app.add_api_route("/capabilities", capabilities_endpoint(CAPABILITIES), ...)`.
10. **Runtime** — dataclass + lazy `_get_runtime()`; `_synthesis_lock = threading.Lock()`
    where the model has shared mutable state.
11. **Global exception handler** → 500 with a message (copy verbatim).
12. **Endpoints** — `/` (HTML landing page), `/health`, `/synthesize`:
    `decode_base64` → `_check_reference_audio` (header-only `sf.info`, 400s) → seed resolution →
    timed `generate()` under the lock → `compute_rtf` → `_numpy_to_wav_bytes` (clips to [-1, 1]) →
    500 with message on failure.
13. **Helpers** — `seed_everything`; `_write_temp_audio`/`_cleanup_temp` only if the engine
    demands a file path; `_numpy_to_wav_bytes`.
14. **`__main__`** — `uvicorn.run(app, host=<NAME>_HOST (default 0.0.0.0), port=<NAME>_PORT (default 8000))`.

Env var prefix convention: engine name in caps with underscores (`QWEN3TTS_DEVICE`,
`DOTS_TTS_MODEL`), plus `*_HOST` / `*_PORT`.

## Phase 3 — Test stub: `impl/tests/stubs/<import_name>/`

Import-only stand-in so the suite runs on a bare dev box:

- `__init__.py`: one-line docstring.
- One file per import path the server uses (e.g. `chatterbox/mtl_tts.py` for `from chatterbox.mtl_tts import ...`).
- **Faithful copies of every engine constant the server imports at module level**
  (sample rate, model tables, language dicts) — the import-time config validation and the
  dynamically built `Literal` depend on them matching the real package. Verify against the
  real engine source, don't guess.
- Class placeholder(s) whose `from_pretrained`/`generate` raise `NotImplementedError`.
- Stubs are appended to `sys.path` last, so on a GPU box the real package always wins.

## Phase 4 — Test file: `impl/tests/test_server_<name>.py`

Copy `test_server_chatterbox.py`'s structure:

- `client` fixture: `TestClient(srv.app)` **without** a context manager — entering it
  would run the lifespan (model load).
- `/capabilities`: exact snapshot match via `helpers.load_snapshot("<slug>_capabilities.json")`
  plus targeted assertions (core fields required, knob defaults, enum == engine's table).
- `/health` + landing page.
- 422 validation battery: unknown field, `{}`, empty text, whitespace-only text, empty
  audio, bad enum member, out-of-range for each numeric field. For boolean fields use
  **non-coercible** values — Pydantic v2 lax mode coerces `"yes"`/`"true"`/`"1"`.
- 400 pre-flight with a `fake_runtime` fixture (monkeypatch `srv._runtime` to a
  `types.SimpleNamespace(sample_rate=..., device=...)`): undecodable audio, and a
  `helpers.make_wav_bytes(...)` clip shorter than the minimum.
- Machine-dependent device (auto-select engines): compare the snapshot with `device`
  popped from both sides (the dots.tts pattern) and only assert it is in `("cuda", "cpu")`.

## Phase 5 — Registration

1. `impl/tests/_bootstrap.py` — pin the new env vars to the documented defaults
   (`os.environ["<NAME>_DEVICE"] = "cuda"`, `os.environ.pop("<NAME>_MODEL", None)`).
   This file is shared by `conftest.py` and `update_snapshots.py`; keep it in sync with
   the server docstring.
2. `impl/tests/update_snapshots.py` — add `(server_<name>, "<slug>_capabilities.json")` to `ENGINES`.
3. Regenerate and review: `python impl/tests/update_snapshots.py`, then
   `git diff impl/tests/snapshots/`. Never hand-edit a snapshot — capabilities are derived
   from the Pydantic model (D4).
4. Tables: `impl/README.md` (file/engine/rate/device-env/model-env row + an engine-notes
   bullet) and the root `README.md` Engines table.

## Verify

```bash
python -m pytest tts-engine-common/tests/ impl/tests/   # works with no torch/GPU/engine installed
```

## Gotchas

- Sample rate: two of the four current servers are already 48 kHz surprises waiting to
  happen — always read the engine's own constant.
- `tts-engine-common` must stay torch-free — never import engine or torch symbols there (D7).
- If the engine's in-context mode hard-requires a transcript, mirror Qwen3-TTS's fallback
  (e.g. speaker-embedding-only mode) rather than making `reference_text` required.
- If the engine transcribes the reference itself (OmniVoice's Whisper path), note the lazy
  ASR-model load in the `reference_audio.note` and the docstring.
- The stub `numpy` exists to satisfy pytest's own introspection; if a cross-suite run dies
  with `AttributeError: module 'numpy' has no attribute ...`, see `stubs/numpy/__init__.py`.
