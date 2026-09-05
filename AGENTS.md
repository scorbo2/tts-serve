# AGENTS.md

## Commands (run from repo root)

- Full test suite (no torch, GPU, or engine packages needed — verified on a bare dev box):
  `python -m pytest tts-engine-common/tests/ impl/tests/`
- Single test: plain pytest, e.g. `python -m pytest impl/tests/test_server_omnivoice.py -k seed`
- After changing a server's request schema or engine constants, regenerate snapshots and review the diff:
  `python impl/tests/update_snapshots.py`
- `tts-engine-common` does **not** need to be installed for tests — its conftest (and `impl/tests/_bootstrap.py`) add `src/` to `sys.path`.
- No lint, typecheck, formatter, lockfile, CI, or pre-commit is configured. Don't go looking for it or invent one.
- Actually running a server requires the engine package + a GPU and downloads model weights from HuggingFace on first start. The test suite never does this.

## Layout

- `tts-engine-common/` — shared FastAPI + Pydantic package. Design constraint: **no torch, no engine deps** — it must stay importable and testable on any box (see `pyproject.toml` comment and `docs/01-server-generification.md` D7).
- `impl/` — four standalone FastAPI server scripts, one per engine: `server_chatterbox.py`, `server_omnivoice.py`, `server_qwen3TTS.py`, `server_dotsTTS.py`. Run via `python impl/server_<name>.py` or uvicorn; env config is documented in each module's docstring.
- `impl/tests/` — GPU-free test suite + committed `/capabilities` snapshots (`snapshots/`).
- `docs/` — design docs; `01-server-generification.md` contains the binding decisions (D1–D7).

## Architecture facts that change how you work

- The Pydantic `SynthesisRequest` model is the single source of truth: `GET /capabilities` is **derived** from it at import time via `build_capabilities()` (decision D4). Change the model, then regenerate the snapshot — never hand-edit the capabilities doc or a snapshot to match.
- Request models use `extra="forbid"` deliberately (D5): unknown fields must fail loudly with 422. Don't "fix" that into `extra="ignore"`.
- Servers read env config at **import time** (module-level `os.getenv`). Tests pin env in `impl/tests/_bootstrap.py` *before* any server import — that file is shared by `conftest.py` and `update_snapshots.py`, so keep the pinned values in sync with the servers' documented defaults.
- Core request vocabulary: `text`, `audio_base64`, `reference_text`, `language`, `seed` (`CORE_FIELDS`). Core response: `audio_base64`, `sample_rate`, `seed`, `time_used`, `rtf` plus per-server extras (e.g. `fid`). `language` is *not* guaranteed core — check `/capabilities`.
- The `language` contract (docs/02-language-handling.md): **two-letter lowercase codes** only; null/empty normalizes to `en` at the request boundary (shared helpers in `tts_engine_common.language` — reuse them, don't re-implement). Engines with a different internal format (Qwen3-TTS lowercase names, dots.tts uppercase codes) map codes at their own server; auto-detection is exposed as the special value `auto`. This is why `schema_version` is 2 — don't "fix" servers back to accepting names/free-form.
- dots.tts outputs **48 kHz** (the other three are 24 kHz) and auto-selects CUDA/CPU (no device env var). Its snapshot's `device` field is machine-dependent; the test deliberately compares everything *except* `device`. Don't "fix" a device mismatch.

## Test gotchas

- `impl/tests/stubs/` provides import-only stand-ins for `torch`, `numpy`, `soundfile`, `loguru`, and each engine package. Stubs are appended to `sys.path` **last**, so real installed packages always win on a GPU box.
- The stub `numpy` must satisfy pytest's own introspection (`isscalar`, `bool_`, `ndarray`, `asarray`). If a cross-suite run dies with `AttributeError: module 'numpy' has no attribute ...`, see the comment in `stubs/numpy/__init__.py`.
- Pydantic v2 lax mode coerces `"yes"`/`"true"`/`"1"` to booleans — boolean-rejection tests must use values that are *not* coercible.
- `impl/tests/test_language_contract.py` asserts the shared docs/02 language contract once across *all* servers (null/empty → `en`, 422 for garbage/uppercase/names/non-strings, `auto` per declaration, capabilities default). A new server must be added to its `SERVERS` list; per-server suites keep only engine-specific language tests.
- The suite covers the HTTP surface only (capabilities, health, validation, 400/422 pre-flight). Actual synthesis and the success path of `/synthesize` are **not** tested (they need a real model + GPU).

## Per-engine quirks

- **Chatterbox** — no `reference_text` field (deliberate; it conditions on audio only). Only the first 10 s of the reference are used. Output is PerTh-watermarked by the library.
- **OmniVoice** — omitting `reference_text` triggers on-the-fly Whisper transcription; the first such request pays the ASR model load. References over 20 s are trimmed at the largest silence gap.
- **Qwen3-TTS** — `language` takes two-letter codes (`en`, `zh`) or `auto`; the server maps codes to the engine's lowercase *names* internally (`en` → `english`). Omitting `reference_text` transparently enables speaker-embedding-only mode.
- **dots.tts** — the runtime demands a file path for prompt audio, so the server writes a temp file; reference transcript is optional. `language` takes two-letter codes or `auto`; the server maps them to the engine's uppercase codes / `auto_detect` internally.

## Adding a new engine

Copy the existing pattern: `impl/server_<name>.py` (config + deps in the module docstring, `extra="forbid"` request model, capabilities derived at import) + `impl/tests/test_server_<name>.py` + a snapshot generated by `update_snapshots.py`. See `impl/README.md` for the current parameter table and `docs/00-project-overview.md` Goal 3 for the intended workflow. In OpenCode, the `new-tts-engine` skill (`.agents/skills/new-tts-engine/SKILL.md`, registered via `opencode.json` `skills.paths`) walks through the full workflow including the test stub.
