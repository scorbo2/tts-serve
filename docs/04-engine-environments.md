# Engine environments (resolving venvs between tts-serve and engine projects)

Every server in `impl/` runs in a single Python environment that holds two
separate sets of dependencies:

- **the engine**: the TTS package itself (`omnivoice`, `chatterbox-tts`, ...),
  along with torch, torchaudio, CUDA wheels and the engine's own stack;
- **tts-serve**: `tts-engine-common` plus `fastapi`, `uvicorn`, `loguru` and
  `soundfile`.

This document specifies how to build that environment when the engine is a git
checkout next to tts-serve, without global installs and without breaking the
engine project's own venv.

## Current state

Each `impl/server_<name>.md` has an Installation section with the same recipe:
create a venv (`python3 -m venv .venv`), `pip install <engine>` from PyPI, then
`pip install ./tts-engine-common fastapi uvicorn loguru soundfile`. IndexTTS is
different: it runs `uv sync --all-extras` in the engine's clone and pip-installs
the tts-serve dependencies into that venv. LuxTTS is git-only.

This recipe is still valid and stays supported. The problems below only show up
on some machines, or when working from a git checkout.

## The problem

1. **Bare `pip install` is refused.** Ubuntu's system Python and uv-managed
   Pythons are marked externally managed (PEP 668). Unless a venv is active,
   `pip install` fails with `externally-managed-environment`. An active venv
   isn't always enough: venvs created by `uv venv` / `uv sync` don't contain pip,
   so `pip` on the `PATH` is still the system one and fails the same way.
2. **The engine project's venv isn't ours to extend.** Installing the tts-serve
   packages into the engine's own `.venv` works until the next `uv sync` in the
   engine repo. That sync removes every package the engine's lockfile doesn't
   declare, which includes `tts-engine-common`, `uvicorn` and the rest, and the
   server stops starting.
3. **Venvs can't be stitched together.** Pointing one environment at another
   venv's `site-packages` (via `PYTHONPATH` or `.pth` files) breaks as soon as
   the two were built for different interpreters. Compiled extensions are tied
   to one Python ABI (for example cp312 vs cp313).
4. **The right torch build is hardware-specific, and PyPI doesn't encode it.**
   Engine repos often select a CUDA wheel index in `[tool.uv.sources]`. That
   setting is uv-only project metadata, so `pip install <engine>` from PyPI
   ignores it and gets PyPI's torch. On **aarch64 Linux** (NVIDIA GB10 /
   DGX Spark, Jetson Thor, GH200) this is worse: there are no cu128 CUDA wheels
   at all, so CUDA builds start at torch 2.9 on the cu130 index.
5. **Running from a checkout is a use case of its own.** People want to run an
   engine at its latest commit or with local patches, and have a `git pull`
   show up in the server without reinstalling.

## Proposal

tts-serve gets optional **per-engine environment projects** at
`envs/<engine>/`. Each one is a small uv project that describes exactly one
venv for exactly one server, and builds it with `uv sync`. The engine checkout
and its own venv are left alone.

### Layout and files

```
<parent>/
├── tts-serve/
│   └── envs/<engine>/
│       ├── pyproject.toml     committed
│       ├── .python-version    committed
│       ├── .venv/             gitignored (existing `.venv/` rule)
│       └── uv.lock            gitignored (`envs/*/uv.lock`)
└── <EngineRepo>/              engine git checkout, next to tts-serve
```

- `<engine>` is the lowercase engine name matching the server script, for
  example `envs/omnivoice/` for `impl/server_omnivoice.py`.
- Only `pyproject.toml` and `.python-version` are committed.
- **Sibling-checkout convention:** the engine is cloned next to tts-serve, so its
  path from `envs/<engine>/` is `../../../<EngineRepo>`. A user whose checkout is
  somewhere else edits that one path locally.

### `pyproject.toml` rules

- **Virtual project.** Set `[tool.uv] package = false`. The file describes a
  venv, not something installable. `[project]` gets a
  `tts-serve-env-<engine>` name, a version, and `requires-python`.
- **Dependencies.** List the engine, `tts-engine-common`, `fastapi`, `uvicorn`,
  `loguru` and `soundfile`. List **`torch` and `torchaudio` explicitly** too:
  uv only applies `tool.uv.sources` to requirements that are declared directly,
  not to transitive ones.
- **Sources.**
  - The engine is a path source to the sibling checkout with
    `editable = true`. An engine that is only wanted from PyPI can omit the
    source.
  - `tts-engine-common = { path = "../../tts-engine-common", editable = true }`.
  - `torch` and `torchaudio` point at CUDA index(es) through `marker` lists.
- **Indexes.** Every PyTorch index is declared with `explicit = true`, so only
  the packages whose sources name it are fetched from it. Without that, numpy
  and friends would come from the PyTorch index too. **Reuse the index names the
  engine's own pyproject uses** (see "Decisions" for why).
- **Torch pins are `override-dependencies`, not `constraint-dependencies`.**
  Start from the engine's own pins and add an aarch64 Linux entry (cu130,
  torch ≥ 2.9) wherever the engine only knows cu128.
- **Server script.** A `[tool.tts-serve]` table names the server this env
  runs, as a path relative to the repo root:
  `server = "impl/server_<name>.py"`. uv ignores tables under `tool.*` that
  aren't its own. The launcher reads this key because server script names don't
  follow env names (`server_qwen3TTS.py`, `server_fasterQwen3TTS.py`, ...).
- **Env-var prefix.** `env-prefix = "<PREFIX>"`, in the same table, gives the
  prefix of the server's `<PREFIX>_HOST` / `<PREFIX>_PORT` variables
  (`OMNIVOICE`, `CHATTERBOX`, `DOTS_TTS`, `QWEN3TTS_MLX`, ...). The launcher
  needs it for `--host` and `--port`, because these prefixes don't follow env
  names either.
  It is uppercase letters, digits and underscores, starting with a letter.
- **Header comment.** Say what the file is for, the commands to use it, the
  sibling-checkout convention, how to point at a checkout elsewhere, and why the
  torch pins are overrides.

### `.python-version`

Set it to the minor version of the interpreter the engine's own venv uses. uv
then finds the same wheels (torch alone is several GB) in its cache and
hardlinks them instead of downloading them again.

### Usage

The simplest way is the launcher (next section):

```
python3 tools/serve.py <engine>          # sync the venv if needed, then start the server
```

or directly with uv, from the tts-serve repo root:

```
uv sync --project envs/<engine>                                  # build or update the venv
uv run --project envs/<engine> python impl/server_<name>.py     # run the server
envs/<engine>/.venv/bin/python impl/server_<name>.py            # same, without uv (offline)
```

The server's `<ENGINE>_*` environment variables work unchanged.

### Launcher: `tools/serve.py`

One command starts any engine that has an env, and runs `uv sync` only when
the env actually needs it.

```
python3 tools/serve.py omnivoice                 # sync if needed, then start the server
python3 tools/serve.py omnivoice --sync          # always run uv sync first
python3 tools/serve.py --list                    # every env, its server, and its venv status
python3 tools/serve.py chatterbox --port 7501    # run next to another engine on 7500
python3 tools/serve.py omnivoice --host 127.0.0.1   # local-only (default is 0.0.0.0)
OMNIVOICE_PORT=8500 python3 tools/serve.py omnivoice   # server env vars still work
```

Every server defaults to port 7500, so running two engines at once requires
at least one override.

**Constraints**

- Standard library only, like `speak.py`, so it runs on the system `python3`
  with nothing installed.
- Needs Python 3.11+ for `tomllib`. On an older Python it fails with a clear
  error, not a traceback.
- Works from any working directory. All paths are resolved from the script's
  own location.

**Arguments**

- `ENGINE` (positional): the name of a directory under `envs/` that contains a
  `pyproject.toml`. An unknown name is a usage error (exit 2) that lists the
  available envs. `ENGINE` may be omitted only with `--list`.
- `--sync`: run `uv sync` before starting, even if the venv looks up to date.
- `--port PORT`: the port the server binds, an integer from 1 to 65535
  (anything else is a usage error, exit 2). The launcher passes it to the
  server as `<env-prefix>_PORT`, overriding any value already set in the
  environment, so the flag always wins. If the env's pyproject has no
  `env-prefix`, `--port` fails with exit 1.
- `--host HOST`: the address the server binds; every server defaults to
  `0.0.0.0`. Use `127.0.0.1` for a local-only server. It works like `--port`:
  it's passed as `<env-prefix>_HOST`, the flag wins over the environment, and
  it fails with exit 1 without `env-prefix`. The value must be non-empty and
  contain no whitespace (otherwise exit 2). Anything else, such as an address
  the machine doesn't have, is left for the server to reject when it binds.
- `--list`: print each env with its server script and venv status, then exit 0.
  The status is one of:
  - `ready`;
  - `not built`;
  - `needs sync (<reason>)`;
  - `error: <message>`, for a broken env definition.

**When to sync.** The launcher runs `uv sync --project envs/<engine>` when any
of these is true:

1. `--sync` was given.
2. The venv's Python doesn't exist yet: `.venv/bin/python`, or
   `.venv/Scripts/python.exe` on Windows.
3. The venv has no sync stamp. After every successful sync, the launcher
   touches `envs/<engine>/.venv/.tts-serve-synced`, which is gitignored along
   with the rest of the `.venv`. A venv built by a manual `uv sync` has no
   stamp, so the launcher syncs it once.
4. One of the env's inputs is newer than the stamp. The inputs are the env's
   `pyproject.toml`, its `.python-version`, and the `pyproject.toml` of every
   local path source in `[tool.uv.sources]` (the engine checkout and
   `tts-engine-common`). A `git pull` that changes the engine's dependencies
   therefore triggers a sync on the next start. A pull that only changes code
   doesn't, because editable installs pick up code changes by themselves.

It prints `Syncing envs/<engine> (<reason>) ...` before each sync. It removes
`VIRTUAL_ENV` from uv's environment, so an unrelated activated venv doesn't
produce uv's "does not match the project environment" warning.

**Sync failures**

| Situation | Result |
|---|---|
| `uv` isn't on `PATH` and a sync is needed | Exit 1 with a pointer to the uv install docs. When no sync is needed, uv isn't required at all. |
| Sync fails, the venv exists, and `--sync` wasn't given | Print a warning and start with the existing venv. This keeps the server startable offline. The stamp isn't touched, so the next start tries again. |
| Sync fails with `--sync` given, or the venv doesn't exist | Exit 1. |

**Start.** The launcher prints
`Starting <server> with <venv python>`. When `--host` or `--port` is given, it
appends the variables it set, e.g. ` (OMNIVOICE_HOST=127.0.0.1, OMNIVOICE_PORT=7501)`.
It flushes stdout, then **replaces itself** (`os.execve`)
with `<venv python> <server>`. It doesn't go through `uv run`, which
re-resolves on every start (see "Known quirks"). The working directory and
environment variables are inherited unchanged, except for the
`<PREFIX>_HOST` / `<PREFIX>_PORT` that `--host` / `--port` set. So relative paths in `<ENGINE>_*`
variables behave exactly as when the script is run directly. Because the
process is replaced, Ctrl+C goes straight to uvicorn.

**Errors.** These exit 1 with an `Error:` message on stderr:

- a malformed `pyproject.toml`;
- a missing or non-string `[tool.tts-serve] server`;
- a server script that doesn't exist;
- a malformed `env-prefix`;
- `--host` or `--port` for an env without `env-prefix`. The message names the
  flags that were given, and nothing is synced first;
- a failed `exec`.

**Tests.** `tools/tests/test_serve.py`, GPU-free and without network access.
The tests build fake repos in `tmp_path` and stub `uv`, `subprocess` and
`os.execve`. One test also checks the committed `envs/*/pyproject.toml` files.
Each must name a server script that exists, and an `env-prefix` whose
`<PREFIX>_HOST` and `<PREFIX>_PORT` that script actually reads, so the prefix
can't drift away from the server.

### Boundaries

- **Optional.** The pip recipe in each `impl/server_<name>.md` stays the
  default. The env route is documented next to it as an alternative.
- **D7 is untouched.** `tts-engine-common` still has no torch or engine
  dependencies. Torch appears only in `envs/<engine>/pyproject.toml`.
- **Not used by the test suite.** The GPU-free suite keeps running against the
  stubs in `impl/tests/stubs/`, and needs no env.
- **No committed lockfiles.** AGENTS.md says no lockfile is configured, and a
  lock over an editable engine checkout would change with every commit to that
  checkout.

## Implementation (2026-09)

`envs/omnivoice/` was the first environment. It was verified on aarch64
(NVIDIA GB10, uv 0.9.26, CPython 3.13.11):

- `uv sync --project envs/omnivoice` finished in about 4 s from the uv cache.
  It installed `torch 2.9.1+cu130`, and `torch.cuda.is_available()` returned
  True.
- `omnivoice` and `tts-engine-common` both import from their checkouts (editable
  installs).
- The live `GET /capabilities` output was identical to
  `impl/tests/snapshots/omnivoice_capabilities.json`.
- `tools/speak.py` synthesized 24 kHz 16-bit mono WAVs. The first request paid
  the Whisper load (no transcript given); a warm request ran at RTF 0.47.
- Resolution also works against an **unmodified upstream** OmniVoice checkout.
  Its pyproject sends all Linux torch to cu128.

Other changes: `.gitignore` gained `envs/*/uv.lock`,
`impl/server_omnivoice.md` gained an "Alternative: from a local OmniVoice git
checkout (uv)" section, and `README.md` and `AGENTS.md` mention `envs/`.

Decisions taken while implementing:

- **Overrides, not constraints.** uv also reads the `tool.uv.sources` of a path
  dependency. Upstream OmniVoice maps all Linux torch to the cu128 index. With
  constraints, that requirement sits next to ours and resolution fails:
  `Requirements contain conflicting indexes for package torch in split
  platform_machine == 'aarch64' and sys_platform == 'linux'`. An override
  *replaces* every torch requirement in the graph, including the checkout's, so
  our index wins and an unmodified checkout works.
- **Same index names as the engine.** Matching names mean the root's sources and
  the checkout's sources agree wherever the engine's mapping is already correct
  (for example cu128 on x86_64 Linux).
- **Editable engine install.** A `git pull` in the checkout takes effect on the
  next server start. Nothing gets reinstalled, except when the engine's
  dependencies change, and then `uv sync` picks that up.
- **Python pinned to 3.13** to match OmniVoice's venv, so the cp313 aarch64
  torch wheels come from the cache.

Known quirks (harmless, recorded so nobody chases them):

- **Every `uv sync` / `uv run` re-resolves.** `uv lock --check` always reports
  the lock as out of date for this project, whether the pins are overrides or
  constraints. The cause hasn't been pinned down. Resolving from cache takes
  about 100 ms, but once uv's HTTP cache expires it needs index access. Offline,
  run the venv's Python directly.
- **`nvidia-cusparselt-cu13` is reinstalled on every sync.** OmniVoice's own
  venv does the same, so this isn't specific to envs.
- **torch 2.9.1 warns about the GB10** (CUDA capability 12.1 vs a supported
  maximum of 12.0). Synthesis works anyway.

### Launcher (2026-09)

`tools/serve.py` and `tools/tests/test_serve.py` implement the "Launcher"
section above. `envs/omnivoice/pyproject.toml` gained
`[tool.tts-serve] server = "impl/server_omnivoice.py"`. It was verified on the
GB10 with the system Python 3.12:

- The first start synced once. The existing venv had been built by a manual
  `uv sync`, so it had no stamp.
- The second start reported `ready` and went straight to the server.
- `OMNIVOICE_PORT=7501` reached the server.
- After `--port` was added, `serve.py omnivoice --port 7503` and
  `serve.py chatterbox --port 7502` ran side by side, next to another OmniVoice
  server on the default 7500. Both answered `/health`.
- `serve.py chatterbox --host 127.0.0.1 --port 7502` bound to loopback only:
  `127.0.0.1:7502` answered, and the machine's other interface address was
  refused.
- The server's parent process was the calling shell, which confirms the
  launcher replaced itself rather than staying around as a wrapper.

Decisions taken while implementing:

- **A stamp file rather than `uv sync --check`.** A check re-resolves, which is
  the per-start cost and network dependency the launcher is meant to avoid (see
  "Known quirks"). Comparing mtimes is local and instant.
- **Watch the path sources' `pyproject.toml` files, not their code.** Editable
  installs pick up code changes without a sync; only dependency changes need
  one.
- **Keep the caller's working directory.** IndexTTS's `INDEXTTS_MODEL_DIR`
  defaults to the relative `checkpoints`, so a `chdir` would silently change
  what it points at.
- **Never resolve the venv's `python` symlink.** Running the base interpreter
  it points to would bypass the venv entirely.

### Chatterbox (2026-09)

`envs/chatterbox/` builds the Chatterbox server from a sibling
`resemble-ai/chatterbox` checkout. For this engine the checkout is **required**:
`impl/server_chatterbox.py` imports `MULTILINGUAL_T3_MODELS` (the v2/v3 T3
checkpoints). It was added in resemble-ai/chatterbox#516 on 2026-05-01, ahead
of the v3 multilingual release on 2026-06-10. The latest PyPI release, 0.1.7
from March 2026, doesn't have it. The checkout's own `pyproject.toml` also
says 0.1.7, so the version number can't tell the two installs apart. The
documented `pip install chatterbox-tts` therefore fails at server startup with
an ImportError. `impl/server_chatterbox.md` now installs from git instead.

It was verified on the GB10 with the launcher:

- `python3 tools/serve.py chatterbox` built the venv in about 3 s, with torch
  from the uv cache and `resemble-perth` built from git.
- The first start, including the weight download, took about 40 s. The model
  loaded was `chatterbox-multilingual-v3` on CUDA.
- The live `/capabilities` output was identical to the committed snapshot.
- `tools/speak.py` produced 24 kHz clips in English and French, at RTF
  0.52–0.56 once warm. Whisper transcribed all of them back word for word,
  which confirms that torch 2.9.1 (instead of the engine's pinned 2.6.0) gives
  correct output.

Decisions taken while implementing:

- **Keep the engine's torch pin wherever PyPI can serve it.** Unlike OmniVoice,
  Chatterbox has no index settings of its own. It pins `torch==2.6.0` and
  relies on PyPI, whose x86_64 Linux wheels include CUDA 12.4. So the overrides
  keep 2.6.0 everywhere except linux-aarch64, where no CUDA build of 2.6.0
  exists; that platform gets 2.9.1 from cu130. Only one index is declared
  (`pytorch-cuda-sbsa`).
- **Overrides are still required,** because the engine's `==2.6.0` has to be
  *replaced* on aarch64. A constraint can only narrow a requirement, so it
  couldn't turn 2.6.0 into 2.9.1.
- **`requires-python = ">=3.10,<3.14"`.** Chatterbox pins torch 2.6.0 only
  below Python 3.14, and has no tested pin for 3.14. Python is pinned to 3.13,
  like OmniVoice: numpy 2.x on 3.13, and the same cached cp313 torch wheels.
- **torchaudio 2.9 is safe for this engine.** Chatterbox only uses
  `transforms.Resample` and `compliance.kaldi.fbank`, not the `load`/`save`
  functions that moved to TorchCodec in 2.9.

Side effect worth knowing: on first use, the engine downloads a spaCy/pkuseg
segmentation model to `~/.pkuseg/`, outside the HuggingFace cache.

## Adding an environment for another engine

1. Clone the engine next to tts-serve. If it has its own venv, note that venv's
   Python version and torch build.
2. Copy `envs/omnivoice/` to `envs/<engine>/`. Change the project name, the
   `[tool.tts-serve]` `server` and `env-prefix`, the engine requirement, and its path
   source. If the engine's install needs
   extras or git-only dependencies (IndexTTS's `--all-extras`, LuxTTS's
   git-only `linacodec`), express them here: extras on the engine requirement,
   git sources for the rest.
3. Read the engine's own `pyproject.toml`. Copy its torch index names, URLs and
   pins, turn any `constraint-dependencies` into `override-dependencies`, and
   add the aarch64 Linux cu130 entry if the engine lacks one. If the engine has
   no index settings and just pins a PyPI torch (as Chatterbox does), keep that
   pin where PyPI has a usable build and override only linux-aarch64 (see the
   Chatterbox record above). Also check that the engine doesn't rely on
   torchaudio's `load`/`save`, which moved to TorchCodec in 2.9. An engine
   without CUDA torch (for example Qwen3-TTS MLX on Apple Silicon) needs no
   torch sources or indexes.
   Also compare the engine's PyPI release with what the server imports. PyPI
   can lag behind git (Chatterbox did), and then a checkout is the only
   working install.
4. Set `.python-version`.
5. Verify, following the same checks as the OmniVoice implementation above:
   - `python3 tools/serve.py <engine>` syncs the venv and starts the server
     (or run `uv sync --project envs/<engine>` directly);
   - torch imports with the expected version, and CUDA is available where
     expected;
   - the engine imports from the checkout;
   - the live `/capabilities` output matches the committed snapshot (except
     `device` for dots.tts);
   - `tools/speak.py` synthesizes audio;
   - the full test suite still passes.
6. Document it: add an "Alternative: from a local <Engine> git checkout (uv)"
   section to `impl/server_<name>.md`, and list the new directory in the `envs/`
   bullet of `AGENTS.md`.

Done so far: OmniVoice and Chatterbox. Remaining candidates, from the current
install docs:

| Engine | Current install | Why an env would help |
|---|---|---|
| IndexTTS | `uv sync --all-extras` in the engine clone, tts-serve deps pip-installed into that venv | Problems 1 and 2: that uv venv has no pip, so the documented `pip install` reaches the system pip; and even when the deps are installed, the next `uv sync` there removes them |
| LuxTTS | git-only, plus a git-only dependency | No PyPI route exists, so a checkout is the only option |
| Qwen3-TTS, faster-qwen3-tts, dots.tts, VoxCPM | PyPI | Needed for problem 4 (hardware-specific torch), to run from a checkout, or if PyPI lags behind what the server imports (check this first) |
| Breeze TTS 2 | git-only, no packaging: `pip install -r requirements.txt` (torch 2.9.1) in the clone, server run with the repo on `PYTHONPATH` | Would pin the right torch build, but the checkout has no `pyproject.toml`, so it can't be an editable path dependency; the env would install its requirements and `tools/serve.py` would have to set `PYTHONPATH` |
| Qwen3-TTS MLX | PyPI (`mlx-audio`) | Low value: no CUDA torch involved |

## Open questions

- **Checkout outside the sibling layout.** Today the user edits the committed
  path locally, which then shows up as a modification in `git status`. If that
  becomes common, a non-committed override mechanism may be worth adding.
- ~~**Shared launcher.**~~ Resolved: `tools/serve.py` (see "Launcher").
