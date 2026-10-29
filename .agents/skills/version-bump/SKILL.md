---
name: version-bump
description: Use when starting, finalizing, or tagging a tts-serve release, or whenever a change breaks backwards compatibility of the request/response contract or the capabilities document. Covers the release-version bump workflow, the schema-version bump workflow, and why those two numbers are deliberately independent.
---

# Bumping versions in tts-serve

tts-serve carries **two independent version numbers** plus one innocent
bystander. Confusing them breaks version-gating clients or spawns release
notes nobody can verify. Read this before touching any of them.

## The numbers

| Constant | Where | Type | Example | Bumped when |
|---|---|---|---|---|
| **Release line** | `RELEASE_VERSION` in `tts-engine-common/src/tts_engine_common/core.py` | `str` | `"1.5"` | **Every** release cycle, at its *start*, backwards-compatible or not |
| **Client contract** | `SCHEMA_VERSION` in the same file | `int` | `2` | **Only** on breaking changes to the request/response contract or capabilities doc (docs/01 §3.5) |
| Package version | `version` in `tts-engine-common/pyproject.toml` + `__version__` in `__init__.py` | `str` | `"0.1.0"` | Independent schedule; historically never bumped. **Not** the release line — never wire it into `/health` |

Why two, you ask, when one would involve 50% fewer numbers: they answer
different questions. `RELEASE_VERSION` answers *"which code is deployed?"*
(support/bug-report question; changes constantly, means nothing to clients).
`SCHEMA_VERSION` answers *"will this server speak my language?"* — clients
gate on it: `tools/speak.py` hard-aborts on any value other than its own
`SUPPORTED_SCHEMA_VERSION` ("Server returned unknown schema version N,
expected M; aborting."). Bumping the schema version for a *non-breaking*
release would tell every such client to pack its bags — so don't. Conversely,
a breaking release that only bumps the release line hides the breakage from
the one signal clients actually inspect. `RELEASE_VERSION` moves every
release; `SCHEMA_VERSION` moves *only* when clients must know.

## Where the release version surfaces

- `GET /health` → `version` field, on every server (added in 1.5).
- FastAPI app metadata: every `FastAPI(version=RELEASE_VERSION, ...)` —
  visible in `/docs` and `/openapi.json`.
- **Never** in `/capabilities`: that derived document carries only
  `schema_version`. `impl/tests/test_versioning_contract.py` asserts the
  release line does not leak into it.

The old per-server `version="0.1.0"`–`"0.3.0"` stamps (matching no release
anyone could name) are gone; all ten servers import the shared constant.
The `/health` shape itself lives once in `CoreHealthResponse`
(`tts_engine_common.models`) — servers subclass it and override only
`serverType`; don't re-declare `version` per server. Don't reintroduce
hand-maintained stamps.

## Workflow A — starting a release cycle (the common case)

Do this **at the start** of the cycle, in the same commit that opens the
release, on the release's dev branch (e.g. `1.5-dev`):

1. Bump `RELEASE_VERSION` in `tts-engine-common/src/tts_engine_common/core.py`
   to the new line (e.g. `"1.5"`). One constant, ten servers — no other code
   changes; `/health` and the app metadata follow automatically.
2. Add the placeholder entry to the **Release notes** section at the bottom
   of `README.md`, following the existing style:
   ```markdown
   - **Work in progress; add release date when ready** (v1.5)
     - TODO: release notes for v1.5 release go here
   ```
3. Run the suite and confirm green:
   `python -m pytest tts-engine-common/tests/ impl/tests/ tools/tests/`
   (the version tests pin `/health.version` to the *constant*, not a
   literal, so a bump requires **zero** test edits — deliberately.)

## Workflow B — finalizing and tagging the release

At the end of the cycle, on the dev branch:

1. Replace the README placeholder's "Work in progress" with the release
   date, fill in the release notes, and reword the heading to match the
   previous entries exactly (e.g. `**2026-09-27** [v1.4]`).
2. Verify `RELEASE_VERSION` equals the tag you are about to cut — dotted
   numeric, **no `v` prefix** (`1.5`, not `v1.5`). The tag, the README
   heading, and the constant must agree; git history shows tags are the
   source of truth for the format.
3. Merge the dev branch, then tag the merge: `git tag 1.5`.

## Workflow C — a breaking change lands mid-cycle

First ask whether it is actually breaking. Per docs/01 §3.5, these are
**additive** (no `SCHEMA_VERSION` bump — they are what normal releases are
made of):

- a new engine (the entire point of tts-serve),
- a new optional request parameter,
- a new optional field in the capabilities document,
- a new enum value.

These are **breaking** (bump `SCHEMA_VERSION` by exactly 1):

- renaming or removing a request field,
- changing a field's type, bounds semantics, or the `language` contract
  (the v1→v2 bump *was* the language contract — docs/02),
- changing the shape/meaning of the core response fields.

When it is breaking:

1. Bump `SCHEMA_VERSION` in `core.py`; update its history comment with a
   one-line "vN (date) — what changed" entry (mirror the existing v2
   comment).
2. Implement the change across the affected servers.
3. Regenerate snapshots and **review the diff**:
   `python impl/tests/update_snapshots.py && git diff impl/tests/snapshots/`
   (never hand-edit a snapshot — the doc is derived from the Pydantic
   model, D4).
4. Update `SUPPORTED_SCHEMA_VERSION` in `tools/speak.py` (and any error
   strings/tests hard-coding the expected value, e.g.
   `tools/tests/test_speak.py`). Remember: old `speak.py` builds *will*
   refuse the new servers by design — that loud abort is the feature.
5. Note the break prominently in the README release notes for the cycle.
6. A breaking release bumps **both** numbers on its way out: the schema
   version when the change lands, the release line per Workflows A/B.

## Guard rails (what fails if you get it wrong)

- `impl/tests/test_versioning_contract.py` — cross-server: `/health.version`
  and `app.version` carry the shared constant; every server's `HealthResponse`
  subclasses the shared `CoreHealthResponse` core; `/capabilities` carries only
  `schema_version` and must not leak the release line; the constant is
  well-formed (dotted numeric). A **new server** must be added to its
  `SERVER_MODULES` list — an unwired server fails there even if its own
  suite passes.
- `impl/tests/test_server_<name>.py::test_health` — asserts
  `body["version"] == srv.RELEASE_VERSION` per server.
- `tts-engine-common/tests/test_core.py::TestReleaseVersion` — format,
  package-root export, and the independence of the three numbers.

## Gotchas

- Git tags have no `v` prefix. `git tag v1.5` gives you a tag, a README
  heading, and a constant that tell three different stories.
- `tts-engine-common.__version__` (`"0.1.0"`) is the *package* version.
  It has been 0.1.0 since the beginning, it is not the release line, and
  importing it into a server for `/health` would break
  `TestReleaseVersion.test_releaseVersion_isDistinctFromPackageVersion`
  the way all good guard rails do.
- Adding a new server does **not** bump `SCHEMA_VERSION`, even though the
  capabilities snapshot directory grows.
- `/health` gained the `version` field in 1.5 — that itself is additive:
  clients that ignore unknown fields keep working, so no schema bump.
  Schema version stays 2.
- Don't "helpfully" add the release version to `/capabilities`. A number
  that changes on every release is the worst possible thing to let a
  client gate on.
