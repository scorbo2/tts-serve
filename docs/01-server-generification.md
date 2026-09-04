# TTS Server Generification — Development Plan

Status: **PROPOSAL — under review, no code written yet**
Date: 2026-08-30
Scope: All four TTS REST servers (Chatterbox, OmniVoice, Qwen3-TTS, dots.tts) + TalkWithMe (or other client apps).

---

## 0. TL;DR — Key Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Keep the custom per-engine servers.** Do not adopt a universal TTS API standard. | The only de facto standard (OpenAI `/v1/audio/speech`) has a 4-parameter surface; it solves basic portability but nothing about discoverability of engine-specific knobs. |
| D2 | **Stable core contract**: `text` (+ `audio_base64` where the engine clones). Everything else is engine-declared. | The LCM approach failed because an engine's interesting knobs live *outside* the LCM by definition. A universal superset schema would be a swamp of N/A fields. |
| D3 | **`GET /capabilities` per engine**, returning machine-readable parameter metadata (type, bounds, step, enum, default, description, UI group). Unversioned path; version lives in the body (`schema_version`). | Matches the existing unversioned `/health` + `/synthesize` paths; body versioning means no path breakage when the doc evolves. |
| D4 | **Capabilities doc is derived from the Pydantic request model** (`model_json_schema()`), with a small per-server override map for UI sugar (`step`, `group`, `advanced`). | Single source of truth: the same object that validates `/synthesize` produces the discovery doc, so they cannot drift. |
| D5 | **`extra="forbid"` on all request models.** | Pydantic's default silently *drops* unknown fields — i.e., sending `exaggeration` to the OmniVoice server vanishes without a trace. We want loud 422s that name the offending field. |
| D6 | **Client renders engine-specific options generically from metadata** (number→slider, enum→select, …). Zero per-engine UI code. | This is the actual fix for "impossible to slot unique features into the UI." |
| D7 | **Shared package `tts-engine-common`** (pure fastapi + pydantic, no torch) installed into all four servers. | Kills the copy-paste divergence that produced the OmniVoice-paste bugs. One version, independently testable. |

---

## 1. Problem

We run four TTS engines behind hand-rolled FastAPI servers, each with a different generation-parameter surface (e.g. Chatterbox: `exaggeration`, `cfg_weight`, `language`; OmniVoice: `guidance_scale`, `speaker_scale`, `ode_method`, `num_steps`). The client app today speaks a lowest-common-denominator API. Result:

- Engine-unique features are unreachable from the app, or
- They get bolted onto the shared schema ad hoc, polluting every other engine's contract, or
- Fields silently disappear (Pydantic `extra="ignore"` default) when a request lands on the "wrong" engine.

We need each engine to **advertise** its unique configuration surface in a machine-readable, stable way, so the app can render the right options for whichever engine it is connected to — without per-engine UI code and without a shared request schema.

This is the classic *capability discovery / negotiation* pattern (LLM inference servers: OpenAI-compatible core + vendor extras; OpenTelemetry: namespaced vendor attributes; SDP negotiation). It is a solved problem; we just need to implement it.

## 2. Goals / Non-Goals

**Goals**
1. Any engine's full parameter surface is discoverable at runtime, no app redeploy.
2. App renders per-engine options generically from metadata.
3. Discovery doc cannot drift from actual request validation (single source of truth).
4. Misrouted/unknown fields fail loudly at the HTTP boundary.
5. Server-side upgrade path that never bricks the app (versioning + forward-compat rules).

**Non-Goals**
- A universal TTS parameter standard across engines.
- Rewriting the app's request path (it keeps POSTing top-level fields to `/synthesize`).
- Third-party client support via an `extra: dict` passthrough bag (revisit only if an uncontrolled client appears).
- OpenAI-compatible endpoint (tracked as optional M5; it does not serve any goal above).

## 3. Design

### 3.1 Stable core contract

The only fields the app may *assume* exist:

| Field | Type | Present when |
|---|---|---|
| `text` | string | always (required) |
| `audio_base64` | string | `capabilities.reference_audio.required == true` |
| `language` | string (enum) | `capabilities.languages` is non-null |
| `seed` | integer | listed in `capabilities.parameters` |

Response core (already near-uniform across servers; **freeze it**): `audio_base64`, `sample_rate`, `seed` (echoed), `time_used`, `rtf`, plus engine extras (e.g. `fid`).

Note the deliberately small core: `language` is *not* core (Chatterbox Turbo/Nano are English-only), `seed` is *not* core (declare it, don't assume it). The app must treat everything beyond `text` as "render if advertised."

### 3.2 Capability discovery

Each server exposes:

```
GET /capabilities  ->  200 application/json   (static after startup; safe to cache)
```

The document (spec in §4) contains engine identity, output audio facts (sample rate, watermarking), reference-audio requirements, supported languages, and the full parameter table.

### 3.3 Single source of truth

The parameter table is **derived** from the server's Pydantic request model:

- `model_json_schema()` already carries `type`, `minimum`/`maximum`, `exclusiveMinimum`, `enum`, `minLength`/`maxLength`, `default`, `description` for every `Field(...)`.
- A normalizer maps JSON-Schema → `ParamSpec`, including the `int | None` case (`anyOf: [integer, null]`).
- A per-server **override map** layers on UI sugar the schema can't express (`step`, `group`, `advanced`).

```python
OVERRIDES = {
    "seed":         {"group": "common"},
    "exaggeration": {"step": 0.05, "advanced": False},
    "min_p":        {"step": 0.01, "advanced": True},
}
doc = build_capabilities(SynthesisRequest, engine="chatterbox",
                         overrides=OVERRIDES, **ENGINE_META)
```

Startup guard: assert every override key exists in the model's fields (the one realistic drift vector). The rest is structurally drift-proof.

### 3.4 Strictness

```python
class SynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
```

Consequence: a field the engine doesn't know → `422` naming the field. **Pre-work:** audit existing app payloads for fields already being silently dropped today (expected: none, since the app renders from the same LCM — but verify, because this is the one change with teeth).

### 3.5 Versioning & forward compatibility

- Additive changes (new parameter, new optional doc field, new enum value): **no** `schema_version` bump. App rules: ignore unknown doc *fields*; render unknown *widget types* via the raw-JSON escape hatch (§4.3).
- Breaking changes (rename/remove parameter, change type or semantics): **bump** `schema_version`. App on unrecognized version: fall back to minimal mode (`text` + `audio_base64` only) and display "server contract vN — app supports up to vM."

## 4. Capabilities Document Spec (v1)

### 4.1 Example (Chatterbox Multilingual V3)

```json
{
  "schema_version": 1,
  "engine": "chatterbox",
  "model": "chatterbox-multilingual-v3",
  "device": "cuda",
  "sample_rate": 24000,
  "watermarked": true,
  "endpoint": "/synthesize",
  "reference_audio": {
    "required": true,
    "formats": ["wav", "mp3", "ogg", "flac"],
    "min_duration_s": 2.0,
    "max_duration_s": null,
    "note": "First 10 s used for speaker conditioning; clip longer and it is truncated."
  },
  "languages": ["ar","da","de","el","en","es","fi","fr","he","hi","it","ja","ko",
                "ms","nl","no","pl","pt","ru","sv","sw","tr","zh"],
  "parameters": [
    { "name": "text",           "type": "string",  "default": null,
      "min_length": 1, "group": "common",
      "description": "Text to synthesize." },
    { "name": "audio_base64",   "type": "string",  "default": null,
      "max_length": 10000000, "group": "common",
      "description": "Reference voice sample (base64). ~10 s recommended." },
    { "name": "language",       "type": "string",  "default": null,
      "enum": ["ar","da","de","el","en","es","fi","fr","he","hi","it","ja","ko",
               "ms","nl","no","pl","pt","ru","sv","sw","tr","zh"],
      "group": "common", "description": "Language code. Omit to skip the language tag." },
    { "name": "seed",           "type": "integer", "default": null,
      "min": 1, "max": 1000, "group": "common",
      "description": "Random seed for reproducibility. Random if omitted." },
    { "name": "exaggeration",   "type": "number",  "default": 0.5,
      "min": 0.0, "max": 2.0, "step": 0.05, "group": "engine",
      "description": "Expression/energy boost. ~0.5 general use, ~0.7+ dramatic." },
    { "name": "cfg_weight",     "type": "number",  "default": 0.5,
      "min": 0.0, "max": 1.0, "step": 0.05, "group": "engine",
      "description": "CFG weight. ~0.3 for fast-talking refs or to reduce accent bleed." },
    { "name": "temperature",    "type": "number",  "default": 0.8,
      "min": 0.0, "max": 2.0, "step": 0.05, "group": "engine", "advanced": false },
    { "name": "repetition_penalty", "type": "number", "default": 1.2,
      "min": 1.0, "max": 2.0, "step": 0.05, "group": "engine", "advanced": true },
    { "name": "min_p",          "type": "number",  "default": 0.05,
      "min": 0.0, "max": 1.0, "step": 0.01, "group": "engine", "advanced": true },
    { "name": "top_p",          "type": "number",  "default": 1.0,
      "min": 0.0, "max": 1.0, "step": 0.01, "group": "engine", "advanced": true }
  ]
}
```

### 4.2 Field reference

Top level:

| Field | Type | Notes |
|---|---|---|
| `schema_version` | int | Bumped only for breaking changes (§3.5). |
| `engine` | string | Stable slug: `chatterbox`, `omnivoice`, `qwen3-tts`, … Used by app for profile caching. |
| `model` | string | Human-readable loaded-model label (e.g. `chatterbox-multilingual-v3`). |
| `device` | string | `cuda` / `cpu` / `mps`. Informational. |
| `sample_rate` | int | **Output** sample rate. App resamples downstream if it needs a uniform rate. |
| `watermarked` | bool | Output carries a neural watermark (Chatterbox: PerTh). App should surface this to end users. |
| `endpoint` | string | Synthesis path (`/synthesize`). |
| `reference_audio` | object | See below. `null` for engines that don't take a reference clip. |
| `languages` | string[] \| null | ISO-ish codes; `null` = engine is language-agnostic or single-language with no tag. |

`reference_audio`: `required` (bool), `formats` (string[]), `min_duration_s` / `max_duration_s` (number \| null), `note` (string).

`parameters[]` (`ParamSpec`):

| Field | Type | Notes |
|---|---|---|
| `name` | string | Exact request-body field name. |
| `type` | `string` \| `integer` \| `number` \| `boolean` | |
| `default` | any \| null | Model default; `null` = no default (required or random). |
| `min` / `max` / `step` | number \| null | `step` from override map (JSON schema has no step). |
| `enum` | string[] \| null | For `string` params (e.g. `language`). |
| `min_length` / `max_length` | int \| null | For `string` params. |
| `description` | string | Rendered as tooltip/help text. Keep user-facing quality. |
| `group` | `common` \| `engine` | `common` = core-vocabulary field, app uses its polished widget; `engine` = rendered generically. |
| `advanced` | bool, default false | App collapses `true` params behind an "Advanced" disclosure. |

### 4.3 UI rendering rules (app side)

| ParamSpec shape | Widget |
|---|---|
| `integer` + `min`/`max`, small range | slider (step 1) |
| `number` + `min`/`max` + `step` | slider |
| `number` without `step` | number input |
| `string` + `enum` | select (default = "not set" option when `default: null`) |
| `boolean` | toggle |
| `string` plain | text input |
| anything unrecognized | **raw-JSON escape hatch** (forward-compat rule, §3.5) |

## 5. Shared Package: `tts-engine-common`

**Dependencies: `fastapi`, `pydantic>=2`. Nothing else — no torch, no engine libs.** (Keeps install light, import side-effect-free, trivially testable on any dev box.)

Suggested layout (separate small repo; each server's venv installs it by path or git ref — *open question Q2*):

```
tts-engine-common/
  pyproject.toml
  src/tts_engine_common/
    __init__.py          # re-exports
    models.py            # ParamSpec, Capabilities, ReferenceAudioSpec
    derive.py            # build_capabilities(), JSON-schema normalizer
    route.py             # capabilities_endpoint(doc) -> FastAPI route factory
    core.py              # CORE_FIELDS = {"text", "audio_base64", "language", "seed"}
  tests/
    test_derive.py       # mapping table tests (incl. anyOf-optional, enum, bounds)
    test_route.py        # endpoint returns 200 + exact doc
    test_contract.py     # doc fields ⊆ model fields; override keys ⊆ model fields
```

API sketch:

```python
# models.py
class ParamSpec(BaseModel):
    name: str
    type: Literal["string", "integer", "number", "boolean"]
    default: Any = None
    description: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    enum: list[str] | None = None
    min_length: int | None = None
    max_length: int | None = None
    group: Literal["common", "engine"] = "engine"
    advanced: bool = False

class Capabilities(BaseModel):
    schema_version: int = 1
    engine: str
    model: str
    device: str
    sample_rate: int
    watermarked: bool
    endpoint: str
    reference_audio: ReferenceAudioSpec | None
    languages: list[str] | None
    parameters: list[ParamSpec]

# derive.py
def build_capabilities(
    request_model: type[BaseModel],
    *,
    engine: str,
    overrides: dict[str, dict] | None = None,
    **meta,                      # model, device, sample_rate, watermarked,
) -> Capabilities:               # endpoint, reference_audio, languages
    """Project the Pydantic request schema into the capabilities doc.
    Raises ValueError on unknown override keys (drift guard)."""

# route.py
def capabilities_endpoint(doc: Capabilities) -> Callable:
    """Returns an async route handler serving the doc with Cache-Control: max-age=300."""
```

## 6. Per-Server Integration

### 6.1 Chatterbox (reference implementation)

Changes to `server_chatterbox.py` (all additive, ~30 lines):

```python
from pydantic import ConfigDict
from tts_engine_common import build_capabilities, capabilities_endpoint

class SynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")   # D5 — the only behavioral change
    # ... fields unchanged ...

CAPABILITIES = build_capabilities(
    SynthesisRequest,
    engine="chatterbox",
    model=MODEL_LABEL,
    device=DEVICE,
    sample_rate=24_000,            # S3GEN_SR; known pre-load, no runtime dependency
    watermarked=True,              # PerTh, applied inside the library
    endpoint="/synthesize",
    reference_audio={"required": True,
                     "formats": ["wav", "mp3", "ogg", "flac"],
                     "min_duration_s": MIN_PROMPT_DURATION_S,
                     "max_duration_s": None,
                     "note": "First 10 s used for speaker conditioning."},
    languages=list(SUPPORTED_LANGUAGES),
    overrides={
        "seed": {"group": "common"},
        "language": {"group": "common"},
        "text": {"group": "common"},
        "audio_base64": {"group": "common"},
        "exaggeration": {"step": 0.05},
        "cfg_weight": {"step": 0.05},
        "temperature": {"step": 0.05},
        "repetition_penalty": {"step": 0.05, "advanced": True},
        "min_p": {"step": 0.01, "advanced": True},
        "top_p": {"step": 0.01, "advanced": True},
    },
)

app.add_api_route("/capabilities", capabilities_endpoint(CAPABILITIES),
                  methods=["GET"], tags=["System"])
```

### 6.2 Port checklist (repeat per engine: OmniVoice, Qwen3-TTS, dots.tts)

1. Add `tts-engine-common` to the server venv.
2. `model_config = ConfigDict(extra="forbid")` on the request model.
3. Build `CAPABILITIES` with that engine's meta:
   - [ ] `engine` slug (decide stable slugs for all four — *open question Q3*)
   - [ ] `model` label, `device`, `sample_rate` (output), `watermarked`
   - [ ] `reference_audio` (required? formats? min/max duration? — OmniVoice: ~10 s recommended; Qwen3-TTS: fill in)
   - [ ] `languages` (null for English-only engines)
   - [ ] `overrides` (mark core-vocab fields `common`; set `step`/`advanced` for the rest)
4. Register the `/capabilities` route.
5. Snapshot test of the emitted doc (§8).
6. Deploy + run the §11 verification checklist.

## 7. Client App Changes

1. **On connect**: `GET /capabilities` → store as an *engine profile* keyed by `engine` slug. Refetch on reconnect and on any synthesis 422 mentioning an unknown field (cheap self-healing).
2. **`schema_version` gate**: version > supported → minimal mode (§3.5). Version ≤ supported → full render.
3. **Form rendering**: §4.3 widget rules; `group: common` fields get the polished shared widgets; `group: engine` fields render generically; `advanced: true` collapsed by default.
4. **Request path unchanged**: POST the rendered fields top-level to `profile.endpoint`.
5. **Watermark disclosure**: show a notice when `watermarked: true` (responsible-AI surface — the end user should know the audio is watermarked).
6. **Sample-rate handling**: if the app pipelines audio at a fixed rate, resample from `profile.sample_rate` (or request the server to — out of scope here).

## 8. Testing Strategy

| Layer | What | Where |
|---|---|---|
| Unit | JSON-schema → `ParamSpec` mapping table: plain int/float/str/bool, `int \| None` (`anyOf` + null), `Literal` enum, `str \| Literal \| None`, bounds, min/max length, description passthrough, default extraction | `tts-engine-common/tests/test_derive.py` |
| Unit | Unknown override key → `ValueError` at build time | same |
| Contract | Every `doc.parameters[].name` ∈ `request_model.model_fields`; overrides keys ⊆ model fields (asserted at import, re-checked in test) | `tts-engine-common/tests/test_contract.py` |
| Route | `/capabilities` → 200, `application/json`, body equals built doc, cache header present | `tts-engine-common/tests/test_route.py` |
| Server (per engine) | **Snapshot test** of the full capabilities JSON; PRs that change the doc must show the diff | each server repo |
| Server (per engine) | POST unknown field → 422 naming it (regression test for D5) | each server repo |
| E2E (GPU box) | `/health`, `/capabilities`, `/synthesize` round-trip per engine; app matrix: every engine × every advertised param at default + one non-default value | manual, §11 |

Note: `tts-engine-common` and the contract/derive tests run on any dev box (no GPU, no engine deps) — the only GPU-box-dependent step is the E2E smoke.

## 9. Milestones & Acceptance Criteria

| M | Work | Est. | Acceptance criteria |
|---|---|---|---|
| M0 | Finalize this plan; answer open questions (§10); fix engine slugs | 0.5 d | Plan approved; Q1–Q5 answered |
| M1 | `tts-engine-common` package: models, derive, route, core; full test suite green | 1 d | Unit + contract + route tests pass on a dev box without GPU |
| M2 | Chatterbox integration (reference impl, §6.1) + snapshot + 422 tests | 0.5 d | Tests green locally; on GPU box: `/capabilities` matches snapshot, `extra` field → 422, `/synthesize` unchanged behavior |
| M3 | Port OmniVoice, Qwen3-TTS, 4th engine (§6.2) | 1 d | Each: snapshot + 422 tests green; §11 checklist passed on the box |
| M4 | App: profile fetch/cache, generic renderer, version gate, watermark notice, raw-JSON escape hatch | 1–2 d | Manual matrix passes for all four engines; adding a *new* param to a server requires **zero** app code change (prove with one throwaway param) |
| M5 *(optional)* | OpenAI-compat `POST /v1/audio/speech` per engine (plain text→wav, no reference audio) | 0.5 d each | Any OpenAI-SDK client can synthesize against the fleet |

M1–M3 can proceed in parallel per engine once M1 lands; M4 only needs M2 to start (develop against Chatterbox, verify against the rest in M3).

## 10. Risks, Edge Cases & Open Questions

**Risks / edge cases**
- *R1 — `anyOf` normalizer is the only fiddly code in the shared package.* Mitigated by the exhaustive mapping-table unit tests (M1). Fallback if it ever bites: allow an explicit per-field `ParamSpec` override in the override map, bypassing derivation for that field.
- *R2 — `extra="forbid"` can break a client that already sends fields a server silently drops today.* Mitigation: before flipping it (M2), grep the app's request builders for fields not in each engine's schema. Expected clean; verify anyway.
- *R3 — Same field name, different semantics across engines* (e.g. a future `temperature` with different ranges). Per-engine metadata makes this *visible* (the app renders per-engine bounds), which is strictly better than today — but core-vocabulary names must keep stable semantics; any semantic change is a `schema_version` bump.
- *R4 — Doc grows stale relative to the model.* Structurally prevented (D4); the snapshot test is the second wall.

**Open questions (need owner decision at M0)**
- **Q1.** Core vocabulary: confirm `text` / `audio_base64` / `language` / `seed` as the agreed `group: common` set (does any engine use a different *name* for its reference-audio field today? If so, we standardize names at M2–M3 — a one-time breaking change to those servers, acceptable since the app is the only client).
  - **Answer**: the proposed set of common fields looks insufficient. Most voice cloners (not all) require a `reference_text`. I think that should be included and marked as optional
- **Q2.** Where does `tts-engine-common` live? Private git repo (recommended — installable by path/git ref from each server venv) vs. a folder inside one repo others depend on.
  - **Answer**: this repo (tts-serve) has been created to house both the tts-engine-common and all the implementations of it.
- **Q3.** Stable engine slugs: `dots.ttx`, `omnivoice`, and `qwen3-tts` implementations have all been tested and work with the app. `chatterbox` was a very recent addition and is not yet tested. (Chatterbox was in fact the motivation to finally standardize this). Is the `chatterbox` implementation salvageable?
  - **Answer**: Likely, none of the current implementations are salvageable - this work might involve a ground-up rewrite of all of them.
- **Q4.** chatterbox's actual parameter surface — need to sanity-check that every shape in §4.2 is expressible (e.g. if it has a `string`-without-enum param or a `boolean`, we're covered; if it has something weirder, the escape hatch absorbs it).
  - **Answer**: we can learn this during implementation, and adjust the current chatterbox script (or rewrite it entirely) as needed.
- **Q5.** Do we want `Cache-Control` on `/capabilities` (proposed: `max-age=300`), and does the app ever run against multiple *versions* of the same engine slug simultaneously (affects profile-cache keying — proposed key: `engine + schema_version`)?
  - **Answer**: no cache-control for now. App will interrogate `/capabilities` on each initial connection. Also: our assumption is that the application will never run against multiple versions simultaneously.

## 11. Verification Checklist (GPU server, per engine)

```bash
# 1. Discovery
curl -s localhost:8000/capabilities | jq .
#    - schema_version == 1
#    - parameters[] names match what the UI renders
#    - reference_audio.languages/etc. correct for this engine

# 2. Strictness
curl -s -X POST localhost:8000/synthesize -H 'Content-Type: application/json' \
  -d '{"text":"hi","audio_base64":"...","exaggeration":0.9}'   # on a non-chatterbox engine
#    - expect 422 naming "exaggeration"

# 3. Regression: existing synthesis path unchanged
#    - one /synthesize call per engine at all-default params, audio sanity-checked

# 4. App matrix
#    - connect to each engine → options panel shows exactly the advertised params
#    - move each slider/selector to a non-default value → request contains it → no 422
```

---

## Appendix A: What we explicitly rejected

| Option | Why not |
|---|---|
| Adopt OpenAI `/v1/audio/speech` as *the* API | 4-parameter surface; no discoverability; reference-audio/cloning doesn't map to `voice`. (Kept as optional M5 side-car.) |
| Universal superset request schema (all params, most N/A) | The LCM's failure mode inverted; N/A fields, conflicting semantics, and every engine breaking on every other's validation. |
| `extra: dict` passthrough bag in the request | Right tool for *uncontrolled third-party* clients; we own both ends, so typed fields + `extra="forbid"` + discovery doc is simpler and stricter. Revisit if that assumption changes. |
| Hand-maintained capabilities JSON per server | Drifts from the Pydantic model the moment someone edits a `Field(...)`. Derivation (D4) removes the failure class entirely. |
| App scrapes `/openapi.json` | Workable in a pinch, but OpenAPI carries FastAPI/pydantic noise and no UI-hint slots (`group`, `advanced`, `step`). The capabilities doc is a curated projection; ~40 lines of shared code earn its keep. |
