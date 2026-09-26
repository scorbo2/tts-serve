# Voice design

Status: **Proposal** (open questions answered 2026-09-26 after engine research; ready for review)
Date: Written 2026-09; revised 2026-09-26
Scope: all nine servers in `impl/`, the shared package `tts-engine-common`, and the `speak.py` tool; a future Breeze-TTS server
Clients: `speak.py` (bundled), `TalkWithMe` (external application)

---

## 0. TL;DR — Key decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Extend the existing `/synthesize` — no new endpoints.** Mode (clone vs. voice design) is selected by which fields the request body carries. | The engines already work this way (Breeze picks a template from field presence; OmniVoice's `generate()` has three modes off one API). One endpoint + one Pydantic request model keeps the docs/01 single source of truth (D4) intact. |
| D2 | **Normalized request field: `instructions`**, added to the common vocabulary (`CORE_FIELDS`) as an *advertised-when-supported* field, exactly like `reference_text`. | `instruct` (Qwen3-TTS, OmniVoice) and `instruction` (Breeze-TTS) are the same concept. Clients render it generically; per-engine format differences live in the capabilities doc (§5.2). |
| D3 | **`/capabilities` gains one top-level field: `voice_design`** (object, or `null` when the loaded model cannot do voice design). The `instructions` parameter appears in `parameters[]` automatically via derivation. | Additive per docs/01 §3.5 → **no `schema_version` bump** (stays 2). Old clients ignore the new field; new clients discover support before attempting it. |
| D4 | **Unsupported engines return 422, not 501.** | `extra="forbid"` (docs/01 D5) already 422s unknown fields — zero new code, loud, names the field. The machine-readable "not implemented" signal is `voice_design: null` in the capabilities doc, not an HTTP error. (§5.4 explains the deviation from the original proposal.) |
| D5 | **Qwen3-TTS: the loaded checkpoint decides the mode.** The server introspects `tts_model_type` from the checkpoint config at import time and builds its request model + capabilities accordingly (`base` → clone only, `voice_design` → design only). | The engine raises `ValueError` if you call the wrong generate method for the checkpoint; the server contract must match the checkpoint, not a client preference. |
| D6 | **Per-server mode matrix with explicit cross-field validation** (`model_validator`): each server documents which combinations of `instructions` / `audio_base64` are accepted, 422, or 500-ish engine errors. | The one semantic the engines do *not* agree on is what happens when you send both a description and a reference clip. That must be a per-server, documented, validated decision — not an accident. |

**Is generic access possible? Yes — with one important caveat.** Three of the four example engines
support voice design per-request (OmniVoice, Breeze-TTS, dots.tts in a degraded form), and Qwen3-TTS
supports it via a dedicated checkpoint. But the *shape* of the feature differs per engine enough that
"generic" means: one normalized field name + one capabilities section + a per-server mode matrix.
It does **not** mean one shared description format: OmniVoice wants a comma-separated attribute
list from a fixed vocabulary, not a free sentence ("A young woman's gentle voice" is a 400-class
input to OmniVoice). The design below makes that difference discoverable and renders it generically.

---

## 1. Current state

Each time support for a new TTS engine is added to tts-serve, only voice cloning capabilities are
considered:

- reference audio is mandatory (this is the voice to be cloned)
- reference text is often required (sometimes optional, if the engine uses Whisper or similar)
- reference audio language is often required (sometimes optional)

From these parameters, the engine in question can clone a voice and return a voice response to a
given text prompt using that cloned voice. Our current `/capabilities` and `/synthesize` endpoints
only consider voice cloning options.

## 2. Desired state

Some TTS engines (Qwen3-TTS, Breeze-TTS, OmniVoice, and others) also offer "Voice Design" as a
feature. With Voice Design, there is no reference audio or reference audio transcript. Instead, a
text description of the voice to be generated is supplied. Examples of the *free-form* variety:

- "A young woman's gentle voice"
- "A man speaking in a low and menacing voice"
- "A bright and cheerful female voice"

The exact naming of this instruction parameter varies: Qwen3-TTS and OmniVoice call it `instruct`,
Breeze-TTS calls it `instruction`. (OmniVoice's is *not* free-form — see §3.4; it is a structured
attribute list. The concept is the same: "describe the voice in text instead of showing it audio".)

The `tts-serve` library should offer support for Voice Design when using any of the engines that
support it. Engines that do not offer a Voice Design feature make that fact discoverable in
`/capabilities` (`voice_design: null`) and reject attempts to use it with **422** (the original
proposal said 501; the deviation is explained in §5.4).

### Primary consideration

Addition of this feature **must not break backwards compatibility**! Clients of previous versions
of `tts-serve` should be able to connect, query capabilities, and synthesize a cloned voice exactly
as they did before. This new feature is an **addition, not a breaking change**. (Analysis: §5.5.)

---

## 3. Engine research

Findings from reading the actual engine sources (paths relative to the engine repos):

### 3.1 Qwen3-TTS — voice design is a separate *checkpoint*, not a per-request mode

- `qwen_tts/inference/qwen3_tts_model.py`: `Qwen3TTSModel` wraps
  `Qwen3TTSForConditionalGeneration`, which exposes `model.tts_model_type` ∈
  `{"base", "voice_design", "custom_voice"}` (set by the checkpoint config —
  `core/models/configuration_qwen3_tts.py`).
- **`base`** (the checkpoint our `server_qwen3TTS.py` loads by default):
  `generate_voice_clone(text, ref_audio, ref_text, x_vector_only_mode, ...)` — cloning only.
  No `instruct` parameter exists on this path.
- **`voice_design`** (`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`):
  `generate_voice_design(text, instruct, language, ...)`. `instruct` is **free-form**
  natural language; `""`/`None` is allowed and means "no instruction". The method raises
  `ValueError` on any other checkpoint type.
- **`custom_voice`** (`-CustomVoice` checkpoints): `generate_custom_voice(text, speaker, language, instruct)`
  — a different feature: 9 *named* speakers with optional style steering (`instruct` is ignored on
  the 0.6B family). Out of scope for this design (§7).
- Consequence: **one Qwen3-TTS server instance serves exactly one mode.** A Base checkpoint cannot
  voice-design and a VoiceDesign checkpoint cannot clone. `tts_model_type` is readable from the
  checkpoint *config* (`AutoConfig` — a few KB of metadata, no weights), so the server can decide its
  contract at import time without loading the model.

### 3.2 Breeze-TTS — per-request, free-form, and it also supports "voice direction"

- `breeze_infer/api.py` + `breeze_infer/templates.py`: one model, mode selected per request by
  `select_template_name(request)` from field presence:

  | request carries | template | mode |
  |---|---|---|
  | text only | `tts_plain` | plain TTS (engine picks a voice) |
  | text + `instruction` | `tts_instruction` | **voice design** |
  | text + `ref_audio` + `ref_text` | `ref_clone_tata` | voice cloning (ref pair required together) |
  | text + `instruction` + `ref_audio` + `ref_text` | `ref_edit_tata` | **voice direction** (clone the voice, steer its style) |

- `instruction` is **free-form** natural language, wrapped in `<ins_bos>/<ins_eos>` tokens.
- Breeze-TTS is CUDA-only and ships its own streaming API (raw PCM); it is **not yet a tts-serve
  server**. It is the one engine where `instructions` + reference audio is a *first-class combined*
  mode — the only server whose mode matrix allows both.

### 3.3 dots.tts — "instruction" is embedded in the text; there is no separate parameter

- `src/dots_tts/runtime.py`: `generate(text, prompt_audio_path, prompt_text, template_name, ...)`
  with `RUNTIME_TEMPLATE_BY_NAME = {"tts", "instruction_tts", "text_to_audio", "tts_interleave"}`.
- `src/dots_tts/data/pipelines/tts_pipeline.py`: the `instruction_tts` template is
  `[带指令文本]{text}[文本对应语音]{audio}` — the *text slot itself* is "text-with-instruction".
  The user writes the description inside the text (the Gradio app exposes "instruct_tts" as a mode
  dropdown over a single text box). There is no `instruct`/`instruction` parameter anywhere in the
  runtime API.
- Consequence: generic access requires a **server-side transformation** — a normalized
  `instructions` request field that the server composes into the text and pairs with
  `template_name="instruction_tts"`. The exact composition format (separator, ordering) is not
  documented in the repo and **must be validated on a GPU box** (§8).

### 3.4 OmniVoice — per-request, but a *structured* attribute list, not free text

- `omnivoice/models/omnivoice.py`: `generate(text, language, ref_text, ref_audio, instruct, ...)` —
  one model handles all three modes: **voice clone** (ref audio/text), **voice design**
  (`instruct`, no reference), **auto** (neither; the model picks a random voice).
- `instruct` is **not free-form**. `omnivoice/utils/voice_design.py` defines a fixed vocabulary:
  gender (male/female), age (child → elderly), pitch (very low → very high), style (whisper),
  10 English accents, and 12 Chinese dialects — **23 English items + 12 Chinese items**,
  comma-separated and freely combinable across categories. `_resolve_instruct()` validates every
  item server-side and raises `ValueError` (with close-match suggestions) on anything unknown, so
  "A young woman's gentle voice" is a *client error*, not a design description.
- The engine docs define exactly the three modes above; sending `instruct` *and* reference audio
  together is not a documented mode (both are physically embedded in the prompt, but the behavior is
  undefined). Our server treats them as mutually exclusive (§5.3).
- Note for the OmniVoice rollout: our `server_omnivoice.py` currently *requires* `audio_base64`.
  Relaxing it to optional (with `instructions` as the alternative) is purely additive — see §5.5.

### 3.5 Impact on the current fleet

| Server | Engine voice design? | Today | Plan |
|---|---|---|---|
| `server_chatterbox.py` | No (cloning only) | — | `voice_design: null` (no change) |
| `server_omnivoice.py` | **Yes** — structured attributes (§3.4) | not exposed | adopt: `instructions`, audio optional |
| `server_qwen3TTS.py` | **Checkpoint-dependent** (§3.1) | not exposed | adopt via checkpoint introspection (D5) |
| `server_fasterQwen3TTS.py` | Same checkpoint family (CUDA fork) | not exposed | adopt *if* the fork loads VoiceDesign checkpoints — verify (§8) |
| `server_qwen3TTS_mlx.py` | No (mlx-audio serves the Base family only) | — | `voice_design: null` (no change) |
| `server_dotsTTS.py` | **Yes** — embedded-in-text (§3.3) | not exposed | adopt via server-side transformation |
| `server_indexTTS.py` | No (cloning + emotion controls) | — | `voice_design: null` (no change) |
| `server_luxTTS.py` | No (cloning only) | — | `voice_design: null` (no change) |
| `server_voxcpm.py` | **Yes** — description in parentheses at start of `text`, no reference (see AGENTS.md) | *implicitly* supported, undocumented in capabilities | formalize: `instructions` field the server embeds |
| *(future)* Breeze-TTS | **Yes** — free-form + voice direction (§3.2) | no server yet | new server; the reference implementation for the "combined" case |

So: **four servers adopt in this workstream** (OmniVoice, Qwen3-TTS, dots.tts, VoxCPM),
faster-qwen3-tts tentatively, five unchanged, and Breeze-TTS joins as a new server under the same
contract.

---

## 4. Answers to the open questions

### Q1. Can the existing `/capabilities` endpoint be extended without breaking or confusing existing clients?

**Yes.** Per the forward-compatibility rules already in force (docs/01 §3.5):

- Additive changes — a new optional doc field, a new optional request parameter — require
  **no `schema_version` bump**. It stays **2**.
- Existing clients (TalkWithMe, `speak.py`) already have the contract "ignore unknown doc fields;
  render only what the doc advertises". The new top-level `voice_design` field and the new
  `instructions` parameter entry are invisible to old clients. `speak.py`'s
  `check_schema_version()` passes (no bump), and its parameter discovery picks up `instructions`
  automatically because it renders whatever `parameters[]` advertises.
- The one *observable* behavior change for old clients is that two servers (OmniVoice, dots.tts)
  stop rejecting audio-less requests — i.e. requests that were **422 before may now succeed**.
  That is a strict widening of the accepted surface, which is exactly what "additive" means;
  no previously-valid request becomes invalid, and no client logic that sends reference audio is
  affected.
- "Confusing" is managed by making the new field self-describing: `voice_design: null` on
  unsupported engines is unambiguous, and on supported engines the spec object states the format
  and the reference-audio interaction (§5.2), so a client never has to guess.

### Q2. Can the existing `/synthesize` endpoint be modified to allow either cloning OR voice design? Or do we need a new `/voicedesign` endpoint?

**Modify the existing endpoint; do not add `/voicedesigncapabilities` / `/voicedesignsynthesize`.**

- The original fallback (new endpoints) would duplicate the whole machinery twice: two request
  models, two capabilities docs, two validation surfaces, two client code paths — while the engines
  themselves implement mode selection inside one call (Breeze's `select_template_name`, OmniVoice's
  three-mode `generate`). Mode-by-field-presence mirrors the engines' own design, so the server
  layer stays thin.
- One request model per server preserves docs/01 D4 (capabilities derived from the exact model that
  validates `/synthesize`) and D5 (unknown field → loud 422).
- Voice-design-only deployments (Qwen3-TTS VoiceDesign checkpoint) are just servers whose request
  model happens to have no `audio_base64` — the same pattern Chatterbox already uses for having no
  `reference_text`.
- The endpoint path stays `/synthesize` for everyone; `capabilities.endpoint` remains
  `"/synthesize"`.

---

## 5. Design

### 5.1 Request contract (generic)

The normalized field is:

```
instructions: str | None   # voice description; optional everywhere, required-by-mode where noted
```

- **Always optional at the schema level** (never `...`), so it belongs in the common vocabulary the
  same way `language` does: declared by the engine, rendered by the client, semantics per doc.
- Mode is selected by field presence. The generic matrix a server may implement:

| request carries | mode |
|---|---|
| `instructions`, no reference audio | **voice design** |
| reference audio, no `instructions` | **voice cloning** (today's behavior, unchanged) |
| both | per-server (§5.3) |
| neither | per-server (§5.3) |

`instructions` is added to `CORE_FIELDS` in `tts_engine_common/core.py` (common vocabulary,
advertised-when-supported — same status as `reference_text`, which most but not all engines
carry).

### 5.2 Capabilities document extension

One new top-level field, plus `instructions` in `parameters[]` (the latter for free — D4 derivation):

```python
# tts_engine_common/models.py  (additive; existing fields unchanged)

class VoiceDesignSpec(BaseModel):
    """Voice-design capability of the *loaded model*. None = not supported."""
    model_config = ConfigDict(extra="forbid")

    instruction_format: Literal["free_text", "attribute_list"]
    allowed_attributes: list[str] | None = None   # 'attribute_list' only; null for 'free_text'
    reference_audio: Literal["forbidden", "combined"]
    note: str = ""
```

| field | meaning |
|---|---|
| `instruction_format` | `free_text`: natural-language description (Qwen3-TTS VoiceDesign, Breeze, VoxCPM, dots.tts). `attribute_list`: comma-separated items drawn from a fixed vocabulary (OmniVoice). |
| `allowed_attributes` | The vocabulary, for `attribute_list` engines (drives chip-style UI); `null` for `free_text`. |
| `reference_audio` | `forbidden`: a reference clip may not be combined with `instructions` (mutually exclusive, or the field doesn't exist for this checkpoint). `combined`: they may be sent together — the reference voice is cloned and the description steers its style ("voice direction"; Breeze only, today). |
| `note` | Human-readable contract detail (separators, language rules, server-side text composition, etc.). |

`Capabilities` gains:

```python
voice_design: VoiceDesignSpec | None = Field(
    None, description="Voice-design capability of the loaded model; null = unsupported."
)
```

`build_capabilities()` gains a `voice_design: VoiceDesignSpec | dict | None` kwarg (same pattern as
`reference_audio`), with a startup drift guard: if `voice_design` is provided but the request model
has no `instructions` field, construction raises — the doc must never advertise a mode the
validator can't see.

Example (OmniVoice server):

```json
{
  "schema_version": 2,
  "engine": "omnivoice",
  "...": "...",
  "reference_audio": { "required": false, "formats": ["wav", "mp3", "ogg", "flac"],
                       "min_duration_s": 3.0, "max_duration_s": null,
                       "note": "Omit for voice design mode (send `instructions` instead)." },
  "voice_design": {
    "instruction_format": "attribute_list",
    "allowed_attributes": [
      "male", "female",
      "child", "teenager", "young adult", "middle-aged", "elderly",
      "very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch",
      "whisper",
      "american accent", "british accent", "australian accent", "chinese accent",
      "canadian accent", "indian accent", "korean accent", "portuguese accent",
      "russian accent", "japanese accent"
    ],
    "reference_audio": "forbidden",
    "note": "Comma-separated attribute list, e.g. 'female, low pitch, british accent' — not a free sentence. At most one item per category (gender, age, pitch, style, accent, dialect); unknown items and dialect+accent mixes are rejected, and mixed EN/ZH lists are auto-normalised to one language. Chinese items (e.g. 四川话) use full-width commas. Provide this OR reference audio, not both."
  },
  "parameters": [
    { "name": "instructions", "type": "string", "required": false, "default": null,
      "group": "common",
      "description": "Voice description for voice-design mode (attribute list; see voice_design section)." },
    "... existing parameters unchanged ..."
  ]
}
```

`instructions` gets `group: "common"` via the per-server override map (so TalkWithMe can render a
polished shared widget instead of a generic text box).

### 5.3 Mode matrix (per server, validated at the request boundary)

Cross-field validation via Pydantic `model_validator` (422, naming the fields), mirroring the
VoxCPM `reference_text`-without-`audio` precedent:

| Server | design request | clone request | both present | neither present |
|---|---|---|---|---|
| omnivoice | `instructions`, no audio | audio (+ optional `reference_text`), as today | **422** (engine defines only the 3 modes; combining is undefined) | **422** "provide reference audio or instructions" (the engine's "auto/random voice" mode is deliberately not exposed — see §7) |
| qwen3TTS (Base ckpt) | 422 — field unknown | as today | — | — |
| qwen3TTS (VoiceDesign ckpt) | `text` + optional `instructions` (`None` = engine default voice) | 422 — field unknown | — | allowed (`instructions` optional) |
| dotsTTS | `instructions`, no audio → server composes the text + selects `instruction_tts` template | as today | **422** (until GPU-validated, §8) | **422** |
| voxcpm | `instructions`, no audio → server embeds the `(description)` prefix in `text` | as today | **422** | allowed (today's behavior: engine default voice) |
| *(future)* breeze-tts | `instructions`, no audio | as today | **allowed** → voice direction (`ref_edit_tata`) | **allowed** → plain TTS (`tts_plain`) |
| chatterbox, indexTTS, luxTTS, qwen3TTS_mlx, *(fasterQwen3TTS if clone-only ckpt)* | 422 — field unknown | as today | — | — |

Rationale for the conservative defaults: wherever the engine docs don't define "both" or "neither",
the server fails loudly at the boundary (D5 philosophy) instead of guessing; relaxing to the
permissive behavior is a later, evidence-based change after GPU validation.

### 5.4 Why 422 and not 501 (deviation from the original proposal)

The original proposal said unsupported engines should return `501 Not Implemented` when a client
attempts voice design. The tightened design uses **422** instead, for three reasons:

1. **501 is the wrong verb.** 501 means "the server does not implement this *method*".
   `/synthesize` *is* implemented — the request is simply outside this server's contract.
   The honest status for "your request contains a field this server does not accept" is 422.
2. **It comes for free.** `extra="forbid"` (docs/01 D5) already rejects unknown fields with a 422
   that names the offending field. Zero new code, and the behavior is uniform across all
   non-supporting servers by construction.
3. **501 would poison the discovery contract.** To make 501 reachable, every non-supporting server
   would have to declare an `instructions` field that always fails — a field that advertises
   itself in `parameters[]` yet can never succeed. That breaks "capabilities == validation" (D4)
   and would confuse exactly the clients the capabilities doc exists to help.

The machine-readable "not implemented" signal is `voice_design: null` in `/capabilities`. Well-behaved
clients (TalkWithMe, `speak.py`) discover this before attempting anything; a misbehaving client
that tries anyway gets a loud, field-naming 422. Net effect matches the original intent — clients
learn the engine lacks the feature — without inventing an error path the framework already covers.

### 5.5 Backwards-compatibility analysis

**Old client ↔ new server**

- `GET /capabilities`: gains `voice_design` (and, on adopting servers, an `instructions` entry in
  `parameters[]`). Old-client rule — ignore unknown doc fields, render only advertised params
  (docs/01 §3.5) — renders exactly today's form. No `schema_version` bump, so version gates pass.
- `POST /synthesize`: old payloads are byte-identical; validation and engine calls are unchanged on
  every server. The only deltas are strict widenings:
  - OmniVoice/dots.tts: audio-less requests that used to 422 may now succeed (voice design).
    No previously-accepted payload is rejected.
  - VoxCPM: `instructions` is a new optional field; existing text-with-parens voice design
    keeps working exactly as before.
  - Qwen3-TTS: a Base-checkpoint deployment is byte-identical. A *new* deployment pointed at a
    VoiceDesign checkpoint serves the design contract — a new deployment, not an upgrade of an
    existing one (clients see the difference in the `model` field and `voice_design`/
    `reference_audio` doc sections).
- Nothing in the frozen response core changes; voice-design responses use the same
  `audio_base64 / sample_rate / seed / time_used / rtf` (+ per-server extras) shape.

**New client ↔ old server**

- Old capabilities doc has no `voice_design` field → client treats voice design as unsupported and
  never attempts it. This is the original proposal's 501 goal achieved *by discovery instead of by
  error* — strictly better: no failed request, no special-casing.

### 5.6 Rollout plan (per server)

| step | server | work |
|---|---|---|
| 1 | `tts-engine-common` | add `VoiceDesignSpec`, `Capabilities.voice_design`, `build_capabilities(voice_design=...)` + drift guard; add `instructions` to `CORE_FIELDS`; shared-package tests |
| 2 | `server_omnivoice.py` | `instructions` field; `audio_base64` optional; XOR + neither-present validators; forward `instruct` to `generate()`; capabilities meta (attribute list imported from `omnivoice.utils.voice_design`, not re-typed) |
| 3 | `server_qwen3TTS.py` | import-time `AutoConfig` read of `tts_model_type`; two request models (clone / design) + two capabilities docs selected at import; fail-fast on unknown checkpoint type |
| 4 | `server_voxcpm.py` | `instructions` field → embed `(description)` prefix; XOR with `audio_base64`; capabilities meta |
| 5 | `server_dotsTTS.py` | `instructions` field → compose text + `template_name="instruction_tts"`; XOR with audio; capabilities meta (mark format as GPU-pending, §8) |
| 6 | `server_fasterQwen3TTS.py` | same pattern as step 3 *if* the fork loads VoiceDesign checkpoints (verify §8); otherwise `voice_design: null` |
| 7 | each adopting server | snapshot regeneration (`update_snapshots.py`), per-server 422/400 tests (mode matrix), language-contract test list unchanged (`instructions` is not a language field) |
| 8 | *(future)* `server_breezeTTS.py` | new server under the same contract; reference implementation for the "combined" (voice direction) case |

### 5.7 Client impact

- **TalkWithMe**: when `voice_design != null`, offer a Voice-Design mode (tab or toggle) next to
  cloning. `free_text` → textarea; `attribute_list` → chip picker over `allowed_attributes` with
  free-form fallback (the engine validates, and its error messages suggest fixes). When
  `reference_audio: "combined"`, cloning fields and `instructions` render together (direction
  mode); when `"forbidden"`, the modes are exclusive tabs. Old builds: no change (unknown field
  ignored).
- **`speak.py`**: zero changes — it renders `parameters[]` generically, so `instructions` becomes a
  `--instructions` text flag automatically; the `voice_design` section is ignored until a future
  revision special-cases it.

---

## 6. Testing strategy

GPU-free (same harness as today):

| layer | what | where |
|---|---|---|
| shared package | `VoiceDesignSpec` validation (format/attributes consistency), `build_capabilities(voice_design=...)` derivation + drift guard (advertises-but-no-field → error), null/absent cases | `tts-engine-common/tests/` |
| per server | **snapshot** of the full capabilities doc (regenerated via `update_snapshots.py`; the PR diff shows exactly what changed) | `impl/tests/` |
| per server | mode-matrix validation tests: design-only OK, clone-only OK, both-present 422 (where applicable), neither-present 422 (where applicable), unknown-field 422 | `impl/tests/test_server_<name>.py` |
| per server | argument-forwarding pin (fake model): design requests reach the engine's design API with the normalized `instructions` value; clone requests reach the clone API untouched (pattern: `test_server_fasterQwen3TTS.py`) | same |

GPU box (manual, per adopting server):

```
1. capabilities: voice_design section matches the §5.3 matrix; snapshot matches
2. voice design round-trip: instructions-only request -> sane audio
3. clone regression: reference-audio request -> identical behavior to before
4. boundary: both-present / neither-present -> the 422s the matrix promises
5. OmniVoice: attribute list accepted, garbage item -> 4xx with engine suggestion text
6. dots.tts: composed text produces intended styled voice (validates the §8 format question)
```

---

## 7. Non-goals / future work

- **Qwen3-TTS CustomVoice** (9 named speakers + optional style instruct): a different feature
  (speaker *selection*, not description). A future `speaker` parameter under the same
  capabilities mechanism.
- **OmniVoice "auto voice" mode** (neither reference nor instructions → random voice): deliberately
  not exposed; a random voice is a surprising default for an API. Revisit if someone wants a
  "surprise me" endpoint — it would be an additive `auto` flag, not a mode accident.
- **Streaming / OpenAI-compat endpoints**: unaffected by this design; they remain separate work
  (docs/01 M5).
- **Breeze-TTS server**: out of scope for this change set; the contract here is designed so it
  slots in as a normal new-engine implementation (the `new-tts-engine` skill workflow).

## 8. Open questions (need owner decision / GPU validation)

1. **dots.tts composition format.** The repo never documents how "text-with-instruction" pairs are
   written (the Gradio app leaves it to the user). Proposed default: `f"{instructions}: {text}"`
   with `template_name="instruction_tts"` — **validate on a GPU box** (A/B a few separator formats
   and listen) before the server ships.
2. **dots.tts both-present rule.** Is `instruction_tts` + prompt audio even coherent (the runtime
   allows arbitrary template/prompt combinations)? Default: 422 until proven.
3. **faster-qwen3-tts checkpoint support.** Does the fork load `VoiceDesign` checkpoints (its API
   surface shown by the stub is clone-only)? If yes, step 6 proceeds; if no, the server stays
   `voice_design: null`.
4. **Qwen3-TTS config introspection cost.** Reading `AutoConfig` from the HF hub at import time is a
   few-KB fetch (cached by the HF cache dir). Acceptable, or do we prefer an explicit
   `QWEN3TTS_MODEL_TYPE` env override for air-gapped boxes? (Proposal: config-first, env override as
   an escape hatch.)
5. **VoxCPM description prefix.** Confirm the `(description)` prefix format still applies to
   VoxCPM2 checkpoints in the same shape as documented today (the server already relies on it;
   this is a regression check, not new risk).
6. **OmniVoice attribute list drift.** The 23-item English vocabulary lives in
   `omnivoice/utils/voice_design.py` and can change across engine releases. The server should
   *import* it (not re-type it) so capabilities can't drift — confirm the import is importable in
   the server venv, with a hard-coded fallback + test.
