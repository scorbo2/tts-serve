# tts-engine-common

Shared building blocks for [tts-serve](../README.md) TTS engine servers:
capabilities derivation, the core request/response vocabulary, the
`GET /capabilities` route, and small helpers.

Deliberately pure **FastAPI + Pydantic** — no torch, no engine libraries — so
it installs light, imports with no side effects, and tests on any dev box.

## Why capabilities are *derived*, not written

The classic failure mode of a discovery endpoint is drift: the documentation
says one thing, the validator does another. Here the `/capabilities` document
is **projected from the exact Pydantic model that validates
`POST /synthesize`** (design decision D4 in
[`docs/01-server-generification.md`](../docs/01-server-generification.md)).
There is one source of truth, so the two cannot drift.

What the JSON schema cannot express (UI sugar like `step` and `advanced`, and
static per-server facts like `watermarked`) is layered on via a small,
validated override map. Overrides that would *lie* about validation (changing
a field's `name` or `type`) are rejected at import time.

## API

| Symbol | Purpose |
|---|---|
| `build_capabilities(request_model, *, engine, model, device, sample_rate, watermarked, endpoint=..., reference_audio=..., languages=..., overrides=...) -> Capabilities` | Project a Pydantic request model into the capabilities document. Raises `ValueError` for bad overrides, `DerivationError` for unsupported schema shapes. |
| `capabilities_endpoint(doc) -> handler` | Async FastAPI handler serving the document as JSON (serialized once at build time). |
| `add_capabilities_route(app, doc, path="/capabilities")` | Convenience: mount the endpoint on an existing app. |
| `CoreSynthesisResponse` | Base response model: `audio_base64`, `sample_rate`, `seed`, `time_used`, `rtf`. Subclass it to add engine extras. Sanitizes engine `inf`/`nan` (`rtf` → `null`, `time_used` → `0.0`) because JSON cannot represent them. |
| `Capabilities` / `ParamSpec` / `ReferenceAudioSpec` | The document models (all `extra="forbid"`). |
| `decode_base64(data) -> bytes` | Strict base64 decode (`validate=True`); `ValueError` on malformed input. |
| `compute_rtf(time_used_s, num_samples, sample_rate) -> float \| None` | Real-time factor; `None` when the duration is zero or the result non-finite. |
| `temp_audio_dir(name) -> Path` | Per-server staging directory under the system temp dir (created). Respects `$TMPDIR`. |
| `stage_audio(raw_bytes, directory) -> str` | Content-addressed (SHA-256) staging, written atomically, file **kept**. For engines that cache conditioning per path (repeat requests with the same clip reuse the file). |
| `write_temp_audio(raw_bytes, directory) -> str` / `cleanup_temp(path)` | One-shot UUID-named staging + best-effort removal. For engines that read the prompt once and do not cache by path. |
| `CORE_FIELDS` | The common vocabulary: `{"text", "audio_base64", "reference_text", "language", "seed"}`. Fields in this set are tagged `group: "common"` in the document. |
| `SCHEMA_VERSION` | Currently `2`; bump only on breaking document changes. |

### Minimal usage

```python
from pydantic import BaseModel, ConfigDict, Field
from tts_engine_common import (
    CoreSynthesisResponse, build_capabilities, capabilities_endpoint,
)

class SynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., min_length=1, description="Text to synthesize.")
    audio_base64: str = Field(..., min_length=1, description="Reference sample, base64.")
    seed: int | None = Field(None, ge=1, le=1000)
    temperature: float = Field(0.8, ge=0.0, le=2.0)

class SynthesisResponse(CoreSynthesisResponse):
    fid: str

doc = build_capabilities(
    SynthesisRequest,
    engine="example",
    model="example/v1",
    device="cuda",
    sample_rate=24000,
    watermarked=False,
    reference_audio={
        "required": True,
        "formats": ["wav", "mp3"],
        "min_duration_s": 2.0,
        "note": "3-10 s recommended.",
    },
    overrides={"temperature": {"step": 0.05, "advanced": True}},
)

app = FastAPI()
app.add_api_route("/capabilities", capabilities_endpoint(doc), methods=["GET"])
```

`/capabilities` now describes `text`/`audio_base64` as required
(`group: "common"`), `seed`/`temperature` as optional with their bounds, and
`temperature` with `step: 0.05, advanced: true`.

## The capabilities document

```jsonc
{
  "schema_version": 2,
  "engine": "chatterbox",                  // stable slug
  "model": "chatterbox-multilingual-v3",   // checkpoint in use
  "device": "cuda",
  "sample_rate": 24000,
  "watermarked": true,
  "endpoint": "/synthesize",
  "reference_audio": {                     // null if the engine does not clone
    "required": true,
    "formats": ["wav", "mp3", "ogg", "flac"],
    "min_duration_s": 2.0,
    "max_duration_s": null,                // omitted when null
    "note": "Only the first 10 s are used for speaker conditioning."
  },
  "languages": ["ar", "da", "..."],        // null when free-form / agnostic
  "parameters": [
    {
      "name": "text",
      "type": "string",                    // string | integer | number | boolean | array
      "required": true,                    // no default in the model => required
      "default": null,
      "description": "Text to synthesize.",
      "min": null, "max": null, "step": null,
      "enum": null,
      "min_length": 1, "max_length": null,
      "item_type": null, "min_items": null, "max_items": null,
      "item_labels": null,
      "group": "common",                   // "common" | "engine"
      "advanced": false
    }
    // For "type": "array", the item fields describe the elements, e.g.:
    // { "name": "emotion_vector", "type": "array", "item_type": "number",
    //   "min_items": 8, "max_items": 8,
    //   "item_labels": ["happy", "angry", "sad", "afraid", "disgusted",
    //                   "melancholic", "surprised", "calm"], ... }
    // ...
  ]
}
```

Notes:

- `type` values mirror the JSON Schema types, so a client can drive input
  rendering (select / spinner / checkbox / text) straight from the document.
- `enum` values are always strings (numeric literals are coerced for
  rendering).
- `min`/`max` are always numeric or null (inclusive bounds; exclusive bounds
  are normalized to the inclusive key by the derivation).
- `languages: null` means "the engine accepts free-form language input" —
  do not treat it as "no languages supported".
- `item_type`/`min_items`/`max_items` are only meaningful for `"array"`
  parameters (null otherwise). Array items are always flat scalars —
  nested arrays and object items are rejected at import time.
- `item_labels` (per-item display labels) is only valid for **fixed-size**
  arrays: `min_items == max_items` with one non-empty label per item, or the
  `ParamSpec` construction fails at import. It is carried via the
  per-server override map (`build_capabilities(..., overrides={...})`),
  since the JSON schema cannot express display labels.

## Known limitations

- Supported field types: `string`, `integer`, `number`, `boolean` and `array`
  (including `X | None` unions and `Literal` enums). Arrays must have flat
  scalar items (`item_type` ∈ string/integer/number/boolean) plus optional
  `min_items`/`max_items` (and optional `item_labels` for fixed-size
  arrays, via overrides). Any other JSON Schema shape (nested arrays,
  objects as items, ambiguous unions) raises `DerivationError` at import
  time rather than emitting a document that misdescribes validation.
- `step` is a UI hint, not validation — the model's own bounds are enforced.
- The document is static for the server's lifetime (it describes the request
  schema, which is fixed at import time). No `Cache-Control` header is sent:
  clients re-interrogate on every connection and must see changes immediately.

## Testing

```bash
python -m pytest tests/
```

The suite covers the schema-derivation mapping table (bounds, enums, unions,
required-ness, overrides, error paths) and the response sanitizers. The
engine servers' own tests (in `impl/tests/`) snapshot the live
`/capabilities` output end-to-end.
