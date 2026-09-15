# Language handling

The client application prefers to treat `language` as a two-letter code. For example:

- `en`: English
- `fr`: French
- `de`: German
- etc.

By convention, null or empty `language` values should default to `en` implicitly.

## The problem

Not all TTS engines accept language in this way. Some prefer a full language name:

- `English`
- `French`
- `German`
- etc.

## Proposal

The API provided by `tts-serve` should abstract this, so that the client application
can always deal with `language` as a two letter code. Any TTS engine that requires
something else can provide a mapping table internally.

TTS engines that don't require a language (for those engines that are locked
to English-only, for example), can simply ignore the client-provided `language`
value *without throwing an error*.

TTS engines that expect a language may throw an error if the client-provided
`language` code is unknown. For example: `x?` is not a valid language code.

## Implementation (2026-09)

All servers in `impl/` conform to this contract, via shared helpers in
`tts_engine_common.language` (`DEFAULT_LANGUAGE`, `is_language_code`,
`normalize_language`, `validate_language_code`).  The four servers that
pre-dated the contract were migrated in September 2026:

| Server | Before | After |
|---|---|---|
| Chatterbox | code enum, omit = no tag | code enum; null/empty → `en` |
| OmniVoice | free-form code **or** name | two-letter code only (validator); null/empty → `en` |
| Qwen3-TTS | lowercase *names* + `auto` | two-letter codes + `auto`; server maps codes → names via `LANGUAGE_CODE_TO_NAME` |
| dots.tts | free-form code/name/`none`/`auto_detect` | two-letter codes + `auto`; server maps to uppercase codes / `auto_detect` via `_to_engine_language` |

Servers added to `impl/` after the contract was written were built to it
from the start (no migration):

- **faster-qwen3-tts** — two-letter codes + `auto`; server maps codes →
  lowercase *names* via `LANGUAGE_CODE_TO_NAME` (same mapping as Qwen3-TTS).
- **IndexTTS** — restricted five-code enum (`ar`, `en`, `es`, `ja`, `zh`),
  no `auto`; any other value 422s at the boundary.
- **LuxTTS** — the engine has no language parameter at all: the server
  accepts any two-letter code but does not forward it, and capabilities
  advertise `languages: null` (the no-support case of this contract).

Decisions taken while implementing:

- **Nullability is kept, not removed.** `language` remains an optional field
  (`... | None`), but its default is now `"en"` and a `mode="before"`
  validator normalizes `None`/empty/whitespace to `en`. Explicit `null` still
  parses (backward compatible) and ends up as `en` — the "null or empty
  defaults to `en`" convention is enforced at the request boundary, and
  clients may stop sending the field entirely.
- **`auto` stays as a documented special value** where the engine offers
  auto-detection (Qwen3-TTS, dots.tts). It is not a language code, and it
  appears in `/capabilities` enum lists as such; the two-letter-code rule
  applies to *language values*, not to engine sentinels.
- **Engine-specific quirks move inside the server.** Qwen3-TTS's name
  mapping and dots.tts's uppercase/`auto_detect` mapping are private to
  `impl/server_*.py`; the capabilities document and the API surface only ever
  show codes (and `auto` where supported).
- **`schema_version` bumped 1 → 2** (see `tts_engine_common.core`): clients
  must re-check `/capabilities` because default values and the Qwen3-TTS
  enum changed.
- **Non-string values are rejected outright.** `language: 7` or `language:
  ["en"]` is not a code: the shared `normalize_language` (a `mode="before"`
  validator, i.e. raw pre-coercion JSON) raises `ValueError` for anything
  that is neither null nor a string, so such requests 422 at the boundary
  instead of 500ing deeper down.

