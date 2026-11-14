# LuxTTS

Quick stats:

- **Server script**: [server_luxTTS.py](server_luxTTS.py)
- **Sample rate**: 48 kHz
- **Notes**:
  - reference audio transcript **not needed** — the engine always transcribes
    the reference clip itself with Whisper (openai/whisper-base on GPU,
    whisper-tiny on CPU), so there is no `reference_text` field and every
    request pays the ASR cost.
  - the engine has **no language parameter**: its tokenizer auto-detects
    English and Chinese per text segment (other scripts are dropped). The
    API accepts any two-letter `language` code for consistency but does not
    forward it.
  - `seed` is meaningful — the flow-matching solver's initial noise comes
    from the PyTorch RNG. The same seed and inputs give bit-identical audio
    on CPU; on CUDA the audio is near-identical but not bit-identical
    (residual ~1e-7 sample differences from non-deterministic GPU kernels).
  - the reference clip must yield an English or Chinese transcription: the
    engine conditions on Whisper's output, and a clip that produces none
    (silence, a clip in another language) is rejected with **400**
    `reference audio contains no recognizable speech in its first 5 s: ...`.
    Without that check the engine divides the predicted output length by the
    (zero) prompt-token count and dies with `upper bound and lower bound
    inconsistent with step sign` — a 500 no client can act on.  Two gotchas
    about that window (and about what the check does *not* catch) live under
    ["The reference clip"](#the-reference-clip).
  - the text must contain something the engine can actually say — see
    ["Nothing to speak"](#nothing-to-speak).
  - the first request after startup also pays a one-time librosa
    initialisation (~10 s).

## Nothing to speak

LuxTTS' tokenizer (`EmiliaTokenizer` in `zipvoice/tokenizer/tokenizer.py`)
splits text character by character into *Chinese* (`U+4E00`–`U+9FA5`), *en*
(ASCII `A–Z`/`a–z`) and *other*, and **silently drops whole "other"
segments** with a `No English or Chinese characters found` warning.  A text
that is nothing but "other" characters therefore becomes an **empty token
list**, and the engine has no path for that: the predicted output length
collapses to (or below) the length of the reference clip, so zero frames of
new audio reach the vocoder, which aborts in its first `Conv1d`:

```
RuntimeError: Calculated padded input size per channel: (6). Kernel size: (7)
```

That reaches the client as a bare **HTTP 500**.  A chat application that
speaks one sentence per TTS request hits it routinely: streamed replies break
on `.`/`!`/`?`/newline, so a bare `...`, a Markdown fence, or a table rule
can arrive as a whole request.  The server therefore runs the engine's own
tokenizer as a pre-flight (sub-millisecond, and the only rule that cannot
drift) and answers **422** with an actionable message:

```json
{
  "detail": "text contains nothing this engine can speak: LuxTTS keeps only ASCII letters, Chinese characters and <pinyin>/[tag] parts, and drops punctuation, digits, emoji and every other script (Cyrillic, Greek, Japanese, Korean, Arabic, ...).  Send at least one English or Chinese word."
}
```

The pre-flight runs **under the synthesis lock**: the tokenizer normalizes text
through a shared, stateful `inflect` engine (`EnglishTextNormalizer` keeps one
`inflect.engine()` on the tokenizer object), so it cannot run alongside a
generation.  A client that fires a bare `...` while a 30-second request is in
flight sees its 422 *after* that request finishes, in queue order — the check
is cheap, not out-of-band.

Measured verdicts (YatharthS/LuxTTS, `texts_to_token_ids`, CUDA box, engine
version pinned by this repo's install notes — re-run them with the snippet
below if you doubt the table).  The counts are indicative: they move with the
normalizer dependencies the tokenizer pulls in (`jieba`, `pypinyin`,
`piper_phonemize`, `inflect`).  The **0 vs non-zero** split is the contract:

| text | tokens | result |
|------|-------:|--------|
| `Hello there` | 11 | 200 |
| `café` / `über groß` | 6 / 12 | 200 (Latin letters survive) |
| `你好，世界` | 9 | 200 |
| `test` | 5 | 200 |
| `1234` | 0 | 422 |
| `....` / `!?!` / `…?` | 0 | 422 |
| ` ``` ` (Markdown fence) | 0 | 422 |
| `[laughs]` / `[music]` | 0 | 422 (a whole `[tag]` is looked up as one token) |
| `→ ❤ 😀` | 0 | 422 |
| `2026-11-10 08:30` | 0 | 422 |
| `こんにちは` | 0 | 422 (kana is "other": no Japanese) |
| `Привет мир` / `한국어` / `עברית` / `ไทย` | 0 | 422 (non-Latin/non-Chinese scripts are dropped) |

Note the last two groups: this engine speaks **English and Chinese only** (the
`language` field is accepted but never forwarded — see the Notes above), so a
Russian, Japanese or Korean conversation is unvoiceable whatever the client
does.  Use a different engine for those languages; the 422 at least says so.

One caveat on the message: `tokenize_EN`/`tokenize_ZH`/`tokenize_pinyin` each
swallow their own exceptions and return `[]`, so an internal normalizer
failure on perfectly speakable text also reaches the guard as zero tokens and
is reported as unspeakable.  The engine sees the same zero tokens and 500s on
the same input, so the 422 is still the better answer — and the server's
`Text tokenizes to nothing (text_len=…, tokens=0)` log line, next to the
engine's own `Tokenization of English texts failed` warning, is how you tell
the two cases apart.

<details>
<summary>Snippet used to produce the table</summary>

```python
from huggingface_hub import snapshot_download
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer

tok = EmiliaTokenizer(
    token_file=f"{snapshot_download('YatharthS/LuxTTS')}/tokens.txt"
)
for text in ["Hello there", "1234", "....", "```", "[laughs]", "Привет мир"]:
    print(repr(text), len(tok.texts_to_token_ids([text])[0]))
```
</details>

## The reference clip

There is no `reference_text` field: the engine transcribes the clip itself
(`process_audio` in `zipvoice/modeling_utils.py`) and conditions on that
transcript.  Two consequences are worth knowing before you pick a clip.

**The transcribable window is `prompt_duration`, not the clip.** The value
goes straight into `librosa.load(..., duration=...)`, so with the default
`prompt_duration=5.0` Whisper only ever hears the **first five seconds**.
A 10 s clip that opens with 6 s of silence is a 400 — the pristine middle of
the clip buys it nothing.  Lead with speech, trim the leading silence, or
raise `prompt_duration`.  The 400 names the window it actually looked at, so
the client can render advice about *that* request:

```json
{
  "detail": "reference audio contains no recognizable speech in its first 5 s: the engine transcribes only that window with Whisper and conditions on the transcript, so silence, music, or speech other than English / Chinese cannot be cloned.  Start the clip with speech, or raise prompt_duration."
}
```

**The check is necessary, not sufficient.** Whisper hallucinates text on
silence and music (`"Thank you."` is the canonical output), so a clip with no
speech in it can clear the check with a plausible transcript.  The request then
returns 200 with audio that is not the speaker: the model speaks Whisper's
invented words over whatever timbre the audio features carry.  Nothing here
catches that — a clean, spoken reference is the only reliable input.  The 400
covers the *empty* transcript, which is the case that crashes the solver.

## Installation

Start by setting up a venv (or use your conda setup):

```
mkdir LuxTTS
cd LuxTTS
python3 -m venv .venv
source .venv/bin/activate
```

Now install LuxTTS in this environment. It is a git repo, not a PyPI
package, and one of its dependencies (`linacodec`) is **git-only** (not on
PyPI), so the install takes two steps — and the second one is easy to miss:

```
git clone https://github.com/ysharma3501/LuxTTS.git
cd LuxTTS

# Install dependencies:
pip install -r requirements.txt

# We also need zipvoice, which isn't installed above:
pip install . --no-deps
```

If you use `uv`, a single `uv pip install .` works instead of the above,
and installs everything in one step.

Now clone `tts-serve` and install its dependencies:

```
git clone https://github.com/scorbo2/tts-serve
cd tts-serve
pip install ./tts-engine-common fastapi uvicorn loguru soundfile
```

Start it up!

```
python impl/server_luxTTS.py
```

### Changing host

By default, `0.0.0.0` is used. To force a local-only server:

```
export LUX_TTS_HOST=127.0.0.1
python impl/server_luxTTS.py
```

### Changing port

By default, `7500` is used. To choose a different port:

```
export LUX_TTS_PORT=8500
python impl/server_luxTTS.py
```

### To use a local model path instead of huggingface

By default, the server script will download the needed model from huggingface
(YatharthS/LuxTTS) on first run. After that, the model should exist in your
local huggingface cache dir (`~/.cache/huggingface/hub/`).
If you have downloaded the model yourself and want to force a local path:

```
export LUX_TTS_MODEL=/path/to/LuxTTS/
python impl/server_luxTTS.py
```

Two upstream engine quirks to be aware of (neither is a tts-serve bug):

- On the **GPU** path, the engine treats *any* non-default `LUX_TTS_MODEL`
  value as a local directory — it only ever downloads the default id. So a
  non-default HuggingFace id will not be fetched; you must pass a local path
  to already-extracted model files.
- On the **CPU** path, the engine ignores `LUX_TTS_MODEL` entirely and always
  downloads the default model (an upstream bug in `zipvoice`'s CPU loader).

### Run on a different device

By default, LuxTTS will run on `cuda`. To force a different device:

```
export LUX_TTS_DEVICE=cpu
python impl/server_luxTTS.py
```

Note: if `cuda` is requested but unavailable, the engine silently falls back
to MPS, then CPU. `GET /health` (and `GET /capabilities`) report the
*configured* device (`LUX_TTS_DEVICE`), not the one actually resolved —
check the startup log ("CUDA not available, switching to MPS/CPU") to see
what the server is really running on.

Accepted values are `cuda`, `mps`, and `cpu` — **not** `cuda:1`. The engine's
loader rebuilds the device as `torch.device(device, 0)`
(`zipvoice/modeling_utils.py`), which hard-codes the ordinal, while its Whisper
transcriber is handed the raw string — so an indexed value is either rejected by
torch or ends up splitting the model (GPU 0) and the transcriber (GPU 1) across
cards. To choose the GPU, pin it upstream instead:

```
export CUDA_VISIBLE_DEVICES=1
export LUX_TTS_DEVICE=cuda
python impl/server_luxTTS.py
```

A value that names no device (`gpu`, `cuda:1`, `tpu`) stops the server at start
with a message naming `LUX_TTS_DEVICE` — before the checkpoint is read.

### CPU thread count

The CPU path is ONNX-based and takes a thread count:

```
export LUX_TTS_THREADS=8
python impl/server_luxTTS.py
```
