# Qwen3-TTS (MLX)

The Apple-Silicon-native counterpart to [Qwen3-TTS](server_qwen3TTS.md): the
same Qwen3-TTS Base checkpoint family, run through
[mlx-audio](https://github.com/Blaizzy/mlx-audio) instead of
`qwen_tts`/PyTorch. It exists for a controlled A/B comparison between the
stock PyTorch/MPS server and MLX on the same Mac, so it deliberately mirrors
`server_qwen3TTS.py`'s core request/response shape as closely as mlx-audio's
API allows.

Quick stats:

- **Server script**: [server_qwen3TTS_mlx.py](server_qwen3TTS_mlx.py)
- **Sample rate**: 24 kHz
- **Device**: reported as `mlx` (no device-selection env var -- mlx-audio has
  no such knob; MLX itself picks its compute backend, Metal on Apple Silicon)
- **`POST /synthesize`**: complete-file synthesis. ICL (in-context learning)
  only -- `reference_text` is **required**; there is no speaker-embedding-only
  fallback on this endpoint.
- **`POST /stream`**: native MLX model-level streaming -- see
  [Streaming](#streaming-post-stream) below. Supports **both** ICL and
  x-vector (speaker-embedding) voice cloning, selected implicitly by whether
  `reference_text` is supplied.
- some short / emoji-only inputs make the model emit no audio; both endpoints
  fail with detail 'The model produced no audio for
  the supplied text.' (500 for `/synthesize`; the same failure on `/stream`
  is caught before the streaming response is committed, so it is also a
  normal 500, never a 200 followed by an empty body).

## Differences from `server_qwen3TTS.py`

This is a first version focused on a tight, apples-to-apples comparison of
the two backends' `/synthesize` endpoints. `POST /synthesize` intentionally
does **not** implement:

- `x_vector_only_mode` (speaker-embedding-only cloning) on `/synthesize` --
  `reference_text` is required instead of optional. (`POST /stream` *does*
  support x-vector cloning -- see below -- but `/synthesize`'s contract is
  deliberately left untouched.)
- Long-text chunking beyond mlx-audio's own newline-based segmentation.
- Voice-library / preset-voice profiles.
- `speed`.
- OpenAI API compatibility.
- `temperature` / `top_p` / `repetition_penalty` -- inspection of the
  installed mlx-audio source (`mlx_audio/tts/models/qwen3_tts/qwen3_tts.py`)
  confirms `Model.generate()` does accept these, but they are left out of
  this first version to keep the request surface identical to the fields
  compared in the A/B test. They can be added later without touching the
  core comparison.

`seed` **is** exposed on both endpoints: MLX's global PRNG (`mx.random.seed()`)
drives the talker's token sampling (`mx.random.categorical` in mlx-audio's
own sampling code), so seeding is genuine and verified, not guessed.

## Streaming (`POST /stream`)

Native MLX model-level streaming: mlx-audio's own incremental decoder
(`Model.generate(..., stream=True)`) yields audio as it is generated. This is
**not** the project's long-text chunking mechanism -- there is no text
splitting, no silence joining, no crossfade. `/synthesize` is completely
unaffected by this endpoint's existence.

Request fields (separate `StreamRequest` model -- `/synthesize`'s
`SynthesisRequest` is never weakened to support this):

| Field | Required | Notes |
|---|---|---|
| `text` | yes | same validation as `/synthesize` |
| `audio_base64` | yes | same validation as `/synthesize` |
| `reference_text` | **no** | presence/absence selects the conditioning mode -- see below. A *supplied* blank/whitespace value is rejected (422), never silently treated as absent. |
| `language` | no | same semantics as `/synthesize` |
| `seed` | no | same semantics as `/synthesize` |
| `streaming_interval` | no | seconds of **generated audio** per streamed chunk (not a wall-clock delay); default `2.0` |

**Conditioning mode** -- there is no explicit `mode` parameter:

- Nonblank `reference_text` **supplied** -> ICL (in-context learning), matching
  `/synthesize`'s conditioning.
- `reference_text` **omitted or null** -> x-vector (speaker-embedding) cloning.
  mlx-audio's `Model.generate()` makes exactly this same presence/absence
  decision internally.

**`streaming_interval`** is measured in seconds of audio the model
accumulates before yielding one chunk, *not* how long the client waits: the
default `2.0` means mlx-audio accumulates roughly two seconds of generated
speech before emitting a normal full-sized chunk, and the actual wall-clock
time to produce that chunk is typically much shorter (streaming exists to
reduce time-to-first-audio, not to slow generation down). Smaller values
(mlx-audio's own examples use values as low as `0.32`) trade lower latency
for more per-chunk overhead; there is no documented or enforced upper bound.
The value must be finite and greater than 0 (zero, negative, `NaN`, and
`+/-Infinity` are all rejected with 422, before any generation starts). The
**final chunk of a stream may contain less audio** than the configured
interval. `streaming_context_size` (an mlx-audio `Model.generate()`
parameter) is **not** exposed: it is not honored by the single-request
generation path this server calls.

**Response format** is fixed (no output-format negotiation): raw headerless
IEEE-754 float32 PCM, little-endian (`pcm_f32le`), mono, 24 kHz, media type
`application/octet-stream`. Response headers `X-Audio-Format`,
`X-Sample-Rate`, and `X-Audio-Channels` echo the same fixed shape, which is
also advertised at `GET /capabilities` under the `streaming` block. There is
no WAV container and no base64 encoding -- chunks are written to the response
body as soon as mlx-audio produces them, without buffering the complete
utterance.

**First-chunk commitment**: the server obtains the first audio chunk from
the model *before* returning HTTP 200. If reference-audio validation fails,
if generation raises before producing anything, or if generation produces no
audio at all, the client gets a normal HTTP error response (400/500) --
never a 200 followed by an empty or broken stream. Once the first chunk has
been sent, a later failure can no longer change the HTTP status (the
response is already committed); the stream simply ends, and the server
performs its cleanup regardless (closing the underlying MLX generator,
resetting the streaming decoder state, clearing the MLX cache, and releasing
the shared synthesis lock so the next request is unaffected).

**Cancellation**: there is no `/stop` endpoint, no session ID, and no
pause/resume protocol. The HTTP request *is* the stream's lifetime -- closing
or aborting the client's request triggers cooperative cleanup at a generation
boundary. An already-running synchronous generation step may finish; once
disconnect or cancellation is observed, no further chunk is intentionally
generated. The iterator is explicitly closed on its owning worker, entering
the existing engine cleanup path before response execution finishes. Playback
pause/resume is entirely a client-side concern.

**Thread ownership and backpressure**: the local `_MLXStreamingResponse`
overrides only Starlette's streaming execution method (plus initialization).
One non-abandoning AnyIO worker invocation owns reference preparation,
generator creation, first prefetch, every advancement, PCM conversion, close,
and cleanup. ASGI sends and disconnect polling execute on the event loop via
worker-to-loop calls; each send completes before the next chunk is generated.
There is no producer queue. Starlette's inherited response lifecycle retains
ASGI 2.4 send-error handling and the older disconnect listener; worker boundary
checks observe both explicit disconnects and listener-triggered cancellation.

**Locking**: `/stream` shares the exact same synthesis lock as `/synthesize`.
Only one generation (streaming or not) runs against the shared MLX model at
a time.

**Discovery**: `GET /capabilities` advertises streaming support
presence-based -- the `streaming` key is present (with `endpoint`, `format`,
`sample_rate`, `channels`, `voice_conditioning`) because this engine supports
it; engines that don't stream omit the key entirely. This block describes
transport and conditioning support, not the full streaming request parameter
list; use the table above or the `StreamRequest` schema in `/docs`.

### Verification (2026-09-27)

**Portable automated verification** uses fake engines and in-memory reference
audio, with no MLX, Metal, model weights, network, or platform-specific test
assets required. Final branch verification:

- `python -m pytest impl/tests/test_server_qwen3TTS_mlx.py`: 73 passed.
- `python -m pytest tts-engine-common/tests/ impl/tests/ tools/tests/`: 616 passed.
- `git diff --check`: passed.

Coverage includes multi-chunk ICL/x-vector dispatch, one-worker thread
affinity, first-chunk-before-HTTP-200 errors, real FastAPI/Starlette ASGI
disconnect and send-failure paths, cancellation during an active generation
step, backpressure, cleanup, and subsequent resource reuse.

**Native Apple Silicon/MLX verification** was manually performed and reported
by the maintainer, separately from the portable suite. The multi-chunk and
abort tests used `streaming_interval=0.32`; all completed requests below
returned HTTP 200.

| Native operation | TTFB (s) | Total (s) | Response bytes |
|---|---:|---:|---:|
| Multi-chunk ICL | 0.343528 | 4.425419 | 1,367,040 |
| Multi-chunk x-vector | 0.151144 | 4.699508 | 1,528,320 |
| Immediate `/stream` after abort, no restart | 0.144643 | 1.398792 | 437,760 |
| `/synthesize` after abort, no restart | 1.669601 | 1.669930 | 266,452 |

The ICL output decoded to 341,760 float32 samples (14.24 seconds at 24 kHz
mono), with minimum -0.2793789207935333 and maximum 0.26919853687286377.
Manual listening judged the audio correct and high quality. Both ICL and
x-vector completed across multiple chunks without an MLX thread-affinity
exception. Streaming byte counts are raw PCM; the `/synthesize` byte count is
the complete HTTP response, not raw PCM.

For the real disconnect test, a long ICL request (`text_len=3090`) returned
HTTP 200 and `pcm_f32le`. The client received 30,720 bytes / 7,680 samples /
0.32 seconds of audio, then explicitly closed the HTTP response. The server
remained operational. The immediate follow-up stream's 0.144643-second TTFB,
without a restart, resolves the previously reproduced approximately
559.5-second wait behind abandoned generation in this tested scenario.
The subsequent non-streaming synthesis also succeeded; its log reported
1.7 seconds wall-clock, 4.2 seconds audio, and RTF 0.400, demonstrating reuse
of the shared synthesis resource.

**Repeated aborts and memory caveat**: ten consecutive long ICL requests each
returned 0.32 seconds of PCM before explicit client disconnect, followed by
the next request without restarting. Every abort/recovery cycle took
approximately 1.65–1.68 seconds, with no server exceptions.

| Checkpoint | Process RSS (KB, as reported) |
|---|---:|
| Baseline | 3,295,360 |
| 01 | 3,293,824 |
| 02 | 3,302,832 |
| 03 | 3,309,664 |
| 04 | 3,312,000 |
| 05 | 3,317,296 |
| 06 | 3,323,088 |
| 07 | 3,327,856 |
| 08 | 3,332,864 |
| 09 | 3,331,232 |
| 10 | 3,333,760 |

Some memory retention/growth was observed: final RSS increased by 38,400 KB
(approximately 37.5 MiB, or 1.2% of baseline). Growth was small relative to
process size and was not strictly monotonic. This ten-run sample does
**not** prove zero memory leak. No operational degradation, lock wedging,
or runaway behavior was observed; longer-duration memory stability remains
unestablished.

## Installation

Start by setting up a venv (or use your conda setup):

```
mkdir Qwen3TTSMLX
cd Qwen3TTSMLX
python3 -m venv .venv
source .venv/bin/activate
```

Now install mlx-audio in this environment (requires Apple Silicon):

```
pip install -U mlx-audio
```

Now clone `tts-serve` and install its dependencies:

```
git clone https://github.com/scorbo2/tts-serve
cd tts-serve
pip install ./tts-engine-common fastapi uvicorn loguru soundfile
```

Start it up!

```
python impl/server_qwen3TTS_mlx.py
```

### Changing host

By default, `0.0.0.0` is used. To force a local-only server:

```
export QWEN3TTS_MLX_HOST=127.0.0.1
python impl/server_qwen3TTS_mlx.py
```

### Changing port

By default, the server uses port `7500`, matching the other Qwen3-TTS
implementation.

To choose a different port:

```bash
export QWEN3TTS_MLX_PORT=8600
python impl/server_qwen3TTS_mlx.py
```

### To use a different model

By default, the server downloads
`mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` from HuggingFace on first run.
To use a different MLX-converted checkpoint or a local path:

```
export QWEN3TTS_MLX_MODEL=/path/to/model/
python impl/server_qwen3TTS_mlx.py
```
