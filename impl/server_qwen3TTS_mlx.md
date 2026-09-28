# Qwen3-TTS (MLX)

The Apple-Silicon-native counterpart to [Qwen3-TTS](server_qwen3TTS.md).

It runs the same Qwen3-TTS Base checkpoint family through
[mlx-audio](https://github.com/Blaizzy/mlx-audio) instead of
`qwen_tts`/PyTorch.

The server mirrors `server_qwen3TTS.py`'s core request/response shape as
closely as mlx-audio's API allows.

Quick stats:

- **Server script**: [server_qwen3TTS_mlx.py](server_qwen3TTS_mlx.py)
- **Default model**: `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit`
- **Sample rate**: 24 kHz
- **Default port**: `7500`
- **Device**: reported as `mlx` (no device-selection env var -- mlx-audio has
  no such knob; MLX itself picks its compute backend, Metal on Apple Silicon)
- **Cloning mode**: ICL (in-context learning) only
- **Reference transcript**: `reference_text` is required
- **Long-text chunking**: optional
- **Chunk join modes**: direct splice, silence, or linear crossfade
- **Streaming**: not supported
- **OpenAI API compatibility**: not supported
- Some short / emoji-only inputs can make the model emit no audio; the server
  returns `500` with a meaningful error rather than failing with an accidental
  `IndexError`.

## Quick start

### 1. Create an environment

```bash
mkdir Qwen3TTSMLX
cd Qwen3TTSMLX

python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install mlx-audio

This backend requires Apple Silicon.

```bash
pip install -U mlx-audio
```

### 3. Clone and install `tts-serve`

```bash
git clone https://github.com/scorbo2/tts-serve
cd tts-serve

pip install ./tts-engine-common fastapi uvicorn loguru soundfile
```

### 4. Start the server

```bash
python impl/server_qwen3TTS_mlx.py
```

The default server address is:

```text
http://localhost:7500
```

Useful endpoints:

```text
GET  /health
GET  /capabilities
POST /synthesize
GET  /docs
```

## Minimal synthesis request

The MLX server requires text, a reference audio sample, and the exact
transcript of that reference audio.

Example:

```json
{
  "text": "Roma è una città.",
  "audio_base64": "<base64 reference audio>",
  "reference_text": "Exact transcript of the reference audio.",
  "language": "it"
}
```

The response contains 24 kHz WAV audio encoded as base64.

## Configuration

### Changing host

By default, the server binds to `0.0.0.0`.

To restrict it to the local machine:

```bash
export QWEN3TTS_MLX_HOST=127.0.0.1
python impl/server_qwen3TTS_mlx.py
```

### Changing port

The default port is `7500`.

```bash
export QWEN3TTS_MLX_PORT=8600
python impl/server_qwen3TTS_mlx.py
```

### Using a different model

The default model is:

```text
mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit
```

To use another MLX-converted checkpoint or local model path:

```bash
export QWEN3TTS_MLX_MODEL=/path/to/model/
python impl/server_qwen3TTS_mlx.py
```

## Differences from `server_qwen3TTS.py`

This implementation focuses on the core Qwen3-TTS voice-cloning path.

It does **not** currently implement:

- `x_vector_only_mode`
- speaker-embedding-only cloning
- streaming
- voice-library / preset-voice profiles
- `speed`
- OpenAI API compatibility
- `temperature`
- `top_p`
- `repetition_penalty`

`reference_text` is required because the MLX server currently exposes ICL
voice cloning only.

Inspection of mlx-audio confirms that `Model.generate()` supports parameters
such as `temperature`, `top_p`, and `repetition_penalty`; they are simply not
exposed by this server yet.

### Seed behavior

`seed` is supported.

The server seeds MLX's global PRNG with:

```python
mx.random.seed(seed)
```

The valid range is:

```text
1-1000
```

If no seed is supplied, the server selects a random seed in that range and
returns it in the response.

## Long-text chunking

The server supports optional application-level chunking before text is passed
to the model.

Chunking is disabled by default.

Enable it with:

```json
{
  "chunking_enabled": true
}
```

The chunking parameters are:

- `chunking_enabled`
  - Enables server-side text chunking.
  - Default: `false`.

- `chunk_min_chars`
  - Preferred minimum text-chunk size.
  - Default: `250`.

- `chunk_max_chars`
  - Maximum text-chunk size.
  - Default: `500`.

- `chunk_silence_ms`
  - Silence inserted between completed text chunks.
  - Default: `0`.

- `chunk_crossfade_ms`
  - Linear crossfade between completed text chunks.
  - Valid range: `0-50` ms.
  - Default: `0`.

`chunk_min_chars` must be less than or equal to `chunk_max_chars`.

`chunk_silence_ms` and `chunk_crossfade_ms` are mutually exclusive and cannot
both be greater than zero.

### Chunk splitting

The splitter prefers natural linguistic boundaries rather than blindly cutting
at `chunk_max_chars`.

It prefers, in order:

1. Sentence boundaries: `.`, `!`, `?`
2. Weaker punctuation: `;`, `:`, `,`
3. Whitespace
4. Hard splitting at `chunk_max_chars` when no suitable boundary exists

Each intentional text chunk is synthesized independently.

If mlx-audio produces multiple generator outputs for one text chunk, those
outputs are concatenated normally before any chunk-to-chunk join behavior is
applied.

## Chunk joining

### Direct splice

The default is a direct join:

```text
chunk A | chunk B
```

Configuration:

```json
{
  "chunk_silence_ms": 0,
  "chunk_crossfade_ms": 0
}
```

Because the splitter prefers sentence and punctuation boundaries, direct joins
often preserve the model's natural sentence-ending and sentence-starting
timing well.

### Explicit silence

`chunk_silence_ms` inserts fixed silence between completed text chunks.

Example:

```json
{
  "chunking_enabled": true,
  "chunk_min_chars": 250,
  "chunk_max_chars": 500,
  "chunk_silence_ms": 120,
  "chunk_crossfade_ms": 0
}
```

Conceptually:

```text
chunk A | 120 ms silence | chunk B
```

No artificial silence is inserted when the value is `0`.

### Crossfade

`chunk_crossfade_ms` overlaps the end of one completed text chunk with the
beginning of the next and applies a short linear fade between them.

Example:

```json
{
  "chunking_enabled": true,
  "chunk_min_chars": 250,
  "chunk_max_chars": 500,
  "chunk_silence_ms": 0,
  "chunk_crossfade_ms": 5
}
```

Conceptually:

```text
chunk A --------\
                 \--------
                   chunk B
```

During the overlap, the previous chunk fades out while the next chunk fades
in.

Crossfade is intended to smooth an audible waveform seam or small splice
artifact. It does not insert silence.

Because chunk boundaries normally occur at natural punctuation, crossfading
is optional rather than enabled by default. A direct splice may sound more
natural when the generated chunks already contain appropriate sentence-boundary
timing.

Small values such as `2` or `5` ms are reasonable starting points when trying
to smooth a specific audible join.

The default remains:

```text
chunk_crossfade_ms = 0
```

## Example chunked request

Direct joins with the default chunk sizes:

```json
{
  "text": "Text to synthesize...",
  "audio_base64": "<base64 reference audio>",
  "reference_text": "Exact transcript of the reference audio.",
  "language": "it",
  "seed": 777,
  "chunking_enabled": true,
  "chunk_min_chars": 250,
  "chunk_max_chars": 500,
  "chunk_silence_ms": 0,
  "chunk_crossfade_ms": 0
}
```

With a 5 ms crossfade:

```json
{
  "text": "Text to synthesize...",
  "audio_base64": "<base64 reference audio>",
  "reference_text": "Exact transcript of the reference audio.",
  "language": "it",
  "seed": 777,
  "chunking_enabled": true,
  "chunk_min_chars": 250,
  "chunk_max_chars": 500,
  "chunk_silence_ms": 0,
  "chunk_crossfade_ms": 5
}
```

## Performance

A controlled long-form A/B test was run using the same 3,587-character Italian
lesson, six voice references, seed `777`, and identical output encoding.

The comparison was:

```text
A: chunking disabled

B:
  chunking_enabled = true
  chunk_min_chars = 250
  chunk_max_chars = 500
  chunk_silence_ms = 0
  chunk_crossfade_ms = 0
```

Aggregate results:

| Mode      | Wall time | Generated audio |   RTF |
| --------- | --------: | --------------: | ----: |
| Unchunked |   675.3 s |        1436.1 s | 0.470 |
| Chunked   |   512.6 s |        1588.0 s | 0.323 |

For this workload, server-side chunking reduced aggregate synthesis wall time
by about **24%** and improved aggregate RTF from `0.470` to `0.323`.

Chunking also changed output duration and pause structure, so RTF alone should
not be interpreted as a quality score.

Listening tests found both unchunked and chunked output usable. Direct
sentence-boundary joins generally sounded natural, while crossfade was useful
as an optional smoothing mechanism rather than something that should be
enabled automatically.

Full per-voice timing, silence analysis, loudness measurements, the Marzia
long-context anomaly, and the crossfade experiment are documented in:

[Qwen3-TTS MLX chunking benchmark](server_qwen3TTS_mlx_chunking_benchmark.md)

## Capabilities

The complete request schema is available from:

```text
GET /capabilities
```

The capabilities response is generated from the same Pydantic request model
used by `/synthesize`, so defaults, validation limits, and runtime request
metadata remain synchronized.
