# Qwen3-TTS MLX Chunking Benchmark

This document records a controlled long-form synthesis comparison for the
Qwen3-TTS MLX backend.

The goal was to evaluate the practical effect of server-side text chunking on:

- synthesis wall-clock time
- real-time factor (RTF)
- generated audio duration
- pause structure
- long-context stability
- chunk-boundary behavior

This is an engineering benchmark, not a universal model-quality claim.

The test used one long Italian source text and one fixed seed. Results should
therefore be interpreted as empirical observations for this workload.

## Test setup

The source was a 3,587-character Italian lesson from the chapter
`Città e paesi`.

The same source text, voice references, seed, and MP3 encoding path were used
for all runs.

Tested voices:

- Pier
- Renzo
- Angelica
- Fabiola
- Marzia
- Mayah

Seed:

```text
777
```

The source text remained below the client's 4,096-character outer-request
limit, so each voice was generated using exactly one HTTP request in both the
unchunked and chunked conditions.

This is important because it isolates the effect of the MLX server's internal
chunking rather than mixing it with client-side outer splitting.

## Conditions

### A — Unchunked

```text
chunking_enabled = false
```

The complete 3,587-character text was passed to one model generation call.

### B — Chunked, direct splice

```text
chunking_enabled = true
chunk_min_chars = 250
chunk_max_chars = 500
chunk_silence_ms = 0
chunk_crossfade_ms = 0
```

The server split the source into the same eight chunks for every voice:

```text
478
485
463
496
498
486
381
293
```

These are character counts.

The splitter preferred natural punctuation boundaries, so most joins occurred
at sentence or phrase boundaries.

Each intentional text chunk was synthesized independently and the resulting
chunk waveforms were concatenated directly.

No artificial silence was inserted.

## Synthesis performance

### Per-voice results

| Voice    | Wall unchunked | Wall chunked | Δ wall | Audio unchunked | Audio chunked | Δ audio | RTF unchunked | RTF chunked |
| -------- | -------------: | -----------: | -----: | --------------: | ------------: | ------: | ------------: | ----------: |
| Pier     |        118.1 s |       86.4 s | -26.8% |         245.4 s |       281.0 s |  +14.5% |         0.481 |       0.308 |
| Renzo    |        104.8 s |       81.4 s | -22.3% |         226.2 s |       260.8 s |  +15.3% |         0.463 |       0.312 |
| Angelica |        103.9 s |       78.9 s | -24.1% |         224.3 s |       254.5 s |  +13.5% |         0.463 |       0.310 |
| Fabiola  |         78.8 s |       79.8 s |  +1.3% |         183.8 s |       244.8 s |  +33.2% |         0.428 |       0.326 |
| Marzia   |        161.2 s |       84.7 s | -47.5% |         327.7 s |       258.2 s |  -21.2% |         0.492 |       0.328 |
| Mayah    |        108.5 s |      101.4 s |  -6.5% |         228.7 s |       288.7 s |  +26.2% |         0.474 |       0.351 |

### Aggregate results

```text
Unchunked:
  total wall time: 675.3 s
  total audio:     1436.1 s
  aggregate RTF:   0.470

Chunked:
  total wall time: 512.6 s
  total audio:     1588.0 s
  aggregate RTF:   0.323
```

Chunking reduced aggregate synthesis wall time by approximately:

```text
24.1%
```

Aggregate RTF improved from:

```text
0.470 -> 0.323
```

### Interpreting RTF

RTF is:

```text
generation wall-clock time / generated audio duration
```

Examples:

```text
RTF 1.0  = real-time generation
RTF 0.5  = approximately 2× real-time
RTF 0.25 = approximately 4× real-time
```

Lower values are faster.

RTF must be interpreted carefully in this experiment because generated audio
duration changed significantly between the two conditions.

A lower RTF does not by itself prove better synthesis quality.

For example, longer generated audio could result from:

- slower pacing
- additional pauses
- repeated speech
- regenerated material
- other long-context model behavior

Likewise, shorter audio could represent:

- cleaner synthesis
- faster pacing
- omitted content
- reduced repetition

Listening or transcript alignment is therefore required alongside RTF.

## Audio-duration and silence analysis

The generated MP3 files were analyzed with ffmpeg and ffprobe.

Silence detection used:

```text
threshold: -40 dB
minimum silence duration: 0.25 s
```

The analysis measured:

- total audio duration
- detected silence duration
- estimated non-silent speech duration
- number of detected silence regions
- average silence duration
- maximum silence duration
- integrated loudness
- loudness range
- true peak

### Duration and silence summary

| Voice    |   Dur A |   Dur B |   Δ Dur | Silence A | Silence B | Δ Silence | Δ Speech |
| -------- | ------: | ------: | ------: | --------: | --------: | --------: | -------: |
| Angelica | 224.3 s | 254.5 s | +30.1 s |    29.7 s |    41.6 s |   +11.9 s |  +18.3 s |
| Fabiola  | 183.8 s | 244.8 s | +60.9 s |    23.5 s |    42.3 s |   +18.9 s |  +42.0 s |
| Marzia   | 327.7 s | 258.2 s | -69.5 s |    22.2 s |    50.0 s |   +27.8 s |  -97.2 s |
| Mayah    | 228.7 s | 288.7 s | +60.0 s |    21.8 s |    45.8 s |   +23.9 s |  +36.1 s |
| Pier     | 245.4 s | 281.0 s | +35.5 s |    35.9 s |    52.6 s |   +16.7 s |  +18.9 s |
| Renzo    | 226.2 s | 260.8 s | +34.7 s |    25.7 s |    53.5 s |   +27.8 s |   +6.9 s |

A = unchunked  
B = chunked

Every chunked output contained more detected silence than its corresponding
unchunked output.

Because:

```text
chunk_silence_ms = 0
```

during this test, these additional pauses were generated by the model rather
than inserted by the server.

This suggests that shorter independent generations tend to restore more
sentence-level pause structure.

## Silence-event detail

### Angelica

```text
silence count:    78 -> 96
silence total:    29.74 s -> 41.61 s
average silence:  0.381 s -> 0.434 s
maximum silence:  0.659 s -> 0.901 s
silence fraction: 13.26% -> 16.35%
```

### Fabiola

```text
silence count:    57 -> 98
silence total:    23.45 s -> 42.35 s
average silence:  0.411 s -> 0.432 s
maximum silence:  0.923 s -> 0.735 s
silence fraction: 12.76% -> 17.30%
```

### Marzia

```text
silence count:    49 -> 100
silence total:    22.24 s -> 50.01 s
average silence:  0.454 s -> 0.500 s
maximum silence:  0.693 s -> 0.770 s
silence fraction: 6.79% -> 19.37%
```

### Mayah

```text
silence count:    57 -> 105
silence total:    21.83 s -> 45.76 s
average silence:  0.383 s -> 0.436 s
maximum silence:  0.646 s -> 0.735 s
silence fraction: 9.54% -> 15.85%
```

### Pier

```text
silence count:    87 -> 113
silence total:    35.93 s -> 52.61 s
average silence:  0.413 s -> 0.466 s
maximum silence:  0.771 s -> 0.822 s
silence fraction: 14.64% -> 18.72%
```

### Renzo

```text
silence count:    65 -> 103
silence total:    25.74 s -> 53.49 s
average silence:  0.396 s -> 0.519 s
maximum silence:  0.940 s -> 1.008 s
silence fraction: 11.38% -> 20.51%
```

## Duration decomposition

For the voices whose chunked output became longer, part of the additional
duration came from extra pauses and part from additional estimated non-silent
speech.

Approximate decomposition:

```text
Angelica:
  extra duration: 30.1 s
  ~39% silence
  ~61% non-silent speech

Fabiola:
  extra duration: 60.9 s
  ~31% silence
  ~69% non-silent speech

Mayah:
  extra duration: 60.0 s
  ~40% silence
  ~60% non-silent speech

Pier:
  extra duration: 35.5 s
  ~47% silence
  ~53% non-silent speech

Renzo:
  extra duration: 34.7 s
  ~80% silence
  ~20% non-silent speech
```

This means the duration increase cannot be explained purely by pauses for
every voice.

Fabiola and Mayah in particular produced substantially more estimated
non-silent material when chunked.

Without transcript alignment, it is not possible to determine from these
numbers alone whether that represents:

- slower articulation
- additional legitimate speech
- repeated material
- other generation differences

## Marzia long-context anomaly

Marzia was the strongest outlier in the test.

### Timing

```text
Unchunked:
  wall:  161.2 s
  audio: 327.7 s
  RTF:   0.492

Chunked:
  wall:   84.7 s
  audio: 258.2 s
  RTF:   0.328
```

Wall time decreased by approximately:

```text
47.5%
```

Generated audio duration decreased by approximately:

```text
69.5 s
```

At the same time, detected silence increased by approximately:

```text
27.8 s
```

Estimated non-silent speech therefore decreased by approximately:

```text
97.2 s
```

That combination is unusual:

```text
more pauses
+
much less non-silent speech
+
much lower total duration
```

This strongly suggests that the unchunked Marzia run behaved differently from
the other voices under long-context generation.

The measurements are consistent with a possible long-context degeneration
mode, such as repetition or stretched/restarted generation.

However, the acoustic measurements alone do not prove the exact failure mode.
Listening or timestamped transcript alignment is required to classify it.

## Loudness analysis

The largest acoustic difference was again Marzia.

### Marzia

```text
Unchunked:
  integrated loudness: -24.0 LUFS
  loudness range:       9.2 LU
  true peak:           +1.0 dBTP

Chunked:
  integrated loudness: -28.7 LUFS
  loudness range:       4.2 LU
  true peak:           -9.2 dBTP
```

The unchunked output was therefore substantially hotter and more dynamically
variable.

The reconstructed MP3 true peak above 0 dBTP also indicates possible
inter-sample clipping risk in that output.

Other voices showed smaller changes.

### Loudness summary

```text
Angelica:
  LUFS:      -32.8 -> -30.5
  LRA:         4.1 -> 4.2
  true peak: -12.7 -> -9.5 dBTP

Fabiola:
  LUFS:      -34.5 -> -33.9
  LRA:         2.8 -> 3.9
  true peak: -12.8 -> -9.9 dBTP

Marzia:
  LUFS:      -24.0 -> -28.7
  LRA:         9.2 -> 4.2
  true peak:  +1.0 -> -9.2 dBTP

Mayah:
  LUFS:      -30.9 -> -29.6
  LRA:         2.9 -> 3.5
  true peak: -11.9 -> -10.4 dBTP

Pier:
  LUFS:      -28.9 -> -26.6
  LRA:         4.6 -> 5.0
  true peak:  -8.0 -> -6.6 dBTP

Renzo:
  LUFS:      -27.5 -> -28.7
  LRA:         4.1 -> 3.1
  true peak:  -5.0 -> -5.7 dBTP
```

## Listening results

The first approximately 90 seconds of both unchunked and chunked output were
listened to directly.

Both versions were considered usable.

The chunked version did not exhibit an obvious overall quality regression.

A small audible blip was noticed at one chunk boundary, which appeared to be
related to directly concatenating independently generated waveforms.

This led to the addition of optional crossfade support.

## Crossfade experiment

A third Marzia run was performed using the same source text, voice reference,
seed, and chunk boundaries.

Configuration:

```text
chunking_enabled = true
chunk_min_chars = 250
chunk_max_chars = 500
chunk_silence_ms = 0
chunk_crossfade_ms = 10
```

The text was split into exactly the same eight chunks:

```text
478
485
463
496
498
486
381
293
```

### Results

```text
Chunked, direct splice:
  wall:  84.7 s
  audio: 258.2 s
  RTF:   0.328

Chunked, 10 ms crossfade:
  wall:  76.7 s
  audio: 258.1 s
  RTF:   0.297
```

The expected duration reduction from seven 10 ms overlaps is:

```text
7 × 10 ms = 70 ms
```

The measured audio-duration change was consistent with that expected overlap.

The wall-clock difference should not be interpreted as evidence that
crossfading improves model-generation speed.

The crossfade operation itself is computationally trivial relative to model
generation, so the timing difference is most likely ordinary run-to-run
variation.

## Crossfade listening result

The 10 ms crossfade was perceptually more noticeable than the original direct
splice on this material.

It sounded like a more obvious break even though the server inserted no
silence.

The likely reason is that the text splitter was already cutting at natural
sentence boundaries.

Each independently generated chunk therefore already contained appropriate
sentence-ending or sentence-starting timing.

A 10 ms overlap modified that natural boundary enough to become perceptible.

This does not make crossfade undesirable as a feature.

Instead, it suggests that:

```text
direct splice = good default for natural punctuation boundaries
small crossfade = useful corrective tool for a specific audible seam
```

Crossfade therefore remains optional and defaults to:

```text
chunk_crossfade_ms = 0
```

Smaller values such as:

```text
2 ms
5 ms
```

are better candidates for further experimentation than 10 ms.

## Conclusions

For this long-form Italian workload, server-side chunking produced a clear
computational improvement.

Aggregate wall-clock generation time decreased by approximately:

```text
24.1%
```

Aggregate RTF improved from:

```text
0.470 -> 0.323
```

The effect was not uniform across every voice, but five of six voices generated
faster in wall-clock time.

Chunking also materially changed the model's temporal behavior.

Every chunked output contained more detected pause structure, despite
`chunk_silence_ms=0`, showing that the additional pauses came from the model
rather than server-inserted silence.

Marzia provided the strongest evidence that shorter independent generations
may avoid a long-context degeneration mode. The unchunked output was much
longer, much hotter, and contained approximately 97 seconds more estimated
non-silent speech than the chunked version.

The benchmark also showed that chunk joining is a separate problem from text
chunking itself.

Direct joins at natural sentence boundaries can sound very good. Crossfade is
valuable as an optional smoothing mechanism but should not automatically be
applied to every boundary.

The practical configuration resulting from this experiment remains:

```text
chunking_enabled = true
chunk_min_chars = 250
chunk_max_chars = 500
chunk_silence_ms = 0
chunk_crossfade_ms = 0
```

with small crossfade values available when an audible splice needs correction.

## Limitations

This benchmark used:

- one source text
- one language
- six voices
- one seed
- one long-form text length
- one MLX model/checkpoint
- one machine
- one silence-detection threshold

The results therefore should not be generalized into universal performance or
quality claims.

For stronger validation, future testing could include:

- multiple seeds
- multiple source texts
- multiple text lengths
- other supported languages
- timestamped ASR transcription
- reference-text alignment
- omission/repetition detection
- multiple silence thresholds
- additional perceptual listening tests

In particular, transcript alignment would help distinguish legitimate pacing
changes from duplicated, omitted, or regenerated speech.
