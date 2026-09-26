# Voice design

This document describes a significant new feature for tts-serve: Voice Design.

## Current state

Each time support for a new TTS engine is added to tts-serve, only voice cloning capabilities are considered:
- reference audio is mandatory (this is the voice to be cloned)
- reference text is often required (sometimes optional, if the engine uses Whisper or similar)
- reference audio language is often required (sometimes optional)

From these parameters, the engine in question can clone a voice and return a voice response
to a given text prompt using that cloned voice. Our current `/capabilities` and `/synthesize` endpoints
only consider voice cloning options.

## Desired state

Some TTS engines (Qwen3-TTS, Breeze-TTS, and others) also offer "Voice Design" as a feature.
With Voice Design, there is no reference audio or reference audio transcript. Instead, a text
description of the voice to be generated is supplied. Some examples:

- "A young woman's gentle voice"
- "A man speaking in a low and menacing voice"
- "A bright and cheerful female voice"

The exact naming of this instruction parameter varies. Qwen3-TTS calls it `instruct`, Breeze-TTS
calls it `instruction`, but the general concept is the same.

The `tts-serve` library should offer support for Voice Design when using any of the engines that
support it. Engines that do not offer a Voice Design feature should return `501 Not Implemented`
when attempting to access this feature.

### Primary consideration

Addition of this feature **must not break backwards compatibility**! Clients of previous versions
of `tts-serve` should be able to connect, query capabilities, and synthesize a cloned voice exactly
as they did before. This new feature is an **addition, not a breaking change**.

### Design proposal

(Subject to "open questions" as documented below... this is a rough proposal)

**If possible**, the existing `/capabilities` endpoint should be extended to include information
about whether voice design is possible with the current engine. A simple boolean flag like
`voiceDesignSupported` or similar.

**If possible**, the existing `/synthesize` endpoint should be extended to allow Voice Design
as well as voice cloning.

If the existing endpoints cannot be so modified, then new endpoints can be introduced for this
feature (existing clients need not know or care about the new endpoints):

- `GET /voicedesigncapabilities` - returns whether or not Voice Design is an option.
- `POST /voicedesignsynthesize` - submit the voice instructions and text to be generated.

We should normalize the input parameter name. The engine in question may call it `instruct`
or `instruction` or some other name, but `tts-serve` should provide generic access to this
feature with a single generic parameter name: `instructions`.

### Open questions

- Can the existing `/capabilities` endpoint be extended to return information about Voice Design
  without breaking or confusing existing clients?
- Can the existing `/synthesize` endpoint be modified to allow either cloning OR voice design?
  Or do we need a new `/voicedesign` endpoint or similar?
