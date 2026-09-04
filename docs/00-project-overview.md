# Project overview and goals

There are a large (and growing) number of open-source TTS engines that can be cloned and run locally. Some of them ship with a demo web application, and some of them ship with a REST API so that they can be accessed programmatically. The problem is that each engine supplies a different set of configuration parameters, OR the same/similar configuration parameters, but with different names. An application that wants to offer support for multiple TTS engines must therefore either support a lowest-common-denominator set of options, OR present options specific to each supported TTS engine (a heavy lift for the UI).

The `tts-serve` project aims to provide a single REST API with discoverable configuration parameters. Client applications only need to add support for `tts-serve` - the client application doesn't need to know or care which specific TTS engine it is connected to.

A shared package `tts-engine-common` (pure fastapi + pydantic, no torch) will be built that can be used as a basis for building new engine-specific scripts.

## Goal 1: Define common properties

TTS requests and responses should have a certain bare minimum parameter set. These are the "core" parameters.

For a TTS request:
- the text to be synthesized into speech
- the language of the text to be synthesized
- (for voice cloning) the reference audio (base64-encoded)
- (for voice cloning) the reference audio transcript (text)
- (for voice cloning) the language of the reference audio (2-letter language code, e.g. `en`, `de`, `fr`, etc.)
- seed: an integer seed value (optional)

For a TTS response:
- the synthesized audio (base64-encoded)
- sample rate
- seed (echoed if provided as input, else the seed that was randomly generated)
- time used to generate response
- rtf (real-time factor = time used to generate response / audio duration)

## Goal 2: Define a discoverability mechanism

Any request or response parameter not covered by the common set falls under discoverability. Client applications need a way (`/capabilities` endpoint or similar) to discover what additional parameters are supported by the TTS engine. Details of this mechanism will be covered in a separate document.

The client application can dynamically build a UI to allow the user to enter values for these additional parameters.

## Goal 3: Make adding new TTS engines easier

It should be possible to develop an agent skill to analyze a new TTS engine's Python API and build a new implementation script.

General workflow:
- clone the new TTS engine
- inspect the README and the code to determine capabilities
- add a new implementation script that uses `tts-engine-core` to map common parameters and expose any unique parameters.
- roll out the implementation script and test it with the client application against a running instance of the new engine.

