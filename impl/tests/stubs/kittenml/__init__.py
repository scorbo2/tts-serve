"""Stub of the ``kittenml`` package (test machines without the real engine).

``server_kittenTTS`` imports only ``KittenTTS`` from the package root; the
constructor's shape mirrors the real one (kittenml/get_model.py) so the
server's keyword arguments (device/decoder/weights) bind exactly as they
would against the real package.  Model loading is never exercised in tests,
so instantiation raises loudly instead of pretending to synthesize.

No constants need mirroring: the server's sample-rate constant was verified
against the engine's own ``S3GEN_SR`` at research time, and the voice /
preset / decoder tables are per-repository data the engine only knows after
a (never-executed) model load.
"""

__version__ = "0.9.7"  # mirrors the real PyPI release the suite was checked against


class KittenTTS:
    """Placeholder -- real model loading is never exercised in the tests."""

    def __init__(
        self,
        model_name="KittenML/kitten-tts-nano-0.8",
        cache_dir=None,
        backend=None,
        device=None,
        hf_token=None,
        decoder=None,
        weights=None,
        vllm_options=None,
        mlx_options=None,
    ):
        raise NotImplementedError(
            "kittenml stub: loading a model is not available in tests"
        )
