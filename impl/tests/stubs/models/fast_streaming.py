"""Import-only stand-in for ``models.fast_streaming`` (test machines only).

``FastStreamingConfig`` is a faithful copy of the engine's frozen dataclass
(the server constructs it at load time with exactly these fields); the
runtime class is a loud placeholder — the real one hard-requires a CUDA
device and a loaded model, and is never instantiated in the tests.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FastStreamingConfig:
    max_new_tokens: int = 750
    max_seq_len: int = 1024
    collect_timing: bool = False
    fast_all: bool | None = None
    fast_text_encoder: bool = False
    fast_backbone_prefill: bool = False
    fast_backbone_decode: bool = False
    fast_depth_decoder: bool = False
    fast_codec: bool = False
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool | None = None
    repetition_penalty: float = 1.1


class FastBreezeStreamingRuntime:
    """Placeholder — real streaming requires a CUDA device and a loaded model."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "models stub: FastBreezeStreamingRuntime() is not available in tests"
        )
