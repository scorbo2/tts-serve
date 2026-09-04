"""Minimal ``torch`` stub for machines without PyTorch installed.

Only the tiny API surface the tts-serve servers touch at import time (and in
the request paths covered by the tests) is provided.  Real model calls are
never expected in the test suite; anything deeper than the surface below is
intentionally absent so a mistaken real-inference path fails loudly.

The real PyTorch always wins when installed (this stub is appended to
``sys.path`` last by conftest).
"""


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def empty_cache() -> None:
        pass

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        pass


cuda = _Cuda()


def manual_seed(seed: int) -> None:
    pass


class _Dtype:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"torch.{self.name}"


# dtype placeholders (referenced by precision-mapping code paths)
float16 = _Dtype("float16")
float32 = _Dtype("float32")
bfloat16 = _Dtype("bfloat16")
