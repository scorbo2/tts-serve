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
    def device_count() -> int:
        # Must stay consistent with is_available(): a stubbed box has no GPU.
        # Servers that reject an out-of-range 'cuda:<index>' need this to exist.
        return 0

    @staticmethod
    def empty_cache() -> None:
        pass

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        pass


cuda = _Cuda()


def manual_seed(seed: int) -> None:
    pass


class _Device:
    """Stand-in for ``torch.device`` (parsed enough for the servers' use)."""

    def __init__(self, spec: str) -> None:
        if not isinstance(spec, str):
            spec = str(spec)
        self.type = spec.split(":", 1)[0]
        self.index = int(spec.split(":", 1)[1]) if ":" in spec else None

    def __repr__(self) -> str:
        return f"torch.device({self.type})"


def device(spec: str) -> _Device:
    return _Device(spec)


class Generator:
    """Stand-in for the per-request RNG handle a server hands to a sampler.

    Nothing in the test suite samples from it (the engine boundary is stubbed),
    so it only has to be constructible and seedable.  Real torch is used on any
    machine that has it installed.
    """

    def __init__(self, device=None) -> None:
        self.device = device
        self.seed = None

    def manual_seed(self, seed: int) -> "Generator":
        self.seed = seed
        return self


class _Dtype:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"torch.{self.name}"


# dtype placeholders (referenced by precision-mapping code paths)
float16 = _Dtype("float16")
float32 = _Dtype("float32")
bfloat16 = _Dtype("bfloat16")
