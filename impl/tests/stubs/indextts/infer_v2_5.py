"""Stub of ``indextts.infer_v2_5`` for test machines without the real package.

The server only needs the name ``IndexTTS2`` at import time (constructor
call and type annotation); real model loading is never exercised in the
tests.
"""


class IndexTTS2:
    """Placeholder — real model loading is never exercised in the tests."""

    def __init__(
        self,
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_bf16=False,
        device=None,
        use_cuda_kernel=None,
        use_deepspeed=False,
        use_accel=False,
        use_torch_compile=False,
        use_qwen_emo=False,
        aux_paths=None,
    ):
        raise NotImplementedError(
            "indextts stub: IndexTTS2() is not available in tests"
        )

    def infer(self, *args, **kwargs):
        raise NotImplementedError("indextts stub: infer() is not available in tests")

    def normalize_emo_vec(self, *args, **kwargs):
        raise NotImplementedError(
            "indextts stub: normalize_emo_vec() is not available in tests"
        )
