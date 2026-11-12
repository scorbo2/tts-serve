"""Minimal ``transformers`` stub for machines without Transformers installed.

For this engine ``transformers`` *is* the engine dependency: Audio8 publishes
no Python package of its own -- the model and processor code ships inside the
checkpoint (``auto_map`` in its config.json) and is loaded through
``AutoModel`` / ``AutoProcessor`` with ``trust_remote_code=True``.  So this
stub stands in for the engine import, exactly like ``stubs/chatterbox/``.

Only the two loader entry points a server module touches at import time are
provided; both raise, since a real load needs the real package plus the
checkpoint.  Request paths that reach the model are exercised in tests with a
fake runtime, never through this stub.

The real package always wins when installed (the stubs are appended to
``sys.path`` last).
"""


class _StubModelLoader:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError(
            "transformers stub: from_pretrained() needs the real package "
            "and the Audio8 checkpoint"
        )


class _StubProcessorLoader:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError(
            "transformers stub: from_pretrained() needs the real package "
            "and the Audio8 checkpoint"
        )


AutoModel = _StubModelLoader
AutoProcessor = _StubProcessorLoader
