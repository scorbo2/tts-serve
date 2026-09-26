"""Import-only stand-in for ``breeze_infer.runtime`` (test machines only).

Real model loading never runs in the test suite: ``load_runtime`` raises if
a test accidentally reaches it.  The seed / generation-config helpers are
no-ops so the fake-model success path can exercise the request plumbing
without a real engine.
"""


def load_runtime(*args, **kwargs):
    raise NotImplementedError(
        "breeze_infer stub: load_runtime() is not available in tests"
    )


def resolve_device(explicit_device=None):
    raise NotImplementedError(
        "breeze_infer stub: resolve_device() is not available in tests"
    )


def set_all_seeds(seed):
    # no-op: the tests pin no real RNG.
    pass


def update_generation_config_for_breeze(model, generation_config=None):
    # no-op: the stub model has no generation config to patch.
    pass
