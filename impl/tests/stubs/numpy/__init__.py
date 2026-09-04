"""Minimal ``numpy`` stub for machines without NumPy installed.

The servers reference ``np`` only inside request handlers (``np.random.seed``,
``np.clip``) and in deferred annotations (``from __future__ import
annotations``), so importability plus a couple of no-op helpers is enough.

Note: pytest's ``approx`` introspects ``sys.modules["numpy"]`` whenever a
module by that name is importable (``_as_numpy_array``), so this stub must
also provide ``isscalar`` / ``ndarray`` / ``asarray`` or cross-suite test
runs blow up with AttributeError.
"""

import random as _stdlib_random


class ndarray:
    """Placeholder so ``np.ndarray`` resolves if ever evaluated eagerly."""


def isscalar(obj) -> bool:
    # numpy returns True for scalars (numbers, strings, numpy scalars) and
    # False for array-likes.  Good enough for pytest's introspection: test
    # objects here are plain floats/ints.
    return not hasattr(obj, "__iter__")


# pytest's ApproxBase.is_bool does isinstance(val, np.bool_); the built-in
# bool is the right stand-in since the stub can never produce a numpy bool.
bool_ = bool


def ndindex(*shape):
    raise NotImplementedError("numpy stub: ndindex() is not available in tests")


def clip(a, a_min, a_max):
    raise NotImplementedError("numpy stub: clip() is not available in tests")


def asarray(obj, dtype=None):
    raise NotImplementedError("numpy stub: asarray() is not available in tests")


# ``np.random.seed(...)`` in the servers maps onto the stdlib RNG.
random = _stdlib_random
