"""Shared contract for the ``<PREFIX>_DEVICE`` environment variables.

Every server that lets the operator pick a device reads one string out of the
environment and has to turn it into a decision *before* a checkpoint is
downloaded.  Two halves of that job exist, and only one of them is shared:

* the **grammar** half ("is this string a device we can even talk about?") is
  identical everywhere.  It used to be hand-rolled six different ways across
  the servers, with six different answers to ``cuda:9`` -- one server checked
  the ordinal, none of the others did, and two accepted anything that merely
  *started with* "cuda".  That half lives here (issue #73).

* the **probing** half ("does this box actually have a GPU, and how many?")
  needs torch, and ``tts-engine-common`` must stay torch-free (docs/01 D7).
  So the helpers below take the probe *results* as plain bool/int and the
  servers keep the ``torch.cuda`` calls.

Per-engine capability stays per-engine **as data**: each server declares the
device families its engine can actually serve (and whether it can address an
indexed GPU), so nothing silently gains a device its engine cannot run.
"""

from __future__ import annotations

import re
from typing import Callable, NamedTuple

# The device families any tts-serve server may declare.  The names match
# torch's own device types so a parsed family can be handed straight back to
# ``torch.device()``.  ``'auto'`` is not a device at all -- it is a request to
# *choose* one -- which is exactly why the availability guard never rejects it.
DEVICE_FAMILIES: tuple[str, ...] = ("auto", "cpu", "cuda", "mps", "xpu")

# The families torch accepts an ordinal for.  CPU takes no index and MPS is
# single-device, so 'cpu:0'/'mps:0' are grammar errors at this boundary rather
# than a torch surprise halfway through a model load.
INDEXABLE_FAMILIES: tuple[str, ...] = ("cuda", "xpu")

# What a server advertises unless it declares otherwise: the CLI grammar of
# the engines that offer a choice.  xpu is deliberately absent -- only
# IndexTTS's engine mentions it, and a family no engine can serve is a family
# the operator can only get wrong.
DEFAULT_DEVICE_FAMILIES: tuple[str, ...] = ("auto", "cpu", "mps", "cuda")

CUDA_FAMILY = "cuda"

_DEVICE_VALUE = re.compile(r"(?P<family>[a-z]+)(?::(?P<index>\d+))?")


class DeviceSpec(NamedTuple):
    """A parsed ``<PREFIX>_DEVICE`` value: family, optional ordinal, raw text.

    ``requested`` keeps the operator's own spelling so an error message can
    quote back what was asked for instead of a normalized guess.
    """

    family: str
    index: int | None
    requested: str


def _grammar_text(families: tuple[str, ...], allow_index: bool) -> str:
    """Render the accepted spellings for an error message, in declared order."""
    forms = [f"'{family}'" for family in families]
    if allow_index:
        forms += [
            f"'{family}:<index>'" for family in families if family in INDEXABLE_FAMILIES
        ]
    if len(forms) == 1:
        return forms[0]
    return ", ".join(forms[:-1]) + ", or " + forms[-1]


def parse_device(
    value: str,
    *,
    env_var: str,
    families: tuple[str, ...] = DEFAULT_DEVICE_FAMILIES,
    allow_index: bool = True,
    note: str | None = None,
) -> DeviceSpec:
    """Parse one ``<PREFIX>_DEVICE`` value, or raise a message that names it.

    Grammar only -- availability is the caller's business (see
    :func:`cuda_ordinal_error`), because the answer changes between import
    time and load time on a machine with a GPU.

    ``families`` is the *engine's* capability, declared by the server: CUDA-only
    engines pass ``("cuda",)``, engines whose device selection runs through
    ``torch.device`` pass the usual set.  A typo in that declaration is a bug in
    the server, so it fails at import with a message naming the env var rather
    than quietly widening what the server accepts.

    ``allow_index=False`` rejects ``cuda:<index>`` for engines that cannot use
    it -- LuxTTS's ``torch.device(device, 0)`` pins GPU 0 whatever the string
    says, so accepting the syntax there would mean silently loading on the
    wrong GPU.  Pass ``note=`` with the engine's own reason so the operator
    learns *why* the usual syntax is missing, instead of concluding the server
    is broken.

    Values are stripped but deliberately not case-folded: every engine behind
    these servers takes torch's lowercase device strings, and accepting 'CUDA'
    here would only move the confusion to the next layer.
    """
    if not families or any(family not in DEVICE_FAMILIES for family in families):
        raise ValueError(
            f"{env_var} is declared with unknown device families {families!r}; "
            f"pick from {DEVICE_FAMILIES}"
        )

    text = value.strip() if isinstance(value, str) else ""
    match = _DEVICE_VALUE.fullmatch(text)
    family = match["family"] if match else None
    ordinal = match["index"] if match else None

    index = int(ordinal) if ordinal is not None else None
    indexable = family in INDEXABLE_FAMILIES if family else False
    if (
        family is None
        or family not in families
        or (index is not None and (not allow_index or not indexable))
    ):
        grammar = _grammar_text(families, allow_index)
        reason = f" ({note})" if note else ""
        raise ValueError(f"{env_var} must be {grammar}, got {value!r}{reason}")

    return DeviceSpec(family=family, index=index, requested=text)


def cuda_ordinal_error(
    spec: DeviceSpec,
    *,
    available: bool,
    count: int,
    env_var: str,
) -> str | None:
    """Message for a CUDA request this box cannot serve; ``None`` if it can.

    Built from probes the caller already ran (``available`` =
    ``torch.cuda.is_available()``, ``count`` = ``torch.cuda.device_count()``)
    so this package never has to import torch.  Both failures are worth an
    explicit message because torch reports neither of them usefully:

    * CUDA requested on a box without CUDA -- ``torch.device('cuda')`` happily
      constructs, and the first complaint is a device-mapping error from
      inside the model load, i.e. *after* the checkpoint was downloaded.
    * An ordinal past the last GPU -- ``is_available()`` only says *some* GPU
      exists, so ``cuda:9`` on a one-GPU box survives every import-time check
      and dies at ``.to(device)``, again after the download.

    ``'auto'`` never errors here: it asks the server to choose, so a missing
    GPU is an answer, not a failure.  An un-indexed ``cuda`` request is only
    rejected when the box has no CUDA at all -- how many GPUs exist is the
    operator's problem only if they named one.
    """
    if spec.family != CUDA_FAMILY:
        return None
    if not available:
        return (
            f"{env_var}={spec.requested} requests CUDA, but CUDA is not "
            "available on this box"
        )
    if spec.index is not None and spec.index >= count:
        return (
            f"{env_var}={spec.requested} asks for GPU {spec.index}, but this "
            f"box exposes {count} CUDA device(s)"
        )
    return None


def require_cuda_available(
    spec: DeviceSpec,
    *,
    is_available: Callable[[], bool],
    device_count: Callable[[], int],
    env_var: str,
) -> None:
    """Run the CUDA probes and raise unless this box can serve ``spec``.

    A convenience over :func:`cuda_ordinal_error` for the servers' load path:
    the probes are passed as callables and only invoked here, which is how a
    torch-free package can ask a torch question (docs/01 D7), and why tests can
    pin a fake GPU count without a GPU.

    Servers call this *before* touching a checkpoint; the RuntimeError it raises
    is deliberately not a ValueError like :func:`parse_device`'s — a bad grammar
    is a broken launch, a bad device is a broken machine, and the two need
    different fixes.
    """
    if spec.family != CUDA_FAMILY:
        # Nothing to ask the machine: 'cpu'/'mps' ask for no GPU, and 'auto'
        # asked the server to choose, so "no CUDA" is an answer it can use.
        return
    available = is_available()
    message = cuda_ordinal_error(
        spec,
        available=available,
        # Only count GPUs once CUDA is known to exist: torch.cuda.device_count()
        # itself raises on a torch build compiled without CUDA, and "how many?"
        # is a meaningless follow-up to "there are none".
        count=device_count() if available else 0,
        env_var=env_var,
    )
    if message:
        raise RuntimeError(message)
