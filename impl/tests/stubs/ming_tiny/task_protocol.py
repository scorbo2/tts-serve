"""Stub for ``ming_tiny.task_protocol`` (test machines only).

The server imports the engine's "formal" sampler defaults rather than
hardcoding its own, so the stub must mirror the vendor's values *exactly*.
Source: Ming-3.0-tts ``ming_tiny/task_protocol.py`` (``FORMAL_CFG``,
``FORMAL_TEMPERATURE``, ``FORMAL_FLOW_STEPS``, ``FORMAL_SIGMA``).  If the
vendor changes these numbers, the server's defaults follow automatically on
a real install — but regenerate this stub too, so the test suite reflects
the same defaults the servers would load.
"""

FORMAL_CFG = 2.0
FORMAL_TEMPERATURE = 0.0
FORMAL_FLOW_STEPS = 10
FORMAL_SIGMA = 0.25
