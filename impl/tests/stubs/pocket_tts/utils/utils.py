"""Stub of ``pocket_tts.utils.utils`` (pocket-tts 3.3.0).

Only the predefined-voice names matter to the server (they become the
``voice`` enum), so the values are shortened placeholders; the key order is
a faithful copy of the real dict.
"""

_ORIGINS_OF_PREDEFINED_VOICES = {
    name: f"hf://stub/{name}.wav"
    for name in (
        "cosette", "marius", "javert", "alba", "jean", "anna", "vera",
        "fantine", "charles", "paul", "eponine", "azelma", "george", "mary",
        "jane", "michael", "eve", "bill_boerst", "peter_yearsley",
        "stuart_bell", "caro_davy", "giovanni", "lola", "juergen", "rafael",
        "daan", "estelle",
    )
}
