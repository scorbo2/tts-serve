"""Stub of ``pocket_tts.default_parameters``.

Faithful copies of the real engine's values (pocket-tts 3.3.0) so the
import-time configuration validation and the default-voice resolution match
the real package -- if the vendor changes these, the stub must be
regenerated to match (same discipline as the ming_tiny task_protocol stub).
"""

from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

DEFAULT_LANGUAGE = "english"

DEFAULT_VOICE_FOR_LANGUAGE = {
    "italian": "giovanni",
    "spanish": "lola",
    "german": "juergen",
    "portuguese": "rafael",
    "french": "estelle",
    "dutch": "daan",
}
DEFAULT_VOICE_FALLBACK = "alba"
# Mirrors the engine: predefined voices are precomputed states tied to the
# released weights, so custom configs/checkpoints fall back to the audio
# file behind the fallback voice -- any model can clone it.
DEFAULT_VOICE_FOR_CUSTOM_MODEL = _ORIGINS_OF_PREDEFINED_VOICES[DEFAULT_VOICE_FALLBACK]


def get_default_voice_for_language(language, config=None, checkpoint=None):
    if config is not None or checkpoint is not None:
        return DEFAULT_VOICE_FOR_CUSTOM_MODEL
    for key, voice in DEFAULT_VOICE_FOR_LANGUAGE.items():
        if language is not None and key in language:
            return voice
    return DEFAULT_VOICE_FALLBACK
