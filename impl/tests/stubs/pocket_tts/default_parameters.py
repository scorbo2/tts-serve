"""Stub of ``pocket_tts.default_parameters``.

Faithful copies of the real engine's values (pocket-tts 3.3.0) so the
import-time configuration validation matches the real package.
"""

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


def get_default_voice_for_language(language, config=None, checkpoint=None):
    for key, voice in DEFAULT_VOICE_FOR_LANGUAGE.items():
        if language is not None and key in language:
            return voice
    return DEFAULT_VOICE_FALLBACK
