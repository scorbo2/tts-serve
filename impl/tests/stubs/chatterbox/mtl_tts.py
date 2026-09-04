"""Stub of ``chatterbox.mtl_tts`` for test machines without the real package.

The constants below are faithful copies of the real engine's values (checked
against private/chatterbox) so that the import-time configuration validation
and the dynamically-built language ``Literal`` behave exactly as they would
with the real package.
"""

S3_SR = 16000
S3GEN_SR = 24000

MULTILINGUAL_T3_MODELS = {
    "v2": "t3_mtl23ls_v2.safetensors",
    "t3_mtl23ls_v2": "t3_mtl23ls_v2.safetensors",
    "v3": "t3_mtl23ls_v3.safetensors",
    "t3_mtl23ls_v3": "t3_mtl23ls_v3.safetensors",
}

# Supported languages for the multilingual model (code -> English name).
SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}


class ChatterboxMultilingualTTS:
    """Placeholder — real model loading is never exercised in the tests."""

    sr = S3GEN_SR

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError(
            "chatterbox stub: from_pretrained() is not available in tests"
        )

    def generate(self, *args, **kwargs):
        raise NotImplementedError("chatterbox stub: generate() is not available in tests")
