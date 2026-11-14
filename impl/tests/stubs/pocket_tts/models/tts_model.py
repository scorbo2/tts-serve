"""Stub of ``pocket_tts.models.tts_model`` for test machines without the real package."""


class TTSModel:
    """Placeholder: the tests never load or run a real model."""

    sample_rate = 24000

    @classmethod
    def load_model(cls, *args, **kwargs):
        raise NotImplementedError("pocket_tts stub: no real model on test machines")

    def get_state_for_audio_prompt(self, *args, **kwargs):
        raise NotImplementedError("pocket_tts stub")

    def generate_audio(self, *args, **kwargs):
        raise NotImplementedError("pocket_tts stub")
