"""
Step 3: Detect the spoken language.

Uses faster-whisper's built-in language detection. The Whisper model
is shared with the transcription stage, so it is loaded only once.

Runs entirely offline after the initial model download.
"""

import numpy as np
from . import config


class LanguageIdentifier:
    def __init__(self, whisper_model):
        self.model = whisper_model

    def identify(
        self,
        audio: np.ndarray,
        sr: int = config.SAMPLE_RATE,
    ) -> dict:
        """
        Returns:
        {
            "lang_code": "th",
            "lang_name": "Thai",
            "confidence": 0.97,
            "is_asean": True,
            "backend": "whisper",
            "low_confidence": False,
        }
        """

        segments, info = self.model.transcribe(
            audio,
            beam_size=1,
            language=None,          # auto detect
            task="transcribe",
            vad_filter=False,        # already removed silence
        )

        # Force execution so detection actually runs
        list(segments)

        lang_code = info.language
        confidence = float(info.language_probability)

        if lang_code in config.ASEAN_LANGUAGES:
            lang_name = config.ASEAN_LANGUAGES[lang_code]["name"]
            backend = config.ASEAN_LANGUAGES[lang_code]["backend"]
            is_asean = True
        else:
            lang_name = lang_code
            backend = "unsupported"
            is_asean = False

        return {
            "lang_code": lang_code,
            "lang_name": lang_name,
            "confidence": round(confidence, 3),
            "backend": backend,
            "is_asean": is_asean,
            "low_confidence": (
                confidence < config.LANG_ID_CONFIDENCE_THRESHOLD
            ),
        }


