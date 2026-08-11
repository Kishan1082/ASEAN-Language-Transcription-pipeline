"""
Step 4: Transcribe speech to text.

Routes by detected language:
  - High-resource ASEAN languages -> faster-whisper
  - Low-resource languages -> Meta MMS ASR

Models are loaded lazily so only the required backend is initialized.
Everything runs locally after the first model download.
"""


import numpy as np
import torch
from . import config


class Transcriber:
    def __init__(self, whisper_model):
        self.whisper_model = whisper_model
        self._mms_processor = None
        self._mms_model = None

    # ------------------------------------------------------------------
    # Lazy model loaders
    # ------------------------------------------------------------------


    def _load_mms(self):
        if self._mms_model is None:
            from transformers import AutoProcessor, Wav2Vec2ForCTC

            print("Loading MMS model...")

            self._mms_processor = AutoProcessor.from_pretrained(
                config.MMS_MODEL_ID
            )

            self._mms_model = Wav2Vec2ForCTC.from_pretrained(
                config.MMS_MODEL_ID
            )

        return self._mms_processor, self._mms_model

    # ------------------------------------------------------------------
    # Whisper backend
    # ------------------------------------------------------------------

    def _transcribe_whisper(
        self,
        audio: np.ndarray,
        lang_code: str,
    ) -> dict:

        model = self.whisper_model

        segments, info = model.transcribe(
            audio,
            language=lang_code,
            beam_size=5,
            vad_filter=False,      # already removed silence
        )

        segments = list(segments)

        text = " ".join(
            segment.text.strip()
            for segment in segments
        )

        duration = round(len(audio) / config.SAMPLE_RATE, 2)

        return {
            "text": text.strip(),
            "language": lang_code,
            "backend": "faster-whisper",
            "duration": duration,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text.strip(),
                }
                for s in segments
            ],
        }

    # ------------------------------------------------------------------
    # MMS backend
    # ------------------------------------------------------------------

    def _transcribe_mms(
        self,
        audio: np.ndarray,
        lang_code: str,
    ) -> dict:

        processor, model = self._load_mms()

        mms_code = config.ASEAN_LANGUAGES[lang_code]["mms_code"]

        processor.tokenizer.set_target_lang(mms_code)
        model.load_adapter(mms_code)

        inputs = processor(
            audio,
            sampling_rate=config.SAMPLE_RATE,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(**inputs).logits

        ids = torch.argmax(logits, dim=-1)[0]

        text = processor.decode(ids)

        duration = round(len(audio) / config.SAMPLE_RATE, 2)

        return {
            "text": text.strip(),
            "language": lang_code,
            "backend": "mms-1b-all",
            "duration": duration,
            "segments": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        lang_code: str,
    ) -> dict:

        if lang_code in config.ASEAN_LANGUAGES:

            backend = config.ASEAN_LANGUAGES[lang_code]["backend"]

            if backend == "mms":
                return self._transcribe_mms(audio, lang_code)

            return self._transcribe_whisper(audio, lang_code)

        # Unknown language -> Whisper auto detection

        model = self.whisper_model

        segments, info = model.transcribe(
            audio,
            language=None,
            beam_size=5,
            vad_filter=False,
        )

        segments = list(segments)

        text = " ".join(
            segment.text.strip()
            for segment in segments
        )

        duration = round(len(audio) / config.SAMPLE_RATE, 2)

        return {
            "text": text.strip(),
            "language": info.language,
            "backend": "faster-whisper (auto)",
            "duration": duration,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text.strip(),
                }
                for s in segments
            ],
        }

def save_transcript(transcript: str, audio_path: str) -> str:
    """
    Save transcript as a .txt file with the same base name
    as the input audio file.

    Example:
        sample2.wav -> sample2_transcript.txt
    """
    from pathlib import Path

    audio_path = Path(audio_path)

    config.TRANSCRIPTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
    )
    output_path = (
        config.TRANSCRIPTS_DIR /
        f"{audio_path.stem}_transcript.txt"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    return str(output_path)

