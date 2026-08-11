"""
End-to-end orchestrator.

Pipeline:
    1. Load audio
    2. Remove silence
    3. Detect language
    4. Transcribe speech

All models are loaded once when the pipeline is created and reused
across multiple files.
"""

import time

from . import audio_loader
from . import config
from faster_whisper import WhisperModel
from .silence_removal import SilenceRemover, save_audio
from .lang_id import LanguageIdentifier
from .transcriber import Transcriber, save_transcript


class AseanAudioPipeline:
    def __init__(self):
        print("======================================")
        print("Initializing ASEAN Audio Pipeline...")
        print("Loading models (first run downloads weights)...")
        print("======================================")

        print("Loading Whisper model...")
        self.whisper = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )

        print("Loading Silero VAD...")
        self.vad = SilenceRemover()

        print("Initializing language detector...")
        self.lang_id = LanguageIdentifier(self.whisper)

        print("Initializing transcriber...")
        self.transcriber = Transcriber(self.whisper)

        print("Pipeline ready.\n")

    def process(self, audio_path: str) -> dict:
        """
        Process one audio file.

        Returns a dictionary containing:
            - audio information
            - VAD statistics
            - language identification
            - transcription
            - timing information
        """
        print(f"\nProcessing: {audio_path}")
        start_time = time.time()

        try:

            # ---------------------------------------------------------
            # Step 1: Load audio
            # ---------------------------------------------------------
            print("Loading audio...")
            audio, sr = audio_loader.load_audio(audio_path)

            duration = audio_loader.get_duration_seconds(audio, sr)

            # ---------------------------------------------------------
            # Step 2: Remove silence
            # ---------------------------------------------------------
            print("Removing silence...")
            trimmed_audio, vad_stats = self.vad.remove_silence(audio, sr)

            if vad_stats.get("warning"):

                return {
                    "file": audio_path,
                    "status": "no_speech_detected",
                    "duration_s": round(duration, 2),
                    "vad_stats": vad_stats,
                    "elapsed_s": round(time.time() - start_time, 2),
                }

            trimmed_file = save_audio(
                trimmed_audio,
                audio_path,
                sr,
            )

            # ---------------------------------------------------------
            # Step 3: Language Identification
            # ---------------------------------------------------------
            print("Detecting language...")
            language = self.lang_id.identify(trimmed_audio, sr)

            # ---------------------------------------------------------
            # Step 4: Speech Transcription
            # ---------------------------------------------------------
            print("Transcribing...")
            transcription = self.transcriber.transcribe(
                trimmed_audio,
                language["lang_code"],
            )

            # ---------------------------------------------------------
            # Save transcript
            # ---------------------------------------------------------
            print("Saving transcript...")
            transcript_file = save_transcript(
                transcription["text"],
                audio_path,
            )

            # ---------------------------------------------------------
            # Final Result
            # ---------------------------------------------------------

            result = {
                "file": audio_path,
                "status": "ok",

                # Audio
                "duration_s": round(duration, 2),

                # VAD
                "vad_stats": vad_stats,
                "trimmed_audio_file": trimmed_file,

                # Language Identification
                "language": language,
                "language_code": language["lang_code"],
                "language_name": language["lang_name"],
                "language_confidence": language["confidence"],
                "language_backend": language["backend"],

                # Transcription
                "transcript": transcription["text"],
                "transcript_file": transcript_file,
                "transcription_duration": transcription["duration"],
                "asr_backend": transcription["backend"],
                "segments": transcription["segments"],

                # Timing
                "elapsed_s": round(time.time() - start_time, 2),
            }

            return result
        except Exception as e:
            return {
                "file": audio_path,
                "status": "error",
                "error": str(e),
                "elapsed_s": round(time.time() - start_time, 2),
            }