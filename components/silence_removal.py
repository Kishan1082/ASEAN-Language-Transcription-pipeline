"""
Step 2: Strip dead air / blank silence using Silero VAD.

Silero VAD is a ~1MB torch model that classifies short audio chunks
as speech/non-speech far more reliably than energy-threshold silence
detection (which fails on quiet speakers, background hum, etc.).
It runs fully offline/local once the weights are cached (first call
downloads them from torch.hub; no API key, no per-call cost).
"""
import numpy as np
import torch
from . import config
import soundfile as sf
from pathlib import Path


class SilenceRemover:
    def __init__(self):
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils

    def get_speech_segments(self, audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> list[dict]:
        """
        Returns a list of {'start': sample_idx, 'end': sample_idx}
        marking where speech occurs in `audio`.
        """
        tensor = torch.from_numpy(audio).float().cpu()
        timestamps = self.get_speech_timestamps(
            tensor,
            self.model,
            sampling_rate=sr,
            threshold=0.5,          # speech probability cutoff
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
        )
        return timestamps

    def remove_silence(self, audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> tuple[np.ndarray, dict]:
        """
        Returns (trimmed_audio, stats) where trimmed_audio is the
        concatenation of all detected speech segments and stats
        reports how much was cut.
        """
        segments = self.get_speech_segments(audio, sr)

        if not segments:
            # No speech detected at all — return the original audio
            # so downstream steps can still report "no speech found"
            # rather than crashing on an empty array.
            return audio, {
                "original_duration_s": len(audio) / sr,
                "trimmed_duration_s": len(audio) / sr,
                "silence_removed_s": 0.0,
                "speech_segments": 0,
                "warning": "No speech detected by VAD",
            }

        chunks = [audio[seg["start"]:seg["end"]] for seg in segments]

        if len(chunks) == 1:
            trimmed = chunks[0]
        else:
            trimmed = np.concatenate(chunks)

        original_dur = len(audio) / sr
        trimmed_dur = len(trimmed) / sr

        stats = {
            "original_duration_s": round(original_dur, 2),
            "trimmed_duration_s": round(trimmed_dur, 2),
            "silence_removed_s": round(original_dur - trimmed_dur, 2),
            "speech_segments": len(segments),
        }
        return trimmed, stats

def save_audio(
    audio: np.ndarray,
    audio_path: str,
    sr: int = config.SAMPLE_RATE,
) -> str:
    """
    Save trimmed audio to outputs/trimmed_audio.

    Example:
        sample.wav
            ->
        outputs/trimmed_audio/sample_trimmed.wav
    """

    config.TRIMMED_AUDIO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio_path = Path(audio_path)

    output_path = (
        config.TRIMMED_AUDIO_DIR
        / f"{audio_path.stem}_trimmed.wav"
    )

    sf.write(output_path, audio, sr)

    return str(output_path)

