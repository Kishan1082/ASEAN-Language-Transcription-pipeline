"""
Step 1: Ingest an audio file and normalize it to the format every
downstream model expects: mono, 16kHz, float32 PCM in [-1, 1].

Supports wav/mp3/m4a/flac/ogg via soundfile + librosa fallback.
"""
import numpy as np
import soundfile as sf
import librosa

from . import config


def load_audio(path: str, target_sr: int = config.SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    Load an audio file from disk and return (samples, sample_rate).

    samples: 1-D float32 numpy array, mono, normalized to target_sr.
    """
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        # soundfile can't read some containers (e.g. certain mp3/m4a);
        # librosa (via audioread/ffmpeg) covers those.
        audio, sr = librosa.load(path, sr=None, mono=False)

    # Collapse to mono if stereo/multi-channel.
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1 if audio.shape[0] > audio.shape[1] else 0)
        audio = audio.astype(np.float32)

    # Resample if needed.
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Peak-normalize to avoid clipping/silence-detection issues on
    # very quiet recordings.
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.95

    return audio.astype(np.float32), sr


def get_duration_seconds(audio: np.ndarray, sr: int) -> float:
    return len(audio) / float(sr)

