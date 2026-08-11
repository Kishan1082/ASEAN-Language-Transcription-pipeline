"""
Central configuration: sample rate, ASEAN language table, and
routing rules for which ASR backend handles which language.
"""
from pathlib import Path

# ------------------------------------------------------------------
# Project Paths
# ------------------------------------------------------------------

# Root project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input samples
SAMPLES_DIR = PROJECT_ROOT / "samples"

# Output directory
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Output subfolders
TRANSCRIPTS_DIR = OUTPUTS_DIR / "transcripts"
TRIMMED_AUDIO_DIR = OUTPUTS_DIR / "trimmed_audio"
JSON_DIR = OUTPUTS_DIR / "json"

# Pretrained model cache
PRETRAINED_DIR = PROJECT_ROOT / "pretrained_models"

#-------------------------------------------------------------------

SAMPLE_RATE = 16000  # all models expect 16 kHz mono audio

# ASEAN languages supported by the pipeline.
# 'backend' determines which ASR model will be used after language detection.
# 'mms_code' is only needed when routing to Meta MMS.
ASEAN_LANGUAGES = {
    "id": {
        "name": "Indonesian",
        "backend": "whisper",
        "mms_code": "ind",
    },
    "ms": {
        "name": "Malay",
        "backend": "whisper",
        "mms_code": "zlm",
    },
    "th": {
        "name": "Thai",
        "backend": "whisper",
        "mms_code": "tha",
    },
    "vi": {
        "name": "Vietnamese",
        "backend": "whisper",
        "mms_code": "vie",
    },
    "tl": {
        "name": "Tagalog / Filipino",
        "backend": "whisper",
        "mms_code": "tgl",
    },
    "en": {
        "name": "English (Singapore/Brunei/regional)",
        "backend": "whisper",
        "mms_code": "eng",
    },
    "my": {
        "name": "Burmese",
        "backend": "mms",
        "mms_code": "mya",
    },
    "km": {
        "name": "Khmer",
        "backend": "mms",
        "mms_code": "khm",
    },
    "lo": {
        "name": "Lao",
        "backend": "mms",
        "mms_code": "lao",
    },
}

# Whisper configuration
WHISPER_MODEL_SIZE = "small"
WHISPER_COMPUTE_TYPE = "int8"

# MMS configuration
MMS_MODEL_ID = "facebook/mms-1b-all"

# Language detection confidence threshold
LANG_ID_CONFIDENCE_THRESHOLD = 0.55