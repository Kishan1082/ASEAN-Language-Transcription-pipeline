# ASEAN Audio Pipeline

Local, offline (no paid APIs) pipeline that:
1. Ingests an audio file
2. Strips silence/dead air (VAD)
3. Identifies which ASEAN language is spoken
4. Transcribes the speech to text

## Architecture

```
audio file
   │
   ▼
audio_loader.py      -> mono, 16kHz, normalized float32
   │
   ▼
silence_removal.py   -> Silero VAD, trims dead air
   │
   ▼
lang_id.py            -> SpeechBrain VoxLingua107 (ECAPA-TDNN)
   │
   ▼
transcriber.py        -> faster-whisper (id/ms/th/vi/tl/en)
                          or MMS ASR (my/km/lo, low-resource)
   │
   ▼
pipeline.py            -> orchestrates the above, returns a dict
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run of each model downloads open weights from HuggingFace/torch.hub
and caches them locally (typically under `~/.cache`). After that, everything
runs fully offline. No API keys, no per-request cost.

Approximate one-time download sizes:
- Silero VAD: ~1 MB
- SpeechBrain VoxLingua107: ~20 MB
- faster-whisper `small`: ~250 MB (int8)
- MMS `mms-1b-all` (only loaded if a low-resource language is detected): ~2 GB

## Usage

```bash
python main.py sample.wav
python main.py sample.wav --json
python main.py recordings_folder/ --batch
```

## Language coverage

| Code | Language | ASR backend |
|---|---|---|
| id | Indonesian | faster-whisper |
| ms | Malay | faster-whisper |
| th | Thai | faster-whisper |
| vi | Vietnamese | faster-whisper |
| tl | Tagalog/Filipino | faster-whisper |
| en | English (SG/regional) | faster-whisper |
| my | Burmese | MMS (low-resource fallback) |
| km | Khmer | MMS (low-resource fallback) |
| lo | Lao | MMS (low-resource fallback) |

Not covered out of the box: Tetum (East Timor) — VoxLingua107 doesn't include
it. If you need it, either add a dedicated MMS adapter call for `tdt` (MMS
does cover Tetum) with manual language routing, or fine-tune the lang-ID step.

## Design notes / why these models

- **Silero VAD** over simple energy-threshold silence detection: much more
  robust to quiet speakers, background noise, and non-speech sounds, and it's
  tiny/fast enough to run per-file with negligible overhead.
- **SpeechBrain VoxLingua107** as a *dedicated* language-ID model rather than
  relying on Whisper's built-in detection: dedicated LID models are more
  reliable on short/noisy clips, and VoxLingua107's 107-language set happens
  to cover the full ASEAN list.
- **faster-whisper (CTranslate2, int8)** instead of vanilla `openai-whisper`:
  same accuracy, ~4x faster and much lower RAM on CPU — matters for a
  "lightweight, local" pipeline.
- **MMS fallback for Burmese/Khmer/Lao**: these are lower-resource languages
  where Whisper's training data is thin. Meta's MMS was trained across 1000+
  languages including these and gives noticeably better transcripts there.
  Routing is automatic based on the language-ID result.

## Extending

- Swap `WHISPER_MODEL_SIZE` in `config.py` to `medium` or `large-v3` for
  higher accuracy at the cost of speed/RAM.
- To add a new ASEAN language, add an entry to `ASEAN_LANGUAGES` in
  `config.py` with its MMS ISO-639-3 code, and decide whether it goes in
  `WHISPER_CAPABLE` or `MMS_FALLBACK`.
- `pipeline.py`'s `AseanAudioPipeline` loads all models once — reuse a single
  instance across files in batch/production use rather than re-instantiating.
