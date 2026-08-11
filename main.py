"""
CLI entry point.

Usage:
    python main.py sample.wav
    python main.py sample.wav --json
    python main.py recordings/ --batch
"""

import argparse
import glob
import json
import os

from Components.pipeline import AseanAudioPipeline

from pathlib import Path
from Components import config


SUPPORTED_EXTENSIONS = (
    "*.wav",
    "*.mp3",
    "*.m4a",
    "*.flac",
    "*.ogg",
)


def main():

    parser = argparse.ArgumentParser(
        description="ASEAN Audio Processing Pipeline"
    )

    parser.add_argument(
        "input",
        help="Audio file or folder path",
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all supported audio files in a folder",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON",
    )

    args = parser.parse_args()

    audio_path = Path(args.input)

    if not audio_path.exists():
        candidate = config.SAMPLES_DIR / args.input
        if candidate.exists():
            audio_path = candidate

    pipeline = AseanAudioPipeline()

    # -------------------------------------------------------------
    # Build file list
    # -------------------------------------------------------------

    if args.batch:

        files = []

        for ext in SUPPORTED_EXTENSIONS:
            files.extend(
                glob.glob(
                    os.path.join(args.input, ext)
                )
            )

        files = sorted(files)

        if not files:
            print("No supported audio files found.")
            return

    else:

        files = [args.input]

    # -------------------------------------------------------------
    # Process files
    # -------------------------------------------------------------

    results = []

    for file in files:

        print(f"\nProcessing: {file}")

        result = pipeline.process(file)

        results.append(result)

    # -------------------------------------------------------------
    # JSON output
    # -------------------------------------------------------------

    if args.json:

        print(
            json.dumps(
                results,
                indent=2,
                ensure_ascii=False,
            )
        )

        return

    # -------------------------------------------------------------
    # Human-readable output
    # -------------------------------------------------------------

    for r in results:

        print("\n" + "=" * 60)
        print(os.path.basename(r["file"]))
        print("=" * 60)

        if r["status"] == "error":
            print(f"Status : ERROR")
            print(f"Reason : {r['error']}")
            continue

        if r["status"] != "ok":

            print(f"Status : {r['status']}")
            print(r["vad_stats"])

            continue

        lang = r["language"]

        print(f"Audio Duration        : {r['duration_s']} s")
        print(f"Speech Duration       : {r['transcription_duration']} s")
        print(
            f"Silence Removed       : {r['vad_stats']['silence_removed_s']} s"
        )
        print(
            f"Speech Segments       : {r['vad_stats']['speech_segments']}"
        )

        print()

        print(f"Language              : {lang['lang_name']}")
        print(f"Language Code         : {lang['lang_code']}")
        print(f"Confidence            : {lang['confidence']:.3f}")
        print(f"Language Backend      : {r['language_backend']}")

        if lang["low_confidence"]:
            print("WARNING: Low confidence language detection")

        print()

        print(f"ASR Backend           : {r['asr_backend']}")
        print(f"Transcript Segments   : {len(r['segments'])}")

        print()

        print("Transcript")
        print("-" * 60)
        print(r["transcript"])
        print("-" * 60)

        print()

        print(f"Transcript File       : {r['transcript_file']}")
        print(f"Processing Time       : {r['elapsed_s']} s")

    print("\nDone.")


if __name__ == "__main__":
    main()