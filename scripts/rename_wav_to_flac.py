"""
Rename files with incorrect .wav extensions to .flac when the underlying
file bytes are actually FLAC.

This script scans a directory tree for files ending with .wav (case-insensitive),
inspects the magic header to determine real format, and renames to .flac if the
content is FLAC. By default this is a dry run.

Usage examples:

  Dry run (no changes):
    python rename_wav_to_flac.py /path/to/audio

  Perform renames:
    python rename_wav_to_flac.py /path/to/audio --execute

  Overwrite existing .flac if target exists (careful):
    python rename_wav_to_flac.py /path/to/audio --execute --overwrite
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple


def detect_audio_format_by_magic(file_path: Path) -> Optional[str]:
    """
    Detect basic audio format using file magic bytes.

    Returns one of: "flac", "wav", or None if unknown.
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(12)
        if len(header) < 4:
            return None
        # FLAC magic signature
        if header[:4] == b"fLaC":
            return "flac"
        # WAV/RIFF magic: 'RIFF' .... 'WAVE'
        if header[:4] == b"RIFF" and len(header) >= 12 and header[8:12] == b"WAVE":
            return "wav"
        return None
    except Exception:
        return None


def build_target_path(original_path: Path) -> Path:
    """
    Construct the target path with .flac extension, preserving the rest of the name.
    """
    return original_path.with_suffix(".flac")


def choose_nonconflicting_path(target_path: Path) -> Path:
    """
    If target exists, produce a non-conflicting alternative by appending a counter.
    """
    if not target_path.exists():
        return target_path
    stem = target_path.stem
    parent = target_path.parent
    suffix = target_path.suffix
    counter = 1
    while True:
        candidate = parent / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def rename_file(
    source_path: Path,
    target_path: Path,
    execute: bool,
    overwrite: bool,
) -> Tuple[bool, Path]:
    """
    Rename source_path to target_path.

    Returns (changed, final_target_path).
    """
    final_target = target_path
    if target_path.exists():
        if overwrite:
            # os.replace overwrites atomically if possible
            if execute:
                os.replace(source_path, target_path)
            return True, target_path
        else:
            final_target = choose_nonconflicting_path(target_path)
            if execute:
                os.replace(source_path, final_target)
            return True, final_target
    else:
        if execute:
            source_path.rename(target_path)
        return True, target_path


def main():
    parser = argparse.ArgumentParser(description="Rename mislabelled FLAC .wav files to .flac")
    parser.add_argument("root", type=str, help="Root directory to scan recursively")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform renames. Without this flag, runs in dry-run mode",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .flac targets if present (dangerous)",
    )
    parser.add_argument(
        "--also-normalize-uppercase",
        action="store_true",
        help="Also normalize files with uppercase extensions (e.g., .WAV)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root directory does not exist or is not a directory: {root}")

    patterns = ["*.wav"]
    if args.also_normalize_uppercase:
        patterns.append("*.WAV")

    total_examined = 0
    total_flac_mislabelled = 0
    total_renamed = 0
    total_skipped = 0

    for pattern in patterns:
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            total_examined += 1
            fmt = detect_audio_format_by_magic(path)
            if fmt == "flac":
                total_flac_mislabelled += 1
                target = build_target_path(path)
                if args.execute:
                    changed, final_target = rename_file(path, target, execute=True, overwrite=args.overwrite)
                    if changed:
                        print(f"RENAMED: {path} -> {final_target}")
                        total_renamed += 1
                else:
                    print(f"DRY RUN would rename: {path} -> {target}")
            else:
                total_skipped += 1
                reason = "unknown format"
                if fmt == "wav":
                    reason = "actual WAV content"
                print(f"SKIP: {path} ({reason})")

    print("\nSummary:")
    print(f"  Examined: {total_examined}")
    print(f"  Detected FLAC under .wav: {total_flac_mislabelled}")
    print(f"  Renamed: {total_renamed}{' (dry run)' if not args.execute else ''}")
    print(f"  Skipped: {total_skipped}")


if __name__ == "__main__":
    main()





