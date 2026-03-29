#!/usr/bin/env python3
"""
Prepare PyLaia training data from split files.

Produces for each split:
  - <split>.ids       : image stems, one per line
  - <split>.gt.txt    : space-separated character transcriptions, one per line
  - syms.txt          : character-to-index mapping (generated from train split only)

Word spaces in transcriptions are represented by the special symbol <space>.
"""

import argparse
from collections import Counter
from pathlib import Path


SPACE_SYMBOL = "<space>"


def transcription_to_chars(text):
    """Convert a transcription string to a list of PyLaia character tokens."""
    tokens = []
    for ch in text:
        if ch == " ":
            tokens.append(SPACE_SYMBOL)
        else:
            tokens.append(ch)
    return tokens


def read_split(split_file):
    """Read a split file, return list of (stem, transcription) pairs."""
    entries = []
    with open(split_file, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                print(f"WARNING: skipping malformed line {lineno}: {line!r}")
                continue
            filename, transcription = parts
            stem = Path(filename).stem
            entries.append((stem, transcription))
    return entries


def write_split(entries, outdir, split_name):
    ids_path = outdir / f"{split_name}.ids"
    gt_path = outdir / f"{split_name}.gt.txt"

    with open(ids_path, "w", encoding="utf-8") as f_ids, \
         open(gt_path, "w", encoding="utf-8") as f_gt:
        for stem, transcription in entries:
            tokens = transcription_to_chars(transcription)
            f_ids.write(stem + "\n")
            f_gt.write(" ".join(tokens) + "\n")

    print(f"{split_name:6s}: {len(entries)} lines -> {ids_path}, {gt_path}")
    return entries


def build_syms(entries):
    """Build character set from training entries, return sorted list of symbols."""
    chars = Counter()
    for _, transcription in entries:
        chars.update(transcription.replace(" ", ""))  # spaces handled separately
    symbols = sorted(chars.keys())
    return symbols


def write_syms(symbols, outdir):
    syms_path = outdir / "syms.txt"
    # Index 0 is reserved for CTC blank, index 1 for <space>, then characters
    with open(syms_path, "w", encoding="utf-8") as f:
        f.write(f"<ctc> 0\n")
        f.write(f"{SPACE_SYMBOL} 1\n")
        for i, sym in enumerate(symbols, start=2):
            f.write(f"{sym} {i}\n")
    print(f"\nsyms.txt: {len(symbols) + 2} symbols (including <ctc> and {SPACE_SYMBOL}) -> {syms_path}")
    print(f"\nCharacter inventory ({len(symbols)} characters):")
    for sym in symbols:
        print(f"  U+{ord(sym):04X}  {sym}")


def main():
    parser = argparse.ArgumentParser(description="Prepare PyLaia training data")
    parser.add_argument("splits_dir", help="Directory containing train/val/test split files")
    parser.add_argument("--outdir", required=True,
                        help="Output directory for PyLaia data files")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read all splits
    all_entries = {}
    for split_name in ("train", "val", "test"):
        split_file = splits_dir / f"{split_name}.txt"
        if not split_file.exists():
            print(f"WARNING: {split_file} not found, skipping")
            continue
        all_entries[split_name] = read_split(split_file)

    if "train" not in all_entries:
        print("ERROR: train.txt is required to build syms.txt")
        return

    # Build and write syms from training set only
    symbols = build_syms(all_entries["train"])
    write_syms(symbols, outdir)

    # Write splits
    print()
    for split_name, entries in all_entries.items():
        write_split(entries, outdir, split_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
