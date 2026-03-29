#!/usr/bin/env python3
"""
Prepare PyLaia training data from split files.

Produces for each split:
  - <split>_ids.txt      : image stems, one per line
  - <split>.txt          : space-separated character transcriptions, one per line
  - <split>_text.txt     : plain text transcriptions for evaluation, one per line
  - syms.txt             : character-to-index mapping (generated from train split only)

Images are copied into images/train/, images/val/, images/test/ subdirectories.

Word spaces in transcriptions are represented by the special symbol <space>.
"""

import argparse
import shutil
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
    """Read a split file, return list of (stem, filename, transcription) tuples."""
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
            entries.append((stem, filename, transcription))
    return entries


def write_split(entries, image_dir, outdir, split_name, copy_images):
    ids_path = outdir / f"{split_name}_ids.txt"
    tok_path = outdir / f"{split_name}.txt"
    text_path = outdir / f"{split_name}_text.txt"
    img_outdir = outdir / "images" / split_name

    if copy_images:
        img_outdir.mkdir(parents=True, exist_ok=True)

    missing = []

    with open(ids_path, "w", encoding="utf-8") as f_ids, \
         open(tok_path, "w", encoding="utf-8") as f_tok, \
         open(text_path, "w", encoding="utf-8") as f_text:
        for stem, filename, transcription in entries:
            src = image_dir / filename
            if not src.exists():
                missing.append(filename)
                continue

            if copy_images:
                shutil.copy2(src, img_outdir / filename)

            tokens = transcription_to_chars(transcription)
            f_ids.write(stem + "\n")
            f_tok.write(" ".join(tokens) + "\n")
            f_text.write(transcription + "\n")

    print(f"{split_name:6s}: {len(entries) - len(missing)} lines")
    print(f"         ids     -> {ids_path}")
    print(f"         tokens  -> {tok_path}")
    print(f"         text    -> {text_path}")
    if copy_images:
        print(f"         images -> {img_outdir}")
    if missing:
        print(f"         WARNING: {len(missing)} missing images:")
        for m in missing[:10]:
            print(f"           {m}")
        if len(missing) > 10:
            print(f"           ... and {len(missing) - 10} more")


def build_syms(entries):
    """Build character set from training entries, return sorted list of symbols."""
    chars = Counter()
    for _, _, transcription in entries:
        chars.update(transcription.replace(" ", ""))
    return sorted(chars.keys())


def write_syms(symbols, outdir):
    syms_path = outdir / "syms.txt"
    with open(syms_path, "w", encoding="utf-8") as f:
        f.write("<ctc> 0\n")
        f.write(f"{SPACE_SYMBOL} 1\n")
        for i, sym in enumerate(symbols, start=2):
            f.write(f"{sym} {i}\n")
    print(f"syms.txt: {len(symbols) + 2} symbols -> {syms_path}")
    print(f"\nCharacter inventory ({len(symbols)} characters):")
    for sym in symbols:
        print(f"  U+{ord(sym):04X}  {sym}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Prepare PyLaia training data")
    parser.add_argument("image_dir", help="Directory containing images")
    parser.add_argument("splits_dir", help="Directory containing train/val/test split files")
    parser.add_argument("--outdir", required=True,
                        help="Output directory for PyLaia data files")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images into images/train|val|test/ subdirectories")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    splits_dir = Path(args.splits_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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

    symbols = build_syms(all_entries["train"])
    write_syms(symbols, outdir)

    for split_name, entries in all_entries.items():
        write_split(entries, image_dir, outdir, split_name, args.copy_images)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
