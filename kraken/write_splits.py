#!/usr/bin/env python3
"""
Prepare Kraken training data from split files.
Writes .gt.txt files alongside images and generates Kraken-format manifests.
"""

import argparse
from pathlib import Path


def process_split(split_file, image_dir, output_dir, split_name):
    split_file = Path(split_file)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / f"{split_name}.txt"
    missing = []
    written = 0

    with open(split_file, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for lineno, line in enumerate(lines, 1):
            parts = line.split(" ", 1)
            if len(parts) != 2:
                print(f"WARNING: skipping malformed line {lineno}: {line!r}")
                continue
            filename, transcription = parts
            image_path = image_dir / filename

            if not image_path.exists():
                missing.append(filename)
                continue

            gt_path = image_dir / (Path(filename).stem + ".gt.txt")
            gt_path.write_text(transcription, encoding="utf-8")

            manifest.write(str(image_path.resolve()) + "\n")
            written += 1

    print(f"{split_name:6s}: {written} lines written to {manifest_path}")
    if missing:
        print(f"         WARNING: {len(missing)} missing images:")
        for m in missing[:10]:
            print(f"           {m}")
        if len(missing) > 10:
            print(f"           ... and {len(missing) - 10} more")

    return written


def main():
    parser = argparse.ArgumentParser(description="Prepare Kraken training data")
    parser.add_argument("image_dir", help="Directory containing images")
    parser.add_argument("splits_dir", help="Directory containing train/val/test split files")
    parser.add_argument("--outdir", default=None,
                        help="Output directory for manifests (default: splits_dir)")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    outdir = Path(args.outdir) if args.outdir else splits_dir

    total = 0
    for split_name in ("train", "val", "test"):
        split_file = splits_dir / f"{split_name}.txt"
        if not split_file.exists():
            print(f"WARNING: {split_file} not found, skipping")
            continue
        total += process_split(split_file, args.image_dir, outdir, split_name)

    print(f"\nTotal: {total} lines processed")
    print(f"Manifests written to {outdir}")


if __name__ == "__main__":
    main()
