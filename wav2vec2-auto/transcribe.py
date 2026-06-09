import argparse
import json
import os
from pathlib import Path

import torch
from transformers import pipeline


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe audio files from /input to /output with Wav2Vec2.")
    parser.add_argument("--input", default="/input", help="Input audio file or directory.")
    parser.add_argument("--output", default="/output", help="Output directory.")
    parser.add_argument("--model", default=None, help="Hugging Face model id.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Inference device.")
    parser.add_argument("--chunk-length-s", type=float, default=30.0, help="Chunk length for long files.")
    parser.add_argument("--stride-length-s", type=float, default=5.0, help="Stride length for chunked inference.")
    return parser.parse_args()


def audio_files(path):
    if path.is_file():
        return [path]
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in AUDIO_EXTENSIONS)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = args.model or os.environ.get("MODEL_ID", "facebook/wav2vec2-base-960h")
    requested_device = args.device or os.environ.get("DEVICE", "cpu")
    device = 0 if requested_device == "cuda" and torch.cuda.is_available() else -1

    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        chunk_length_s=args.chunk_length_s,
        stride_length_s=args.stride_length_s,
    )

    files = audio_files(input_path)
    if not files:
        raise SystemExit(f"No audio files found in {input_path}")

    for audio_path in files:
        result = transcriber(str(audio_path))
        text = result["text"].strip()
        stem = audio_path.stem
        (output_dir / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
        (output_dir / f"{stem}.json").write_text(
            json.dumps({"audio": str(audio_path), "model": model_id, "text": text}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {output_dir / f'{stem}.txt'}")


if __name__ == "__main__":
    main()

