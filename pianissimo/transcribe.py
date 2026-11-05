import argparse
import json
import os
from pathlib import Path

import torch
from nemo.collections.asr.models import ASRModel


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe Swedish audio with Pianissimo")
    parser.add_argument("audio", nargs="+", help="One or more audio files")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "KlangAI/pianissimo-sv"))
    return parser.parse_args()


def main():
    args = parse_args()
    missing = [path for path in args.audio if not Path(path).is_file()]
    if missing:
        raise SystemExit(f"Audio file not found: {', '.join(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model)
    if model_path.is_file():
        model = ASRModel.restore_from(
            restore_path=str(model_path),
            map_location=device,
        ).eval()
    else:
        model = ASRModel.from_pretrained(
            model_name=args.model,
            map_location=device,
        ).eval()

    with torch.inference_mode():
        hypotheses = model.transcribe(
            audio=args.audio,
            batch_size=args.batch_size,
            return_hypotheses=True,
            timestamps=args.timestamps,
        )

    for audio_path, hypothesis in zip(args.audio, hypotheses):
        result = {"audio": audio_path, "text": hypothesis.text}
        if args.timestamps:
            result["timestamps"] = hypothesis.timestamp
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
