import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import tempfile
import wave
from pathlib import Path

_KNOWN_OUTPUT_DIRECTORIES = set()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe a directory tree with one persistent NeMo process per GPU"
    )
    parser.add_argument("input", type=Path, help="Input directory containing WAV files")
    parser.add_argument("output", type=Path, help="Output directory for JSON files")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "/models/pianissimo-sv.nemo"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpus", default="all", help="Comma-separated visible GPU indices, or 'all'")
    parser.add_argument("--timestamps", choices=("none", "word", "segment", "word-segment", "all"), default="word-segment")
    parser.add_argument("--bucket-seconds", type=float, default=5.0)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def audio_duration(path):
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def iter_wavs(root):
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".wav"):
                    yield Path(entry.path)


def output_path(input_path, input_root, output_root):
    return (output_root / input_path.relative_to(input_root)).with_suffix(".json")


def write_json(path, value):
    directory = os.fspath(path.parent)
    if directory not in _KNOWN_OUTPUT_DIRECTORIES:
        path.parent.mkdir(parents=True, exist_ok=True)
        _KNOWN_OUTPUT_DIRECTORIES.add(directory)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except FileNotFoundError:
            pass
        raise


def select_timestamps(timestamp, mode):
    if mode == "all":
        return timestamp
    keys = []
    if mode in ("word", "word-segment"):
        keys.append("word")
    if mode in ("segment", "word-segment"):
        keys.append("segment")
    return {key: timestamp.get(key, []) for key in keys}


def run_worker(worker_id, gpu_id, tasks, config):
    import torch
    from nemo.collections.asr.models import ASRModel

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    model = ASRModel.restore_from(config["model"], map_location=device).eval()
    want_timestamps = config["timestamps"] != "none"
    processed = 0
    failed = 0
    last_progress = 0

    def transcribe(items):
        nonlocal last_progress, processed
        paths = [str(item[0]) for item in items]
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            hypotheses = model.transcribe(
                audio=paths,
                batch_size=len(paths),
                return_hypotheses=True,
                timestamps=want_timestamps,
            )
        for item, hypothesis in zip(items, hypotheses):
            source, destination = item
            value = {
                "audio": str(source),
                "text": hypothesis.text,
                "model": os.environ.get("MODEL_ID", "KlangAI/pianissimo-sv"),
                "model_revision": os.environ.get("MODEL_REVISION"),
            }
            if want_timestamps:
                value["timestamps"] = select_timestamps(hypothesis.timestamp, config["timestamps"])
            write_json(destination, value)
            destination.with_suffix(".error.json").unlink(missing_ok=True)
            processed += 1
        progress = config["progress_every"]
        if progress and processed // progress > last_progress:
            last_progress = processed // progress
            print(
                f"worker {worker_id} GPU {gpu_id}: {processed} files complete",
                file=sys.stderr,
                flush=True,
            )

    while True:
        batch = tasks.get()
        if batch is None:
            break
        try:
            transcribe(batch)
        except Exception as batch_error:
            print(
                f"worker {worker_id} GPU {gpu_id}: batch failed, retrying individually: {batch_error}",
                file=sys.stderr,
                flush=True,
            )
            torch.cuda.empty_cache()
            for item in batch:
                try:
                    transcribe([item])
                except Exception as error:
                    failed += 1
                    error_path = item[1].with_suffix(".error.json")
                    write_json(error_path, {"audio": str(item[0]), "error": repr(error)})
    return processed, failed


def worker(worker_id, gpu_id, tasks, results, config):
    try:
        processed, failed = run_worker(worker_id, gpu_id, tasks, config)
        results.put({"worker": worker_id, "processed": processed, "failed": failed, "fatal": None})
    except BaseException as error:
        results.put({"worker": worker_id, "processed": 0, "failed": 0, "fatal": repr(error)})
        raise


def put_task(tasks, value, workers):
    while True:
        try:
            tasks.put(value, timeout=5)
            return
        except queue.Full:
            if not any(process.is_alive() for process in workers):
                raise RuntimeError("All GPU workers exited while work was still being queued")


def enqueue_batch(tasks, batch, workers):
    if batch:
        put_task(tasks, batch, workers)
        return 1
    return 0


def main():
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Input directory not found: {input_root}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.bucket_seconds <= 0:
        raise SystemExit("--bucket-seconds must be greater than zero")
    if args.progress_every < 0:
        raise SystemExit("--progress-every cannot be negative")

    import torch

    available = torch.cuda.device_count()
    if available == 0:
        raise SystemExit("No CUDA GPUs are visible")
    gpu_ids = list(range(available)) if args.gpus == "all" else [int(value) for value in args.gpus.split(",")]
    if not gpu_ids or any(gpu < 0 or gpu >= available for gpu in gpu_ids):
        raise SystemExit(f"Invalid --gpus value; visible GPU indices are 0 through {available - 1}")

    context = mp.get_context("spawn")
    tasks = context.Queue(maxsize=len(gpu_ids) * 4)
    results = context.Queue()
    config = {
        "model": args.model,
        "timestamps": args.timestamps,
        "progress_every": args.progress_every,
    }
    workers = [
        context.Process(target=worker, args=(index, gpu, tasks, results, config))
        for index, gpu in enumerate(gpu_ids)
    ]
    for process in workers:
        process.start()

    buckets = {}
    queued_files = 0
    queued_batches = 0
    skipped = 0
    unreadable = 0
    try:
        for source in iter_wavs(input_root):
            destination = output_path(source, input_root, output_root)
            if destination.exists() and not args.overwrite:
                skipped += 1
                continue
            try:
                duration = audio_duration(source)
            except (OSError, EOFError, wave.Error) as error:
                unreadable += 1
                write_json(destination.with_suffix(".error.json"), {"audio": str(source), "error": repr(error)})
                continue
            bucket = math.ceil(duration / args.bucket_seconds)
            items = buckets.setdefault(bucket, [])
            items.append((source, destination))
            if len(items) >= args.batch_size:
                queued_batches += enqueue_batch(tasks, items, workers)
                queued_files += len(items)
                buckets[bucket] = []

        for batch in buckets.values():
            queued_batches += enqueue_batch(tasks, batch, workers)
            queued_files += len(batch)
    finally:
        for _ in [process for process in workers if process.is_alive()]:
            put_task(tasks, None, workers)

    for process in workers:
        process.join()
    summaries = []
    while True:
        try:
            summaries.append(results.get_nowait())
        except queue.Empty:
            break
    bad_workers = [process.exitcode for process in workers if process.exitcode != 0]
    fatal_workers = [summary for summary in summaries if summary["fatal"]]
    processed = sum(summary["processed"] for summary in summaries)
    failed = sum(summary["failed"] for summary in summaries)
    print(
        json.dumps(
            {
                "queued_files": queued_files,
                "queued_batches": queued_batches,
                "processed": processed,
                "failed": failed,
                "skipped": skipped,
                "unreadable": unreadable,
                "gpus": gpu_ids,
                "worker_exit_codes": bad_workers,
                "fatal_workers": fatal_workers,
            },
            separators=(",", ":"),
        )
    )
    if failed or unreadable or bad_workers or fatal_workers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
