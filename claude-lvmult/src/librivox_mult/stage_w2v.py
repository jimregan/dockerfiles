"""wav2vec2 ASR (regenerates the lost wav2vec2 outputs). Runs on the worker.

    ssh worker 'cd /work && PYTHONPATH=src /opt/venv/bin/python -m librivox_mult.stage_w2v \
        [--work W] [--reading SLUG] [--limit N] [--force]'

Model: facebook/wav2vec2-large-960h-lv60-self, pinned snapshot (docs/decisions.md).
Greedy CTC over the whole file in overlapping windows; only the centre of each
window is kept, so every output frame has at least ``--context`` seconds of
audio on each side (except at file edges).

Per mp3 ``<stem>`` writes under ``<out>/w2v/<work>/<slug>/``:

  <stem>.json        model/revision/params, ``words`` [{word,start,end,score}],
                     ``tokens`` [[char, start_frame, end_frame, prob], ...]
                     (char tokens incl. "|"), times in seconds on the
                     recording's timeline
  <stem>.logp.npy    float16 [T, V] CTC log-posteriors (V = 32), frame shift 20 ms,
                     for later forced alignment / candidate scoring

The output is a record of what was said: nothing is filtered or corrected.
Files are written atomically; existing outputs with the same model revision and
parameters are skipped unless --force.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

MODEL_ID = "facebook/wav2vec2-large-960h-lv60-self"
MODEL_REVISION = "54074b1c16f4de6a5ad59affb4caa8f2ea03a119"
SR = 16000
HOP = 320  # samples per output frame (conv feature encoder stride)
FRAME_SHIFT = HOP / SR  # 0.02 s
RECEPTIVE = 400  # samples in the first output frame's receptive field

log = logging.getLogger("w2v")


# --------------------------------------------------------------------------- audio
def load_audio(path: str | Path) -> np.ndarray:
    """Decode with ffmpeg (same decoder as Stage 0) to 16 kHz mono float32."""
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    if p.stderr.strip():
        log.warning("decode %s: %s", path, p.stderr.decode(errors="replace").strip()[:200])
    return np.frombuffer(p.stdout, dtype=np.float32)


# --------------------------------------------------------------------------- windows
def n_frames(n_samples: int) -> int:
    """Output frames wav2vec2's conv encoder produces for ``n_samples``."""
    return 0 if n_samples < RECEPTIVE else (n_samples - RECEPTIVE) // HOP + 1


def plan_windows(n_samples: int, chunk_s: float, context_s: float) -> list[dict]:
    """Split a file into windows ``[core ± context]``; all boundaries on the hop grid.

    Returns dicts with sample ranges and which frames of the window's output
    to keep (``keep_from`` .. ``keep_to``) and where they go globally (``dst``).
    """
    chunk = int(round(chunk_s * SR / HOP)) * HOP
    ctx = int(round(context_s * SR / HOP)) * HOP
    total = n_frames(n_samples)
    wins = []
    core_start = 0
    while core_start * HOP < n_samples and core_start < total:
        s0 = core_start * HOP
        w0 = max(0, s0 - ctx)
        w1 = min(n_samples, s0 + chunk + ctx)
        core_end = min(core_start + chunk // HOP, total)
        keep_from = (s0 - w0) // HOP
        avail = n_frames(w1 - w0)
        keep_to = min(keep_from + (core_end - core_start), avail)
        wins.append({"w0": w0, "w1": w1, "keep_from": keep_from, "keep_to": keep_to,
                     "dst": core_start})
        core_start = core_end
    return wins


# --------------------------------------------------------------------------- decoding
def greedy_ctc(logp: np.ndarray, id2tok: dict[int, str], blank: int = 0,
               delim: str = "|") -> tuple[list[list], list[dict]]:
    """Greedy CTC: argmax, collapse repeats, drop blanks.

    Returns (tokens, words). ``tokens`` = [[tok, start_frame, end_frame, prob]]
    where the frame span is the run of that symbol and prob is the max posterior
    in the run. ``words`` = [{word, start, end, score}] split on ``delim``;
    score is the mean token prob (a record of what was said, no filtering).
    """
    if logp.shape[0] == 0:
        return [], []
    ids = logp.argmax(axis=1)
    probs = np.exp(logp[np.arange(len(ids)), ids].astype(np.float32))
    tokens: list[list] = []
    prev = -1
    for t, i in enumerate(ids):
        i = int(i)
        if i == prev and i != blank:
            tokens[-1][2] = t
            tokens[-1][3] = max(tokens[-1][3], float(probs[t]))
        elif i != blank:
            tokens.append([id2tok[i], t, t, float(probs[t])])
        prev = i
    words: list[dict] = []
    cur: list[list] = []

    def flush():
        if cur:
            words.append({
                "word": "".join(c[0] for c in cur),
                "start": round(cur[0][1] * FRAME_SHIFT, 3),
                "end": round((cur[-1][2] + 1) * FRAME_SHIFT, 3),
                "score": round(float(np.mean([c[3] for c in cur])), 4),
            })
            cur.clear()

    for tok in tokens:
        if tok[0] == delim:
            flush()
        else:
            cur.append(tok)
    flush()
    for tok in tokens:
        tok[3] = round(tok[3], 4)
    return tokens, words


# --------------------------------------------------------------------------- model
class Recogniser:
    def __init__(self, device: str, batch_size: int):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.proc = Wav2Vec2Processor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model.eval().to(device)
        if device.startswith("cuda"):
            self.model.half()
        vocab = self.proc.tokenizer.get_vocab()
        self.id2tok = {v: k for k, v in vocab.items()}
        self.blank = self.proc.tokenizer.pad_token_id
        self.vocab = [self.id2tok[i] for i in range(len(self.id2tok))]

    def logprobs(self, audio: np.ndarray, chunk_s: float, context_s: float) -> np.ndarray:
        torch = self.torch
        wins = plan_windows(len(audio), chunk_s, context_s)
        out = np.zeros((n_frames(len(audio)), len(self.vocab)), dtype=np.float16)
        for b in range(0, len(wins), self.batch_size):
            batch = wins[b : b + self.batch_size]
            segs = [audio[w["w0"] : w["w1"]] for w in batch]
            # per-window zero-mean/unit-variance (the model's do_normalize), padded
            feats = self.proc(segs, sampling_rate=SR, return_tensors="pt", padding=True)
            x = feats.input_values.to(self.device)
            mask = feats.attention_mask.to(self.device)
            if self.device.startswith("cuda"):
                x = x.half()
            with torch.inference_mode():
                lp = torch.log_softmax(self.model(x, attention_mask=mask).logits.float(), -1)
            lp = lp.cpu().numpy()
            for w, l in zip(batch, lp):
                k = w["keep_to"] - w["keep_from"]
                out[w["dst"] : w["dst"] + k] = l[w["keep_from"] : w["keep_to"]]
        return out


# --------------------------------------------------------------------------- driver
def params(a) -> dict:
    return {"model": MODEL_ID, "revision": MODEL_REVISION, "decoder": "greedy-ctc",
            "chunk_s": a.chunk, "context_s": a.context, "sampling_rate": SR,
            "frame_shift": FRAME_SHIFT}


def logp_path(json_path: Path) -> Path:
    return json_path.with_name(json_path.name[: -len(".json")] + ".logp.npy")


def is_done(json_path: Path) -> bool:
    """Resumable: a file is done when both outputs exist.

    Both are written atomically (tmp + rename), logp first, then json, so an
    interrupted file has no json and is redone. Use --force to recompute
    (e.g. after changing model or windowing parameters).
    """
    return json_path.exists() and logp_path(json_path).exists()


def atomic_write_bytes(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        write(f)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", type=Path, default=Path("/out/inventory.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("/out"))
    ap.add_argument("--work", action="append")
    ap.add_argument("--reading", action="append", help="LibriVox slug (repeatable)")
    ap.add_argument("--file", action="append", help="mp3 stem (repeatable)")
    ap.add_argument("--limit", type=int, help="process at most N files (pilot)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--decoders", type=int, default=4, help="parallel ffmpeg decode threads")
    ap.add_argument("--chunk", type=float, default=30.0, help="window core, seconds")
    ap.add_argument("--context", type=float, default=5.0, help="context each side, seconds")
    a = ap.parse_args(argv)

    (a.out / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(a.out / "logs" / "w2v.log"), logging.StreamHandler(sys.stderr)],
    )
    prm = params(a)
    jobs = []
    seen = set()
    for line in a.inventory.read_text().splitlines():
        r = json.loads(line)
        if a.work and r["work"] not in a.work:
            continue
        if a.reading and r["librivox_slug"] not in a.reading:
            continue
        for s in r["sections"]:
            if a.file and s["stem"] not in a.file:
                continue
            if s["path"] in seen:  # multi-version rows share a directory, not files
                continue
            seen.add(s["path"])
            dst = a.out / "w2v" / r["work"] / r["librivox_slug"] / f"{s['stem']}.json"
            jobs.append((s, dst))
    todo = [(s, d) for s, d in jobs if a.force or not is_done(d)]
    if a.limit:
        todo = todo[: a.limit]
    log.info("%d files selected, %d already done, %d to do",
             len(jobs), len(jobs) - len([1 for s, d in jobs if a.force or not is_done(d)]), len(todo))
    if not todo:
        return 0

    import torch

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        log.warning("running on CPU")
    rec = Recogniser(device, a.batch_size)
    t_audio = 0.0
    # ffmpeg decoding/resampling is ~60% of per-file time on the GPU, so decode
    # ahead in threads (bounded look-ahead keeps memory to a few files of audio).
    pool = ThreadPoolExecutor(max_workers=a.decoders)
    pending: dict[int, object] = {}

    def prefetch(i: int) -> None:
        if i < len(todo) and i not in pending:
            pending[i] = pool.submit(load_audio, todo[i][0]["path"])

    for i in range(a.decoders + 1):
        prefetch(i)
    t_start = time.time()
    for n, (s, dst) in enumerate(todo, 1):
        t0 = time.time()
        fut = pending.pop(n - 1)
        prefetch(n + a.decoders)
        try:
            audio = fut.result()
            logp = rec.logprobs(audio, a.chunk, a.context)
            tokens, words = greedy_ctc(logp, rec.id2tok, rec.blank)
        except Exception as e:  # noqa: BLE001 - log and continue with the next file
            log.error("FAILED %s: %s", s["path"], e)
            continue
        dur = len(audio) / SR
        dst.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(logp_path(dst), lambda f: np.save(f, logp))
        rec_out = {**prm, "vocab": rec.vocab, "blank": rec.blank, "audio": s["path"],
                   "duration_16k": round(dur, 6), "n_frames": int(logp.shape[0]),
                   "device": device, "torch": torch.__version__,
                   "words": words, "tokens": tokens}
        atomic_write_bytes(dst, lambda f: f.write(json.dumps(rec_out, ensure_ascii=False).encode()))
        el = time.time() - t0
        t_audio += dur
        log.info("[%d/%d] %s %.0fs audio in %.1fs (%.0fx) %d words",
                 n, len(todo), s["stem"], dur, el, dur / max(el, 1e-9), len(words))
    pool.shutdown()
    wall = time.time() - t_start  # true elapsed; per-file times exclude overlapped decoding
    log.info("done: %.2f h audio in %.2f h wall (%.0fx realtime)", t_audio / 3600, wall / 3600,
             t_audio / max(wall, 1e-9))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.exit(main())
