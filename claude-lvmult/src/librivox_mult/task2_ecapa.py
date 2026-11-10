"""Task 2: ECAPA embeddings for turns (worker).

`ssh worker 'cd /work && PYTHONPATH=src /opt/venv/bin/python -m librivox_mult.task2_ecapa
    --reading lv5009 [--enroll 01] [--device cpu]'`

Embeds every turn of `/out/task2/turns/<work>/<rid>.json.gz` lasting at least MIN_DUR
(the middle MAX_DUR seconds of longer turns). With `--enroll SECTION`, it also embeds that
section's wav2vec2 word runs separated by pauses of at least ENROLL_GAP (the cast list's
lines). Writes `/out/task2/ecapa/<work>/<rid>.npz` with arrays `turn_emb`, `turn_idx`
(section index, turn index) and `enroll_emb`, plus `enroll_spans` (start, end) and
`enroll_text` (JSON).
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

OUT = Path("/out")
SR = 16000
MIN_DUR = 1.0
MAX_DUR = 10.0
ENROLL_GAP = 0.35
MODEL = "/cache/hf/hub/models--speechbrain--spkrec-ecapa-voxceleb/snapshots/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"


def load_audio(path: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def clip(audio: np.ndarray, t0: float, t1: float) -> np.ndarray:
    if t1 - t0 > MAX_DUR:
        mid = (t0 + t1) / 2
        t0, t1 = mid - MAX_DUR / 2, mid + MAX_DUR / 2
    return audio[int(t0 * SR):int(t1 * SR)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reading", required=True)
    ap.add_argument("--enroll")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    torch.set_num_threads(16)
    enc = EncoderClassifier.from_hparams(source=MODEL, savedir="/tmp/ecapa", run_opts={"device": a.device})

    def embed(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(x.copy()).unsqueeze(0).to(a.device)).squeeze().cpu().numpy()
        return e / np.linalg.norm(e)

    inv = {json.loads(x)["reading_id"]: json.loads(x) for x in (OUT / "inventory.jsonl").read_text().splitlines()}
    r = inv[a.reading]
    paths = {f"{r['work']}_{r['reading_id']}_{s['section']}": s["path"] for s in r["sections"]}
    turns = json.load(gzip.open(OUT / "task2" / "turns" / r["work"] / f"{a.reading}.json.gz", "rt"))
    embs, idx = [], []
    for si, sec in enumerate(turns["sections"]):
        audio = load_audio(paths[sec["recording_id"]])
        for ti, t in enumerate(sec["turns"]):
            if t["end"] - t["start"] >= MIN_DUR:
                embs.append(embed(clip(audio, t["start"], t["end"])))
                idx.append((si, ti))
        print(sec["recording_id"], len(embs), flush=True)
    en_emb, en_spans, en_text = [], [], []
    if a.enroll:
        s = next(s for s in r["sections"] if s["section"] == a.enroll)
        audio = load_audio(s["path"])
        wd = json.load(open(OUT / "w2v" / r["work"] / r["librivox_slug"] / f"{s['stem']}.json"))["words"]
        runs = [[wd[0]]] if wd else []
        for p, w in zip(wd, wd[1:]):
            if w["start"] - p["end"] >= ENROLL_GAP:
                runs.append([])
            runs[-1].append(w)
        for run in runs:
            t0, t1 = run[0]["start"], run[-1]["end"]
            if t1 - t0 >= MIN_DUR:
                en_emb.append(embed(clip(audio, t0, t1)))
                en_spans.append((t0, t1))
                en_text.append(" ".join(w["word"] for w in run).lower())
    dst = OUT / "task2" / "ecapa" / r["work"] / f"{a.reading}.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, turn_emb=np.array(embs), turn_idx=np.array(idx), enroll_emb=np.array(en_emb),
             enroll_spans=np.array(en_spans), enroll_text=np.array([json.dumps(en_text)]))
    print(dst, len(embs), "turns", len(en_emb), "enrollment runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
