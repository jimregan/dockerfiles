"""VibeVoice re-run for files whose VibeVoice output is missing (too long for the GPUs).

    uv run python -m librivox_mult.vibevoice_rerun plan  [--max-min 30]
    uv run python -m librivox_mult.vibevoice_rerun cut   [--work W] [--reading SLUG]   # writes FLAC pieces
    uv run python -m librivox_mult.vibevoice_rerun stitch                                # after VibeVoice ran

- **plan:** for each non-excluded file with no (or empty) VibeVoice output, choose cut
  points so every piece is at most ``--max-min`` minutes. Each cut is at the midpoint of
  the longest gap between wav2vec2 words within +-90 s of an even target, so no speech
  is cut. Writes ``/out/vibevoice_rerun/plan.jsonl``.
- **cut:** decode each piece with ffmpeg (sample-accurate, ``-ss``/``-to`` after
  ``-i``) to ``/out/vibevoice_rerun/pieces/<work>/<slug>/<stem>.pNN.flac``, at the
  native rate, mono.
- **stitch:** read VibeVoice JSON per piece (``<stem>.pNN.json`` next to the FLAC),
  shift Start/End by the piece start onto the recording timeline, concatenate, and
  write ``/out/vibevoice/<work>/<slug>/<stem>.json`` (same list format as /data, plus
  ``Piece`` per segment).

Speaker labels are per piece: VibeVoice numbers speakers independently in each piece,
so ``Speaker`` is rewritten as ``"<piece>:<speaker>"``. Matching speakers across pieces
is a Task 2 question.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

OUT = Path("/out")
WINDOW = 90.0      # search +-90 s around each target for a gap
MARGIN = 60.0      # aim for pieces 1 min under the cap, so snapping can't exceed it


def needs_rerun(sec: dict) -> bool:
    vv = sec["asr"]["vibevoice"]
    return not sec.get("excluded") and (not vv.get("present") or not vv.get("end"))


def plan_cuts(duration: float, words: list[dict], max_s: float) -> list[dict]:
    """Pieces [{index, start, end, gap}] covering [0, duration], each <= max_s.

    ``gap`` is the silence (s) at the cut that ends the piece (None for the last).
    """
    if duration <= max_s:
        return [{"index": 1, "start": 0.0, "end": duration, "gap": None}]
    n = math.ceil(duration / (max_s - MARGIN))
    gaps = [(words[i]["end"], words[i + 1]["start"]) for i in range(len(words) - 1)
            if words[i + 1]["start"] > words[i]["end"]]
    cuts, prev = [], 0.0
    for k in range(1, n):
        target = k * duration / n
        lo, hi = max(prev + 1.0, target - WINDOW), min(prev + max_s, target + WINDOW)
        cand = [(b - a, (a + b) / 2) for a, b in gaps if lo <= (a + b) / 2 <= hi]
        if cand:
            g, t = max(cand)
        else:  # no word gap in the window (unlikely): cut at the target
            g, t = 0.0, min(target, prev + max_s)
        cuts.append((round(t, 3), round(g, 3)))
        prev = t
    bounds = [0.0] + [c for c, _ in cuts] + [duration]
    pieces = [{"index": i + 1, "start": bounds[i], "end": bounds[i + 1],
               "gap": cuts[i][1] if i < len(cuts) else None} for i in range(len(bounds) - 1)]
    assert all(p["end"] - p["start"] <= max_s + 1e-6 for p in pieces), pieces
    return pieces


def stitch(pieces: list[dict], outputs: list[list[dict]]) -> list[dict]:
    """Shift each piece's VibeVoice segments onto the recording timeline."""
    out = []
    for p, segs in zip(pieces, outputs):
        for s in segs:
            t = dict(s)
            t["Start"] = round(s["Start"] + p["start"], 3)
            t["End"] = round(s["End"] + p["start"], 3)
            t["Piece"] = p["index"]
            if s.get("Speaker") is not None:
                t["Speaker"] = f"{p['index']}:{s['Speaker']}"
            out.append(t)
    return out


def piece_path(root: Path, row: dict, idx: int, ext: str) -> Path:
    return root / "pieces" / row["work"] / row["slug"] / f"{row['stem']}.p{idx:02d}.{ext}"


def cmd_plan(a) -> int:
    rows = []
    for line in a.inventory.read_text().splitlines():
        r = json.loads(line)
        for s in r["sections"]:
            if not needs_rerun(s):
                continue
            w2v = a.out / "w2v" / r["work"] / r["librivox_slug"] / f"{s['stem']}.json"
            words = json.loads(w2v.read_text())["words"] if w2v.exists() else []
            rows.append({"work": r["work"], "reading_id": r["reading_id"], "slug": r["librivox_slug"],
                         "stem": s["stem"], "path": s["path"], "duration": s["duration"],
                         "pieces": plan_cuts(s["duration"], words, a.max_min * 60)})
    d = a.out / "vibevoice_rerun"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "plan.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = sum(len(r["pieces"]) for r in rows)
    small = min((p["gap"] for r in rows for p in r["pieces"] if p["gap"] is not None), default=None)
    print(f"{len(rows)} files, {n} pieces, longest piece "
          f"{max(p['end'] - p['start'] for r in rows for p in r['pieces']) / 60:.1f} min, "
          f"smallest cut gap {small} s -> {d / 'plan.jsonl'}")
    return 0


def samples(path: Path) -> int:
    """Decoded sample count (mono, native rate)."""
    p = subprocess.run(f"ffmpeg -nostdin -v error -i '{path}' -map 0:a:0 -ac 1 -f u8 - | wc -c",
                       shell=True, capture_output=True, text=True, check=True)
    return int(p.stdout.strip())


def cut_file(args) -> tuple[str, str]:
    """Cut one file into its pieces. Output seeking (-ss after -i) decodes from the
    start and cuts in decoded audio, so piece times match the full decode exactly
    (input seeking in VBR mp3 can land off). Verifies the pieces' samples add up."""
    d, r, force = args
    todo = [p for p in r["pieces"] if force or not piece_path(d, r, p["index"], "flac").exists()]
    for p in todo:
        dst = piece_path(d, r, p["index"], "flac")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp.flac")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", r["path"], "-ss", str(p["start"]),
                        "-to", str(p["end"]), "-map", "0:a:0", "-ac", "1", str(tmp)], check=True)
        tmp.replace(dst)
    if not todo:
        return r["stem"], "skipped"
    total = sum(samples(piece_path(d, r, p["index"], "flac")) for p in r["pieces"])
    whole = samples(Path(r["path"]))
    return r["stem"], "ok" if abs(total - whole) <= len(r["pieces"]) else f"MISMATCH pieces={total} file={whole}"


def cmd_cut(a) -> int:
    from concurrent.futures import ThreadPoolExecutor

    d = a.out / "vibevoice_rerun"
    rows = [json.loads(x) for x in (d / "plan.jsonl").read_text().splitlines()]
    rows = [r for r in rows if (not a.work or r["work"] in a.work) and (not a.reading or r["slug"] in a.reading)]
    bad = 0
    with ThreadPoolExecutor(a.jobs) as ex, (d / "cut.log").open("a") as log:
        for stem, status in ex.map(cut_file, [(d, r, a.force) for r in rows]):
            log.write(f"{stem}\t{status}\n")
            bad += status.startswith("MISMATCH")
    print(f"{len(rows)} files cut; {bad} sample-count mismatches (see {d / 'cut.log'})")
    return 1 if bad else 0


def cmd_stitch(a) -> int:
    d = a.out / "vibevoice_rerun"
    done = missing = 0
    for line in (d / "plan.jsonl").read_text().splitlines():
        r = json.loads(line)
        outs = [piece_path(d, r, p["index"], "json") for p in r["pieces"]]
        if not all(o.exists() for o in outs):
            missing += 1
            continue
        segs = stitch(r["pieces"], [json.loads(o.read_text()) for o in outs])
        dst = a.out / "vibevoice" / r["work"] / r["slug"] / f"{r['stem']}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(segs, ensure_ascii=False, indent=1))
        done += 1
    print(f"stitched {done} files; {missing} still missing piece outputs")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "cut", "stitch"])
    ap.add_argument("--inventory", type=Path, default=OUT / "inventory.jsonl")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-min", type=float, default=30.0)
    ap.add_argument("--work", action="append")
    ap.add_argument("--reading", action="append")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "cut": cmd_cut, "stitch": cmd_stitch}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
