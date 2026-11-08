"""Stage 6: pauses at candidate boundaries.

    uv run python -m librivox_mult.stage6_pauses energy [--work W] [--reading ID] [--jobs N]
    uv run python -m librivox_mult.stage6_pauses pauses [--work W] [--reading ID] [--force] [--jobs N]

``energy``: per audio file, frame energy (10 ms RMS, dB) from an ffmpeg 16 kHz mono decode,
saved as /out/stage6/energy/<work>/<slug>/<stem>.npy (float16). Skips existing files.

``pauses``: per Stage 5 section, every candidate boundary between reference units
(sentence ``S``, verse line ``L``, block ``P``, header/footer element ``H``). The gap is
between the last wav2vec2 word of the unit before it and the next wav2vec2 word, so
inserted material counts with the following unit. Inside that gap (plus one CTC frame),
``silence`` is the longest run of frames below the file's silence threshold (where Stage 7
cuts) and ``pause`` the extent from the first to the last quiet frame (breaths inside a
pause stay in it):
threshold = floor + SIL_FRAC * (speech - floor), with floor and speech the 10th and
90th energy percentiles. Respiro breaths overlapping the gap are recorded (``breath``)
but never decide anything. Word gaps inside units are measured the same way as a
per-reading baseline. Writes /out/stage6/<work>/<reading_id>.json.gz and the review in
/out/review/stage6/.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

OUT = Path("/out")
VERSION = "stage6-0.1"
FRAME = 0.01
SR = 16000
SIL_FRAC = 0.2
PAUSE_MIN = 0.15  # seconds of silence that count as a pause in the summaries
CTC_PAD = 0.02


def energy_path(r: dict, stem: str) -> Path:
    return OUT / "stage6" / "energy" / r["work"] / r["librivox_slug"] / f"{stem}.npy"


def frame_db(path: str) -> np.ndarray:
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    n = int(SR * FRAME)
    x = x[: len(x) // n * n].reshape(-1, n)
    return (10 * np.log10(np.mean(x * x, axis=1) + 1e-10)).astype(np.float16)


def energy_job(job: tuple[str, str]) -> str:
    src, dst = job
    p = Path(dst)
    if p.exists():
        return "done"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npy")
    np.save(tmp, frame_db(src))
    tmp.replace(p)
    return "made"


def threshold(db: np.ndarray) -> float:
    lo, hi = np.percentile(db.astype(np.float32), [10, 90])
    return float(lo + SIL_FRAC * (hi - lo))


def silence_in(db: np.ndarray, thr: float, t0: float, t1: float) -> tuple[tuple[float, float], float] | None:
    """Inside [t0, t1]: the longest run of frames below ``thr`` (where to cut) and the
    extent from the first to the last quiet frame (the pause, breaths included)."""
    f0, f1 = max(0, int(t0 / FRAME)), min(len(db), int(np.ceil(t1 / FRAME)))
    if f1 <= f0:
        return None
    quiet = np.concatenate([[False], db[f0:f1] < thr, [False]])
    edges = np.flatnonzero(np.diff(quiet.astype(np.int8)))
    if not len(edges):
        return None
    starts, ends = edges[::2], edges[1::2]
    k = int(np.argmax(ends - starts))
    longest = (round((f0 + starts[k]) * FRAME, 3), round((f0 + ends[k]) * FRAME, 3))
    return longest, round((ends[-1] - starts[0]) * FRAME, 3)


def unit_keys(row: list, s1: dict, els: list[dict]) -> dict:
    """The units a reference row belongs to: S (sentence or element), L (verse line), P (block)."""
    if row[3] >= 0:
        e = els[row[3]]["id"]
        return {"S": e, "L": e, "P": e, "H": e}
    t = row[1]
    line = s1["line"][t]
    return {"S": s1["sent"][t], "L": (s1["block"][t], line) if line is not None else None,
            "P": s1["block"][t], "H": "body"}


def boundaries(sec: dict, s1: dict, db: np.ndarray | None, thr: float, breaths: list) -> tuple[list[dict], list[float]]:
    words, rows, els = sec["words"], sec["ref"], sec["elements"]
    n = len(words)
    last_hyp = [-1] * n  # index into words of the last hyp word at or before p
    cur = -1
    for p, w in enumerate(words):
        if w[1] >= 0:
            cur = p
        last_hyp[p] = cur
    next_hyp = [-1] * (n + 1)
    cur = -1
    for p in range(n - 1, -1, -1):
        if words[p][1] >= 0:
            cur = p
        next_hyp[p] = cur

    def measure(pa: int) -> dict:
        a, b = last_hyp[pa], next_hyp[pa + 1] if pa + 1 < n else -1
        if a < 0 or b < 0:
            return {"prev_end": words[a][4] if a >= 0 else None, "next_start": words[b][3] if b >= 0 else None,
                    "gap": None, "silence": None, "pause": None}
        e, s = words[a][4], words[b][3]
        q = silence_in(db, thr, e, s + CTC_PAD) if db is not None else None
        return {"prev_end": e, "next_start": s, "gap": round(s - e, 3), "silence": q[0] if q else None,
                "pause": q[1] if q else 0.0}

    refp = [(p, w[0]) for p, w in enumerate(words) if w[0] >= 0]
    out, baseline = [], []
    for (pa, ia), (pb, ib) in zip(refp, refp[1:]):
        ka, kb = unit_keys(rows[ia], s1, els), unit_keys(rows[ib], s1, els)
        kinds = [k for k in ("H", "S", "L", "P") if ka[k] != kb[k] and (k != "L" or ka[k] or kb[k])]
        if not kinds:
            if words[pa][2] == "cor" and pb == pa + 1 and words[pb][2] == "cor" and db is not None:
                m = measure(pa)
                baseline.append(m["pause"])
            continue
        m = measure(pa)
        if m["silence"]:
            s0, s1_ = m["silence"]
            m["breath"] = any(b0 < s1_ and b1 > s0 for b0, b1 in breaths)
        b = {"p": pa, "kinds": kinds, "after": str(ka["S"]), "before": str(kb["S"]),
             "verse": bool(ka["L"] or kb["L"]), **m}
        out.append(b)
    return out, baseline


def load_stage1(tid: str, cache: dict) -> dict:
    if tid not in cache:
        tk = json.load(gzip.open(OUT / "stage1" / f"{tid}.json.gz", "rt"))["tokens"]
        cache[tid] = {"sent": tk["sent"], "line": tk["line"], "block": tk["block"]}
    return cache[tid]


def load_breaths(sec_inv: dict) -> list:
    rp = sec_inv.get("asr", {}).get("respiro", {})
    if not rp.get("present"):
        return []
    try:
        return json.loads(Path(rp["path"]).read_text().splitlines()[0]).get("breaths", [])
    except (OSError, ValueError, IndexError):
        return []


def pauses_job(args: tuple[dict, bool]) -> dict:
    r, force = args
    src = OUT / "stage5" / r["work"] / f"{r['reading_id']}.json.gz"
    dst = OUT / "stage6" / r["work"] / f"{r['reading_id']}.json.gz"
    if not src.exists():
        return {"reading_id": r["reading_id"], "status": "no_input"}
    s5 = json.load(gzip.open(src, "rt"))
    key = hashlib.sha256((s5["input_sha"] + s5["version"] + VERSION).encode()).hexdigest()[:16]
    if dst.exists() and not force:
        try:
            if json.load(gzip.open(dst, "rt")).get("input_sha") == key:
                return {"reading_id": r["reading_id"], "status": "done"}
        except (OSError, ValueError):
            pass
    inv = {s["file"].rsplit(".", 1)[0]: s for s in r["sections"]}
    cache: dict = {}
    out = {"version": VERSION, "input_sha": key, "reading_id": r["reading_id"], "work": r["work"], "sections": []}
    missing = 0
    for sec in s5["sections"]:
        ep = energy_path(r, sec["stem"])
        db = np.load(ep) if ep.exists() else None
        missing += db is None
        thr = threshold(db) if db is not None else 0.0
        bs, base = boundaries(sec, load_stage1(sec["text"], cache), db, thr, load_breaths(inv.get(sec["stem"], {})))
        out["sections"].append({"stem": sec["stem"], "threshold_db": round(thr, 2), "boundaries": bs,
                                "baseline": {"n": len(base),
                                             "pause_rate": round(sum(x >= PAUSE_MIN for x in base) / len(base), 4) if base else None,
                                             "median": round(float(np.median(base)), 3) if base else None}})
    if missing:
        return {"reading_id": r["reading_id"], "status": "no_energy", "n": missing}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(out, f, separators=(",", ":"))
    tmp.replace(dst)
    return {"reading_id": r["reading_id"], "status": "made"}


def classify(rates: dict) -> str:
    """Boundary behaviour from pause rates (fraction of boundaries with >= PAUSE_MIN silence)."""
    s, ln = rates.get("sent_end"), rates.get("line_end_only")
    sp = s is not None and s >= 0.5
    if ln is None:
        return "sentences" if sp else "neither"
    lp = ln >= 0.5
    return {(True, True): "both", (True, False): "sentences", (False, True): "lines", (False, False): "neither"}[(sp, lp)]


def reading_rates(d: dict) -> dict:
    c: Counter = Counter()
    for sec in d["sections"]:
        for b in sec["boundaries"]:
            if b["pause"] is None or "H" in b["kinds"]:
                continue
            p = b["pause"] >= PAUSE_MIN
            if "S" in b["kinds"]:
                c["sent_end"] += 1
                c["sent_end_p"] += p
                c["sent_end_breath"] += bool(b.get("breath"))
                if b["verse"] and "L" not in b["kinds"]:
                    c["sent_mid_line"] += 1
                    c["sent_mid_line_p"] += p
            elif "L" in b["kinds"]:
                c["line_end_only"] += 1
                c["line_end_only_p"] += p
            elif "P" in b["kinds"]:
                c["block"] += 1
                c["block_p"] += p
        bl = sec["baseline"]
        if bl["n"]:
            c["word"] += bl["n"]
            c["word_p"] += round(bl["pause_rate"] * bl["n"])
    rates = {k: round(c[k + "_p"] / c[k], 3) if c[k] else None
             for k in ("sent_end", "line_end_only", "sent_mid_line", "block", "word")}
    return {"counts": {k: c[k] for k in ("sent_end", "line_end_only", "sent_mid_line", "word")},
            "rates": rates, "breath_at_sent_end": round(c["sent_end_breath"] / c["sent_end"], 3) if c["sent_end"] else None,
            "behaviour": classify(rates)}


def write_review(out: Path, inv: list[dict]) -> None:
    rdir = out / "review" / "stage6"
    rdir.mkdir(parents=True, exist_ok=True)
    verse = {r["reading_id"]: r.get("verse") for r in inv}
    cls: Counter = Counter()
    rows = []
    for p in sorted(out.glob("stage6/*/*.json.gz")):
        d = json.load(gzip.open(p, "rt"))
        rr = reading_rates(d)
        d_sum = {"version": d["version"], **rr}
        (p.parent / (d["reading_id"] + ".summary.json")).write_text(json.dumps(d_sum))
        has_lines = rr["counts"]["line_end_only"] > 0
        cls[(has_lines, rr["behaviour"])] += 1
        rows.append([d["work"], d["reading_id"], verse.get(d["reading_id"]), rr["behaviour"]]
                    + [rr["counts"][k] for k in ("sent_end", "line_end_only", "word")]
                    + [rr["rates"][k] for k in ("sent_end", "line_end_only", "sent_mid_line", "block", "word")]
                    + [rr["breath_at_sent_end"]])
    with (rdir / "readings.tsv").open("w") as f:
        f.write("work\treading_id\tverse\tbehaviour\tn_sent_end\tn_line_end_only\tn_word\t"
                "p_sent_end\tp_line_end_only\tp_sent_mid_line\tp_block\tp_word\tbreath_at_sent_end\n")
        for row in sorted(rows, key=lambda x: (x[3], x[0])):
            f.write("\t".join("" if v is None else str(v) for v in row) + "\n")
    with (rdir / "summary.txt").open("w") as f:
        f.write(f"readings: {len(rows)}; pause = silence >= {PAUSE_MIN}s below floor+{SIL_FRAC}*(speech-floor)\n")
        for (hl, b), v in sorted(cls.items()):
            f.write(f"{'verse' if hl else 'prose'}\t{b}\t{v}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["energy", "pauses"])
    ap.add_argument("--work", action="append")
    ap.add_argument("--reading", action="append")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args(argv)
    inv = [json.loads(x) for x in (OUT / "inventory.jsonl").read_text().splitlines()]
    sel = [r for r in inv if not (a.work and r["work"] not in a.work)
           and not (a.reading and r["reading_id"] not in a.reading and r["librivox_slug"] not in a.reading)]
    st: Counter = Counter()
    with Pool(a.jobs) as pool:
        if a.cmd == "energy":
            jobs = [(s["path"], str(energy_path(r, s["file"].rsplit(".", 1)[0])))
                    for r in sel for s in r["sections"] if not s.get("excluded")]
            for k, res in enumerate(pool.imap_unordered(energy_job, jobs), 1):
                st[res] += 1
                if k % 200 == 0:
                    print(k, len(jobs), dict(st), flush=True)
        else:
            for res in pool.imap_unordered(pauses_job, [(r, a.force) for r in sel]):
                st[res["status"]] += 1
            write_review(OUT, inv)
    print(dict(st))
    return 0


if __name__ == "__main__":
    sys.exit(main())
