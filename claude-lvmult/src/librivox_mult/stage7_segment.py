"""Stage 7: committed segmentations.

    uv run python -m librivox_mult.stage7_segment [--work W] [--reading ID] [--force] [--jobs N]

Per reading, from Stage 5 (word alignment) and Stage 6 (pauses), writes
/out/stage7/<work>/<reading_id>.json.gz with complete segmentations of every aligned
section:

- ``S``: sentences (prose sentences; verse sentences use the poem hierarchy).
- ``L``: verse lines, only for readings whose text has verse. Non-verse material in an
  ``L`` segmentation is segmented by sentence, so ``L`` still covers the whole reading.

Header and footer elements are their own segments in both. Every reference row is in
exactly one segment per segmentation. Inserted hypothesis words between two units go
to the following segment; before the first unit, to the first one.

Cuts (rules in docs/segmentation.md): with a silence of at least MIN_CUT_SIL in the
boundary gap, cut inside it, keeping up to PAD of it on each side (``pause``); otherwise
cut at the midpoint between the two words (``forced``). Units with no spoken words
are kept with zero duration and ``spoken: false``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

OUT = Path("/out")
VERSION = "stage7-0.4"
MIN_CUT_SIL = 0.1
PAD = 0.25
EDGE_PAD = 0.25
X_MIN_WORDS = 15  # a run of this many inserted words is its own segment (unreferenced speech)


def norm_word(w: str) -> str:
    return re.sub(r"[^a-z']", "", w.lower()).strip("'")


def load_stage1(tid: str, cache: dict) -> dict:
    if tid not in cache:
        d = json.load(gzip.open(OUT / "stage1" / f"{tid}.json.gz", "rt"))
        cache[tid] = d
    return cache[tid]


def unit_of(row: list, d1: dict | None, seg: str, els: list[dict]) -> tuple:
    if row[3] >= 0:
        return ("H", els[row[3]]["id"])
    tk = d1["tokens"]
    t = row[1]
    if seg == "L" and tk["line"][t] is not None and d1["blocks"][tk["block"][t]].get("poem") is not None:
        return ("L", tk["block"][t], tk["line"][t])
    return ("S", tk["sent"][t])


def unit_ids(u: tuple, d1: dict | None, text: str, section: str) -> tuple[str | None, str]:
    """(text_id, the supervision id suffix after '{work}/{reader}/') for a unit."""
    if u[0] in ("H", "X"):
        return None, f"{section}/{u[1]}"
    if u[0] == "L":
        b = d1["blocks"][u[1]]
        tail = f"{b['poem']:03d}/{b['stanza']:02d}/{u[2]:02d}"
        return f"{text}/{tail}", f"L/{tail}"
    s = d1["sentences"][u[1]]
    if s.get("unit") == "poem":
        tail = f"{s['poem']:03d}/s{s['n']:02d}"
    else:
        tail = f"{s['division']:03d}/{s['para']:04d}/{s['n']:02d}"
    return f"{text}/{tail}", f"S/{tail}"


def punct_edges(tk: dict, t0: int, t1: int, key: str) -> tuple[int, int]:
    """Widen [t0, t1] over the punctuation tokens next to it in the same sentence (line),
    so `text` keeps opening quotes and final stops, which have no spoken words."""
    while t0 > 0 and tk["punct"][t0 - 1] and tk[key][t0 - 1] == tk[key][t0]:
        t0 -= 1
    while t1 + 1 < len(tk["punct"]) and tk["punct"][t1 + 1] and tk[key][t1 + 1] == tk[key][t1]:
        t1 += 1
    return t0, t1


def cut_at(b: dict | None) -> tuple[float, float, str, float | None]:
    """(end of the segment before, start of the one after, boundary kind, pause)."""
    if b is None or b.get("prev_end") is None or b.get("next_start") is None:
        return None, None, "forced", None
    sil = b.get("silence")
    if sil and sil[1] - sil[0] >= MIN_CUT_SIL:
        pad = min(PAD, (sil[1] - sil[0]) / 2)
        return round(sil[0] + pad, 3), round(sil[1] - pad, 3), "pause", b.get("pause")
    mid = round((b["prev_end"] + b["next_start"]) / 2, 3)
    return mid, mid, "forced", b.get("pause") or 0.0


def segment_section(sec5: dict, sec6: dict | None, sec4: dict | None, hyp: list[str], d1: dict | None,
                    seg: str, section: str, duration: float, multi_text: bool, measure=None) -> list[dict]:
    words, rows, els = sec5["words"], sec5["ref"], sec5["elements"]
    n = len(words)
    unit = [None] * n
    nxt = None
    for p in range(n - 1, -1, -1):
        if words[p][0] >= 0:
            nxt = unit_of(rows[words[p][0]], d1, seg, els)
        unit[p] = nxt
    last = None
    for p in range(n):
        if unit[p] is None:
            unit[p] = last
        last = unit[p]
    p = 0
    nx = 0
    while p < n:
        q = p
        while q < n and words[q][0] < 0:
            q += 1
        if q - p >= X_MIN_WORDS:
            nx += 1
            for r in range(p, q):
                unit[r] = ("X", f"X{nx:02d}")
        p = q + 1
    runs = []
    for p in range(n):
        if runs and runs[-1][0] == unit[p]:
            runs[-1][2] = p + 1
        else:
            runs.append([unit[p], p, p + 1])
    bmap = {b["p"]: b for b in (sec6 or {}).get("boundaries", [])}
    partial_tok = set()
    for sp in sec5["spans"]:
        if sp["type"] in ("partial", "revision"):
            after = next((rows[words[q][0]][0] for q in range(sp["pairs"][1], n) if words[q][0] >= 0), "")
            partial_tok.update(j for j in sp["hyp"] if len(hyp[j]) < len(after) and after.startswith(hyp[j]))
    nsw = {x["nsw"]: x for x in (sec4 or {}).get("nsw", [])}
    units5 = {str(u["unit"]): u for u in sec5["units"]}
    seen: Counter = Counter()
    segs = []
    for k, (u, p0, p1) in enumerate(runs):
        ref_rows = [words[p][0] for p in range(p0, p1) if words[p][0] >= 0]
        hyp_p = [p for p in range(p0, p1) if words[p][1] >= 0]
        text_id, sid = unit_ids(u, d1, sec5["text"], section)
        if multi_text and u[0] not in ("H", "X"):
            sid = f"{sid[:2]}{sec5['text']}/{sid[2:]}"
        seen[sid] += 1
        if seen[sid] > 1:
            sid = f"{sid}.{seen[sid]}"
        last_ref = max(p for p in range(p0, p1) if words[p][0] >= 0) if ref_rows else None
        segs.append({"u": u, "p": [p0, p1], "id": sid, "text_id": text_id, "ref_rows": ref_rows,
                     "hyp_p": hyp_p, "last_ref": last_ref})
    for k, s in enumerate(segs):
        if k == 0:
            first = next((words[p][3] for p in s["hyp_p"]), None)
            s["start"], s["b_start"], s["pause_before"] = (max(0.0, first - EDGE_PAD) if first is not None else None), "edge", None
        if k + 1 < len(segs):
            b = bmap.get(s["last_ref"]) if s["last_ref"] is not None and s["p"][1] == s["last_ref"] + 1 else None
            if b is None and measure is not None:
                b = measure(s["p"][1] - 1)
            end, start, kind, pause = cut_at(b)
            s["end"], s["b_end"], s["pause_after"] = end, kind, pause
            segs[k + 1]["start"], segs[k + 1]["b_start"], segs[k + 1]["pause_before"] = start, kind, pause
        else:
            lastt = next((words[p][4] for p in reversed(s["hyp_p"])), None)
            s["end"], s["b_end"], s["pause_after"] = (min(duration, lastt + EDGE_PAD) if lastt is not None else None), "edge", None
    out = []
    for s in segs:
        spoken = bool(s["hyp_p"])
        if spoken:
            t0, t1 = s["start"], s["end"]
            w0, w1 = words[s["hyp_p"][0]][3], words[s["hyp_p"][-1]][4]
            t0 = w0 if t0 is None or t0 > w0 else t0
            t1 = w1 if t1 is None or t1 < w1 else t1
        else:
            c = [x for x in (s["start"], s["end"]) if x is not None]
            t0 = t1 = round(sum(c) / len(c), 3) if c else 0.0
        align, idx = [], {}
        for p in s["hyp_p"]:
            _, j, _, st, en, sc = words[p]
            idx[p] = len(align)
            align.append({"symbol": hyp[j] + ("-" if j in partial_tok else ""), "start": st,
                          "duration": round(en - st, 3), "score": sc})
        edits = [[words[p][2], words[p][0], idx.get(p, -1)] for p in range(*s["p"])]
        dys = []
        for sp in sec5["spans"]:
            if sp["pairs"][1] <= s["p"][0] or sp["pairs"][0] >= s["p"][1] or sp["type"] in ("orth",):
                continue
            inside = range(max(sp["pairs"][0], s["p"][0]), min(sp["pairs"][1], s["p"][1]))
            toks = [idx[p] for p in inside if p in idx]
            refs = [rows[words[p][0]][1] for p in inside if words[p][0] >= 0]
            dys.append({"type": sp["type"], "tokens": toks, "ref_tokens": refs, "whisperx": sp.get("whisperx")})
        u = s["u"]
        if u[0] == "X":
            text, src_span, nsws, hf = "", None, [], None
        elif u[0] == "H":
            el = next(e for e in sec5["elements"] if e["id"] == u[1])
            text, src_span, nsws = el["source"], None, []
            hf = {"id": el["id"], "template": el["kind"], "variant": el["variant"], "match_score": el["score"]}
        else:
            toks = [rows[i][1] for i in s["ref_rows"]]
            tk = d1["tokens"]
            k0, k1 = punct_edges(tk, min(toks), max(toks), "line" if u[0] == "L" else "sent")
            c0, c1 = tk["start"][k0], tk["end"][k1]
            text, src_span, hf = d1["text"][c0:c1], [c0, c1], None
            nsws = [nsw[k] for k in sorted({rows[i][2] for i in s["ref_rows"] if rows[i][2] >= 0}) if k in nsw]
        cor = sum(1 for p in range(*s["p"]) if words[p][2] == "cor")
        conf_src = units5.get(str(u[1])) if u[0] in ("S", "H") else None
        if conf_src is not None and len(s["ref_rows"]) == conf_src["n"]:
            conf = conf_src["confidence"]
        else:
            sc = [words[p][5] for p in range(*s["p"]) if words[p][2] == "cor"]
            conf = round(cor / max(1, len(s["ref_rows"])) * (sum(sc) / len(sc) if sc else 0.0), 4)
        out.append({"id": s["id"], "text_id": s["text_id"], "unit": u[0] if u[0] not in ("H", "X") else s["id"].rsplit("/", 1)[1][0],
                    "start": round(t0, 3), "duration": round(max(0.0, t1 - t0), 3), "spoken": spoken,
                    "text": text, "src_span": src_span, "verbatim": " ".join(a["symbol"] for a in align),
                    "alignment": align, "edits": edits,
                    "boundary": {"start": s["b_start"], "end": s["b_end"]},
                    "pause_before": s["pause_before"], "pause_after": s["pause_after"],
                    "dysfluencies": dys, "nsw": nsws, "confidence": conf, "header_footer": hf})
    return out


def energy_measure(r: dict, stem: str, words: list):
    """A Stage-6-style boundary measurement after pair p, for cuts Stage 6 didn't measure
    (around unreferenced-speech segments)."""
    import numpy as np

    from librivox_mult import stage6_pauses as s6

    ep = s6.energy_path(r, stem)
    if not ep.exists():
        return None
    db = np.load(ep)
    thr = s6.threshold(db)

    def measure(p: int) -> dict | None:
        a = next((q for q in range(p, -1, -1) if words[q][1] >= 0), None)
        b = next((q for q in range(p + 1, len(words)) if words[q][1] >= 0), None)
        if a is None or b is None:
            return None
        e, st = words[a][4], words[b][3]
        q = s6.silence_in(db, thr, e, st + s6.CTC_PAD)
        return {"prev_end": e, "next_start": st, "silence": q[0] if q else None, "pause": q[1] if q else 0.0}
    return measure


def check(segs: list[dict], n_rows: int, duration: float) -> list[str]:
    errs = []
    covered = Counter(e[1] for s in segs for e in s["edits"] if e[1] >= 0)
    if set(covered) != set(range(n_rows)) or any(v != 1 for v in covered.values()):
        errs.append("ref_coverage")
    for a, b in zip(segs, segs[1:]):
        if a["start"] + a["duration"] > b["start"] + 1e-6:
            errs.append(f"overlap:{a['id']}|{b['id']}")
    for s in segs:
        if s["start"] < 0 or s["start"] + s["duration"] > duration + 1e-3:
            errs.append(f"bounds:{s['id']}")
    return errs


def process(args: tuple[dict, bool]) -> dict:
    r, force = args
    p5 = OUT / "stage5" / r["work"] / f"{r['reading_id']}.json.gz"
    p6 = OUT / "stage6" / r["work"] / f"{r['reading_id']}.json.gz"
    p4 = OUT / "stage4" / r["work"] / f"{r['reading_id']}.json"
    dst = OUT / "stage7" / r["work"] / f"{r['reading_id']}.json.gz"
    if not (p5.exists() and p6.exists()):
        return {"reading_id": r["reading_id"], "status": "no_input"}
    s5 = json.load(gzip.open(p5, "rt"))
    s6 = json.load(gzip.open(p6, "rt"))
    key = hashlib.sha256((s5["input_sha"] + s6["input_sha"] + VERSION).encode()).hexdigest()[:16]
    if dst.exists() and not force:
        try:
            if json.load(gzip.open(dst, "rt")).get("input_sha") == key:
                return {"reading_id": r["reading_id"], "status": "done"}
        except (OSError, ValueError):
            pass
    s4 = {s["stem"]: s for s in json.loads(p4.read_text())["sections"]} if p4.exists() else {}
    s6s = {s["stem"]: s for s in s6["sections"]}
    inv = {s["file"].rsplit(".", 1)[0]: s for s in r["sections"]}
    cache: dict = {}
    multi = len({s["text"] for s in s5["sections"]}) > 1
    has_verse = any(rows[1] >= 0 and (d := load_stage1(sec["text"], cache))["blocks"][d["tokens"]["block"][rows[1]]].get("poem") is not None
                    for sec in s5["sections"] for rows in sec["ref"] if rows[3] < 0)
    segmentations = ["S", "L"] if has_verse else ["S"]
    out = {"version": VERSION, "input_sha": key, "reading_id": r["reading_id"], "work": r["work"],
           "segmentations": {g: [] for g in segmentations}, "errors": []}
    for sec in s5["sections"]:
        stem = sec["stem"]
        iv = inv.get(stem, {})
        hyp = [norm_word(w["word"]) for w in
               json.loads((OUT / "w2v" / r["work"] / r["librivox_slug"] / f"{stem}.json").read_text())["words"]]
        d1 = load_stage1(sec["text"], cache)
        measure = energy_measure(r, stem, sec["words"])
        for g in segmentations:
            segs = segment_section(sec, s6s.get(stem), s4.get(stem), hyp, d1, g, iv.get("section", stem),
                                   iv.get("duration", 1e9), multi, measure)
            errs = check(segs, len(sec["ref"]), iv.get("duration", 1e9))
            out["errors"] += [f"{g}:{stem}:{e}" for e in errs]
            out["segmentations"][g].append({"stem": stem, "section": iv.get("section"), "text": sec["text"],
                                            "readers": iv.get("readers"), "segments": segs})
    for g in segmentations:
        unique_ids(out["segmentations"][g])
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(dst)
    return {"reading_id": r["reading_id"], "status": "made", "errors": len(out["errors"])}


def unique_ids(secs: list[dict]) -> None:
    """Make ids unique across a reading: a unit read again in a later section (alternate
    versions, overlapping sections, re-read fables) gets the next free `.N`, in section order."""
    used: set = set()
    for sec in secs:
        for seg in sec["segments"]:
            sid = seg["id"]
            if sid in used:
                base = re.sub(r"\.\d+$", "", sid)
                n = 2
                while f"{base}.{n}" in used:
                    n += 1
                sid = seg["id"] = f"{base}.{n}"
            used.add(sid)


def write_review(out: Path) -> None:
    rdir = out / "review" / "stage7"
    rdir.mkdir(parents=True, exist_ok=True)
    c: Counter = Counter()
    errs = []
    durs: dict = {"S": [], "L": []}
    for p in sorted(out.glob("stage7/*/*.json.gz")):
        d = json.load(gzip.open(p, "rt"))
        errs += [(d["reading_id"], e) for e in d["errors"]]
        for g, secs in d["segmentations"].items():
            for sec in secs:
                for s in sec["segments"]:
                    body = s["header_footer"] is None
                    c[(g, "segments")] += 1
                    c[(g, "unspoken")] += not s["spoken"]
                    c[(g, "header_footer")] += not body
                    if s["spoken"]:
                        c[(g, "start_" + s["boundary"]["start"])] += 1
                        durs[g].append(s["duration"])
    import statistics as st

    with (rdir / "summary.txt").open("w") as f:
        for g in ("S", "L"):
            if not durs[g]:
                continue
            ds = sorted(durs[g])
            f.write(f"{g}: {c[(g, 'segments')]} segments, {c[(g, 'header_footer')]} header/footer, "
                    f"{c[(g, 'unspoken')]} unspoken; start boundary pause {c[(g, 'start_pause')]}, "
                    f"forced {c[(g, 'start_forced')]}, edge {c[(g, 'start_edge')]}; duration median "
                    f"{st.median(ds):.2f}s, p99 {ds[int(0.99 * len(ds))]:.1f}s, max {ds[-1]:.1f}s\n")
        f.write(f"validation errors: {len(errs)}\n")
        for rid, e in errs[:200]:
            f.write(f"{rid}\t{e}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", action="append")
    ap.add_argument("--reading", action="append")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args(argv)
    inv = [json.loads(x) for x in (OUT / "inventory.jsonl").read_text().splitlines()]
    sel = [(r, a.force) for r in inv if not (a.work and r["work"] not in a.work)
           and not (a.reading and r["reading_id"] not in a.reading and r["librivox_slug"] not in a.reading)]
    st: Counter = Counter()
    with Pool(a.jobs) as pool:
        for res in pool.imap_unordered(process, sel):
            st[res["status"]] += 1
            if res.get("errors"):
                print(res["reading_id"], "errors", res["errors"], flush=True)
    write_review(OUT)
    print(dict(st))
    return 0


if __name__ == "__main__":
    sys.exit(main())
