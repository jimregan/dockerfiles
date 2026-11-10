"""Task 2: text turns (narration vs quotes) on Stage 8 supervisions.

`uv run python -m librivox_mult.task2_turns --reading lv5009 [--section 02 ...]`

For each spoken S supervision, splits `alignment.word` into turns: maximal runs of words
whose reference token lies in the same Stage 1 quote (or in none, which is narration).
Inserted words (no reference token) join the preceding word's turn. A quote left open at
the end of its block (`closed: false`) runs to the end of the block.

Writes `/out/task2/turns/<work>/<reading_id>.json.gz`:
`{sections: [{section, recording_id, turns: [{sup, quote, tok_start, tok_end, start, end,
n_words}]}]}`. `quote` is an index into the text's `quotes[]`, or null for narration;
`tok_*` index the supervision's `alignment.word` (end exclusive).
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import sys
from pathlib import Path

OUT = Path("/out")


def quote_spans(d1: dict) -> tuple[list[int], list[int], list[int]]:
    """Top-level quote spans as sorted (starts, ends, quote index); open quotes run to block end."""
    spans = []
    for i, q in enumerate(d1["quotes"]):
        if q["level"] != 0:
            continue
        end = q["end"] if q["closed"] else d1["blocks"][q["block"]]["char_end"]
        spans.append((q["start"], end, i))
    spans.sort()
    return [s[0] for s in spans], [s[1] for s in spans], [s[2] for s in spans]


def quote_of(char: int, qs: tuple) -> int | None:
    starts, ends, idx = qs
    k = bisect.bisect_right(starts, char) - 1
    return idx[k] if k >= 0 and char < ends[k] else None


def turns_of(sup: dict, tok_start: list[int], qs: tuple) -> list[dict]:
    words = sup["alignment"]["word"]
    label: list = [None] * len(words)
    known = [False] * len(words)
    for kind, _, tok, ai in sup["custom"]["edits"]:
        if ai is not None and tok is not None:
            label[ai] = quote_of(tok_start[tok], qs)
            known[ai] = True
    for i in range(len(words)):
        if not known[i] and i > 0:
            label[i] = label[i - 1]
    out = []
    for i, w in enumerate(words):
        if out and out[-1]["quote"] == label[i]:
            t = out[-1]
            t["tok_end"] = i + 1
            t["end"] = round(w["start"] + w["duration"], 3)
            t["n_words"] += 1
        else:
            out.append({"sup": sup["id"], "quote": label[i], "tok_start": i, "tok_end": i + 1,
                        "start": w["start"], "end": round(w["start"] + w["duration"], 3), "n_words": 1})
    return out


def process(work: str, rid: str, sections: set[str] | None) -> Path:
    src = OUT / "stage8" / work / rid / "supervisions_S.jsonl.gz"
    by_rec: dict = {}
    with gzip.open(src, "rt") as f:
        for line in f:
            s = json.loads(line)
            c = s["custom"]
            if c["unit"] != "S" or (sections and c["section"] not in sections):
                continue
            by_rec.setdefault(s["recording_id"], []).append(s)
    cache: dict = {}
    out = {"reading_id": rid, "work": work, "sections": []}
    for rec, sups in by_rec.items():
        tid = sups[0]["custom"]["gutenberg"]
        if tid not in cache:
            d1 = json.load(gzip.open(OUT / "stage1" / f"{tid}.json.gz", "rt"))
            cache[tid] = (d1["tokens"]["start"], quote_spans(d1))
        tok_start, qs = cache[tid]
        turns = [t for s in sorted(sups, key=lambda s: s["start"]) for t in turns_of(s, tok_start, qs)]
        out["sections"].append({"section": sups[0]["custom"]["section"], "recording_id": rec,
                                "text": tid, "turns": turns})
    dst = OUT / "task2" / "turns" / work / f"{rid}.json.gz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dst, "wt") as f:
        json.dump(out, f)
    return dst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reading", action="append", required=True)
    ap.add_argument("--section", action="append")
    a = ap.parse_args(argv)
    inv = {json.loads(x)["reading_id"]: json.loads(x) for x in (OUT / "inventory.jsonl").read_text().splitlines()}
    for rid in a.reading:
        print(process(inv[rid]["work"], rid, set(a.section) if a.section else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
