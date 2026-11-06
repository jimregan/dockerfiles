"""Stage 3: map each LibriVox section to its span of the Gutenberg text.

    uv run lvm-map [--work W] [--reading SLUG] [--force] [--jobs N]

For each section file: take the wav2vec2 words between the Stage 2 header and
footer, find word 4-grams shared with the reading's source text (Stage 1 tokens
through the same spoken-form normaliser), keep the longest chain of hits that is
increasing in both audio and text position, and report the text span covered by
the chain as tokens, blocks and division/paragraph.

Coarse by design: Stage 5 aligns word by word inside this span (extended to
paragraph boundaries). Consecutive sections should abut in the text; gaps,
overlaps and reversals are flagged for review.

Outputs:
  /out/stage3/<work>/<reading_id>.json
  /out/review/stage3/sections.tsv, flagged.tsv, summary.txt
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .spoken import spoken_words

VERSION = "stage3-0.2"
K = 4              # n-gram length for anchors
MAX_OCC = 8        # ignore n-grams occurring more often than this in the text
JUMP_RATIO = 5     # a text jump > 5x the audio gap (+ slack) splits the anchor chain
JUMP_SLACK = 300   # words
log = logging.getLogger("stage3")


# --------------------------------------------------------------------------- text
_TEXT_CACHE: dict[int, dict] = {}


def load_text(n: int, stage1: Path) -> dict:
    """Normalised word stream of a Stage 1 text, with a pointer back to each token."""
    if n in _TEXT_CACHE:
        return _TEXT_CACHE[n]
    d = json.load(gzip.open(stage1 / f"pg{n}.json.gz", "rt"))
    tk = d["tokens"]
    words, tok_of = [], []
    skip_blocks = {i for i, b in enumerate(d["blocks"]) if b["kind"] in ("toc", "separator")}
    for ti, (surf, punct) in enumerate(zip(tk["surface"], tk["punct"])):
        if punct or tk["block"][ti] in skip_blocks:  # contents entries would anchor chapter titles
            continue
        for w in spoken_words(surf):
            words.append(w.strip("'"))
            tok_of.append(ti)
    index: dict[tuple, list[int]] = defaultdict(list)
    for j in range(len(words) - K + 1):
        index[tuple(words[j:j + K])].append(j)
    index = {g: js for g, js in index.items() if len(js) <= MAX_OCC}
    t = {"n": n, "doc": d, "words": words, "tok_of": tok_of, "index": index}
    _TEXT_CACHE[n] = t
    return t


def locate_token(t: dict, word_pos: int) -> dict:
    d = t["doc"]
    ti = t["tok_of"][word_pos]
    bi = d["tokens"]["block"][ti]
    b = d["blocks"][bi]
    return {"word": word_pos, "token": ti, "char": d["tokens"]["start"][ti], "block": bi,
            "division": b["division"], "para": b["para"], "kind": b["kind"],
            "heading": " | ".join(next(x["heading"] for x in d["divisions"] if x["division"] == b["division"]))[:80]}


# --------------------------------------------------------------------------- anchors
def lis_chain(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Longest chain strictly increasing in both coordinates (O(n log n))."""
    points = sorted(points, key=lambda p: (p[0], -p[1]))
    tails, tails_idx, prev = [], [], [-1] * len(points)
    for k, (_, j) in enumerate(points):
        pos = bisect.bisect_left(tails, j)
        if pos == len(tails):
            tails.append(j)
            tails_idx.append(k)
        else:
            tails[pos] = j
            tails_idx[pos] = k
        prev[k] = tails_idx[pos - 1] if pos else -1
    out, k = [], tails_idx[-1] if tails_idx else -1
    while k >= 0:
        out.append(points[k])
        k = prev[k]
    return out[::-1]


def map_section(hyp: list[str], t: dict) -> dict | None:
    pts = []
    for i in range(len(hyp) - K + 1):
        for j in t["index"].get(tuple(hyp[i:i + K]), ()):
            pts.append((i, j))
    if not pts:
        return None
    chain = lis_chain(pts)
    # split the chain where the text jumps far more than the audio does, keep the run
    # with the most anchors (drops stray hits in other chapters, epigraphs, etc.)
    runs, cur = [], [chain[0]]
    for a, b in zip(chain, chain[1:]):
        if (b[1] - a[1]) > JUMP_RATIO * max(1, b[0] - a[0]) + JUMP_SLACK:
            runs.append(cur)
            cur = []
        cur.append(b)
    runs.append(cur)
    chain = max(runs, key=len)
    (i0, j0), (i1, j1) = chain[0], chain[-1]
    covered_hyp = len({i + d for i, _ in chain for d in range(K)})
    return {"n_anchors": len(chain), "hyp_first": i0, "hyp_last": i1 + K - 1,
            "text_first": j0, "text_last": j1 + K - 1,
            "hyp_covered": covered_hyp, "density": round(covered_hyp / max(1, len(hyp)), 3),
            "text_span_words": j1 + K - j0}


# --------------------------------------------------------------------------- driver
def row_sha(r: dict) -> str:
    return hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()[:16]


def process_reading(args) -> dict:
    r, out_root, w2v_root, s2_root, s1_root = args
    s2p = s2_root / r["work"] / f"{r['reading_id'].replace(':', '_')}.json"
    s2 = {s["stem"]: s for s in json.loads(s2p.read_text())["sections"]} if s2p.exists() else {}
    texts = [load_text(n, s1_root) for n in r["gutenberg_ids"] if (s1_root / f"pg{n}.json.gz").exists()]
    out = {"version": VERSION, "input_sha": row_sha(r), "reading_id": r["reading_id"], "work": r["work"],
           "librivox_slug": r["librivox_slug"], "gutenberg_ids": r["gutenberg_ids"], "sections": []}
    for s in r["sections"]:
        rec = {"section": s["section"], "stem": s["stem"], "title": s["title"], "duration": s["duration"]}
        if s.get("excluded"):
            rec["excluded"] = s["excluded"]
            out["sections"].append(rec)
            continue
        if not texts:
            rec["error"] = "no source text"
            out["sections"].append(rec)
            continue
        wd = json.loads((w2v_root / r["work"] / r["librivox_slug"] / f"{s['stem']}.json").read_text())["words"]
        h2 = s2.get(s["stem"], {})
        lo = h2.get("header_last_word", 0) or 0
        hi = h2.get("footer_first_word", len(wd)) or len(wd)
        if hi <= lo:
            lo, hi = 0, len(wd)
        hyp = [re.sub(r"[^a-z']", "", w["word"].lower()).strip("'") for w in wd[lo:hi]]
        best = None
        for t in texts:
            m = map_section(hyp, t)
            if m and (best is None or m["hyp_covered"] > best[1]["hyp_covered"]):
                best = (t, m)
        rec["book_words"] = [lo, hi]
        if best is None:
            rec["error"] = "no anchors"
        else:
            t, m = best
            rec.update(m)
            rec["text"] = f"pg{t['n']}"
            rec["start"] = locate_token(t, m["text_first"])
            rec["end"] = locate_token(t, m["text_last"])
            rec["start_time"] = wd[lo + m["hyp_first"]]["start"]
            rec["end_time"] = wd[lo + m["hyp_last"]]["end"]
            rec["hyp_unanchored_before"] = m["hyp_first"]
            rec["hyp_unanchored_after"] = len(hyp) - 1 - m["hyp_last"]
        out["sections"].append(rec)
    dst = out_root / r["work"] / f"{r['reading_id'].replace(':', '_')}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False))
    tmp.replace(dst)
    return out


def sec_key(s: dict) -> tuple:
    m = re.search(r"\d+", s.get("section") or "")
    return (int(m.group()) if m else 10 ** 6, s["stem"])


def continuity(sections: list[dict]) -> None:
    """Flag gaps/overlaps/reversals between consecutive sections of one reading."""
    prev = None
    for s in sorted([x for x in sections if "start" in x], key=sec_key):
        s.setdefault("flags", [])
        if prev is not None and prev["text"] == s["text"]:
            gap = s["start"]["block"] - prev["end"]["block"]
            s["gap_blocks_from_prev"] = gap
            if gap < 0:
                s["flags"].append("overlap_or_reversed")
            elif gap > 3:
                s["flags"].append(f"gap_{gap}_blocks")
        prev = s


def write_review(rdir: Path, readings: list[dict]) -> None:
    rdir.mkdir(parents=True, exist_ok=True)
    c = lambda x: re.sub(r"[\t\r\n]+", " ", str(x if x is not None else ""))  # noqa: E731
    stats = Counter()
    head = ("work\treading_id\tsection\tstem\ttitle\ttext\tstart_div\tstart_para\tstart_heading\tend_div\tend_para\t"
            "end_heading\tdensity\tn_anchors\tunanch_before\tunanch_after\tgap_from_prev\tflags\n")
    with (rdir / "sections.tsv").open("w") as f, (rdir / "flagged.tsv").open("w") as g:
        f.write(head)
        g.write(head)
        for r in readings:
            continuity(r["sections"])
            for s in sorted(r["sections"], key=sec_key):
                if s.get("excluded"):
                    stats["excluded"] += 1
                    continue
                stats["sections"] += 1
                fl = list(s.get("flags", []))
                if "error" in s:
                    fl.append(s["error"].replace(" ", "_"))
                elif s["density"] < 0.5:
                    fl.append("low_density")
                if s.get("hyp_unanchored_before", 0) > 60:
                    fl.append("unanchored_start")
                if s.get("hyp_unanchored_after", 0) > 60:
                    fl.append("unanchored_end")
                for x in fl:
                    stats[re.sub(r"\d+", "N", x)] += 1
                st, en = s.get("start", {}), s.get("end", {})
                row = (f"{r['work']}\t{r['reading_id']}\t{s['section']}\t{s['stem']}\t{c(s['title'])[:50]}\t"
                       f"{s.get('text', '')}\t{st.get('division', '')}\t{st.get('para', '')}\t{c(st.get('heading', ''))}\t"
                       f"{en.get('division', '')}\t{en.get('para', '')}\t{c(en.get('heading', ''))}\t"
                       f"{s.get('density', '')}\t{s.get('n_anchors', '')}\t{s.get('hyp_unanchored_before', '')}\t"
                       f"{s.get('hyp_unanchored_after', '')}\t{s.get('gap_blocks_from_prev', '')}\t{','.join(fl)}\n")
                f.write(row)
                if fl:
                    g.write(row)
                if "density" in s:
                    stats["density_ge_0.8"] += s["density"] >= 0.8
    (rdir / "summary.txt").write_text("\n".join(f"{k}\t{v}" for k, v in sorted(stats.items())) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", type=Path, default=Path("/out/inventory.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("/out"))
    ap.add_argument("--work", action="append")
    ap.add_argument("--reading", action="append")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args(argv)
    (a.out / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(a.out / "logs" / "stage3.log"), logging.StreamHandler(sys.stderr)])
    out_root = a.out / "stage3"
    rows = [json.loads(x) for x in a.inventory.read_text().splitlines()]
    sel = [r for r in rows if (not a.work or r["work"] in a.work) and (not a.reading or r["librivox_slug"] in a.reading)]
    todo = []
    for r in sel:
        dst = out_root / r["work"] / f"{r['reading_id'].replace(':', '_')}.json"
        if dst.exists() and not a.force:
            d = json.loads(dst.read_text())
            if d.get("version") == VERSION and d.get("input_sha") == row_sha(r):
                continue
        todo.append(r)
    # group by text so each worker process loads a text once
    todo.sort(key=lambda r: tuple(r["gutenberg_ids"]))
    log.info("%d readings selected, %d to do", len(sel), len(todo))
    args = [(r, out_root, a.out / "w2v", a.out / "stage2", a.out / "stage1") for r in todo]
    with ProcessPoolExecutor(a.jobs) as ex:
        for d in ex.map(process_reading, args, chunksize=4):
            pass
    all_out = [json.loads(p.read_text()) for p in sorted(out_root.glob("*/*.json"))]
    write_review(a.out / "review" / "stage3", sorted(all_out, key=lambda d: (d["work"], d["reading_id"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
