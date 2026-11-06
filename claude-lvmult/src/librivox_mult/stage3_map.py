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

VERSION = "stage3-0.4"
K = 4              # n-gram length for anchors
MAX_OCC = 8        # ignore n-grams occurring more often than this in the text
JUMP_RATIO = 5     # a text jump > 5x the audio gap (+ slack) splits the anchor chain
JUMP_SLACK = 300   # words
MIN_RUN = 10       # anchors for a run to count as its own span
log = logging.getLogger("stage3")


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


def block_words(t: dict, bi: int) -> int:
    d = t["doc"]
    b = d["blocks"][bi]
    return len(spoken_words(d["text"][b["char_start"]:b["char_end"]]))


def block_edge(t: dict, bi: int, last: bool) -> dict:
    """Location of the first/last word of block ``bi`` (same fields as locate_token)."""
    ws = [w for w, ti in enumerate(t["tok_of"]) if t["doc"]["tokens"]["block"][ti] == bi]
    return locate_token(t, ws[-1] if last else ws[0])


def snap_edges(secs: list[dict], tmap: dict) -> None:
    """Assign readable blocks between consecutive sections to a side.

    Anchors need 4 matching words, so short paragraphs at section edges ("Stas' head
    swam.") are left between sections. A gap of W words goes to the previous section
    if its unanchored tail could hold it, else to the next section if its unanchored
    head could, else split; only text neither side can account for is a real gap.
    """
    order = sorted([x for x in secs if "start" in x and x.get("n_spans", 1) == 1], key=sec_key)
    for a, b in zip(order, order[1:]):
        if a["text"] != b["text"] or b["start"]["block"] <= a["end"]["block"] + 1:
            continue
        t = tmap[a["text"]]
        kinds = [x["kind"] for x in t["doc"]["blocks"]]
        gap = [i for i in range(a["end"]["block"] + 1, b["start"]["block"]) if kinds[i] in READABLE]
        if not gap:
            continue
        w = [block_words(t, i) for i in gap]
        ua, ub = a.get("hyp_unanchored_after", 0), b.get("hyp_unanchored_before", 0)
        k = 0  # blocks [0, k) go to a, [k, n) to b
        room = ua
        while k < len(gap) and w[k] <= room / 0.6:
            room -= w[k]
            k += 1
        rest = sum(w[k:])
        if rest and ub < 0.6 * rest:
            b["gap_unaccounted_words"] = rest  # reader skipped text, or text missing from audio
            b.setdefault("flags", []).append(f"gap_{len(gap) - k}_blocks_{rest}_words")
            k_b = len(gap)
        else:
            k_b = k
        if k:
            a["end"] = block_edge(t, gap[k - 1], last=True)
            a["snapped_end_blocks"] = k
        if k_b == k and k < len(gap):
            b["start"] = block_edge(t, gap[k], last=False)
            b["snapped_start_blocks"] = len(gap) - k


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
    # Usually one run. Several substantial runs = the section reads pieces in a different
    # order from the text (fable collections); each becomes its own span.
    big = max(len(x) for x in runs)
    keep = [x for x in runs if len(x) >= max(MIN_RUN, 0.1 * big)] or [max(runs, key=len)]
    spans = [{"hyp_first": x[0][0], "hyp_last": x[-1][0] + K - 1,
              "text_first": x[0][1], "text_last": x[-1][1] + K - 1, "n_anchors": len(x)} for x in keep]
    main = max(keep, key=len)
    covered_hyp = len({i + d for x in keep for i, _ in x for d in range(K)})
    return {"n_anchors": sum(len(x) for x in keep), "n_spans": len(spans), "spans": spans,
            "hyp_first": min(sp["hyp_first"] for sp in spans), "hyp_last": max(sp["hyp_last"] for sp in spans),
            "text_first": main[0][1], "text_last": main[-1][1] + K - 1,
            "hyp_covered": covered_hyp, "density": round(covered_hyp / max(1, len(hyp)), 3)}


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
            for sp in m["spans"]:
                sp["start"] = locate_token(t, sp["text_first"])
                sp["end"] = locate_token(t, sp["text_last"])
            rec["start_time"] = wd[lo + m["hyp_first"]]["start"]
            rec["end_time"] = wd[lo + m["hyp_last"]]["end"]
            rec["hyp_unanchored_before"] = m["hyp_first"]
            rec["hyp_unanchored_after"] = len(hyp) - 1 - m["hyp_last"]
        out["sections"].append(rec)
    snap_edges(out["sections"], {f"pg{t['n']}": t for t in texts})
    dst = out_root / r["work"] / f"{r['reading_id'].replace(':', '_')}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False))
    tmp.replace(dst)
    return out


def sec_key(s: dict) -> tuple:
    m = re.search(r"\d+", s.get("section") or "")
    return (int(m.group()) if m else 10 ** 6, s["stem"])


READABLE = ("para", "verse", "lines")


def continuity(sections: list[dict], kinds: dict[str, list[str]]) -> None:
    """Flag gaps/overlaps/reversals between consecutive sections of one reading.

    A gap counts only readable blocks (para/verse/lines) between the previous
    section's end and this one's start: captions, contents and footnotes that
    nobody reads are not a gap.
    """
    prev = None
    for s in sorted([x for x in sections if "start" in x and s_single(x)], key=sec_key):
        s.setdefault("flags", [])
        if prev is not None and prev["text"] == s["text"]:
            a, b = prev["end"]["block"], s["start"]["block"]
            if b <= a:
                s["gap_blocks_from_prev"] = b - a
                s["flags"].append("overlap_or_reversed")
            else:
                k = kinds[s["text"]]
                s["gap_blocks_from_prev"] = sum(1 for i in range(a + 1, b) if k[i] in READABLE)
        prev = s


def s_single(s: dict) -> bool:
    """Sections read out of text order (several spans) are excluded from continuity."""
    return s.get("n_spans", 1) == 1


def write_review(rdir: Path, readings: list[dict], s1_root: Path) -> None:
    rdir.mkdir(parents=True, exist_ok=True)
    kinds = {}
    for n in sorted({g for r in readings for g in r["gutenberg_ids"]}):
        p = s1_root / f"pg{n}.json.gz"
        if p.exists():
            kinds[f"pg{n}"] = [b["kind"] for b in json.load(gzip.open(p, "rt"))["blocks"]]
    c = lambda x: re.sub(r"[\t\r\n]+", " ", str(x if x is not None else ""))  # noqa: E731
    stats = Counter()
    head = ("work\treading_id\tsection\tstem\ttitle\ttext\tstart_div\tstart_para\tstart_heading\tend_div\tend_para\t"
            "end_heading\tdensity\tn_anchors\tunanch_before\tunanch_after\tgap_from_prev\tflags\n")
    with (rdir / "sections.tsv").open("w") as f, (rdir / "flagged.tsv").open("w") as g:
        f.write(head)
        g.write(head)
        for r in readings:
            dens = sorted(s["density"] for s in r["sections"] if "density" in s)
            # most sections barely anchor: the listed source is probably the wrong book
            r["suspect_source"] = bool(dens) and dens[len(dens) // 2] < 0.2
            stats["readings_suspect_source"] += r["suspect_source"]
            continuity(r["sections"], kinds)
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
                if s.get("n_spans", 1) > 1:
                    fl.append(f"out_of_order_{s['n_spans']}_spans")
                if r.get("suspect_source"):
                    fl.append("suspect_source")
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
    # drop outputs of readings that no longer exist (reading ids change when e.g. a
    # multi-version project is split), so review tables don't mix old and new
    live = {(r["work"], r["reading_id"].replace(":", "_")) for r in rows}
    for p in out_root.glob("*/*.json"):
        if (p.parent.name, p.stem) not in live:
            p.unlink()
    all_out = [json.loads(p.read_text()) for p in sorted(out_root.glob("*/*.json"))]
    write_review(a.out / "review" / "stage3", sorted(all_out, key=lambda d: (d["work"], d["reading_id"])),
                 a.out / "stage1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
