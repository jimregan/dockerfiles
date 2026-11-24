"""Stage 5: word alignment of each section's wav2vec2 words to its reference.

    uv run python -m librivox_mult.stage5_align [--work W] [--reading SLUG] [--force] [--jobs N]

Reference = the Stage 2 header elements matched in the audio (the matched template
variant's words), the Stage 4 body words (Stage 3 span, chosen NSW forms), then the
footer elements. Hypothesis = every wav2vec2 word in the file.

The DP is Kaldi's ``smith_waterman_alignment`` scoring (match +1, sub/ins/del -1, same
tie-breaking), made global: it is traced back from the end of both sequences, so
reference words the reader didn't say are ``del`` and every hypothesis word is
accounted for. For speed it runs between anchors: interiors of exact-match runs of at
least ANCHOR_RUN words from a Levenshtein alignment (rapidfuzz). Chunks bigger than
MAX_CELLS fall back to the Levenshtein opcodes and are counted.

Non-``cor`` runs are kept and labelled (``label_runs``); WhisperX gives a vote per body
run and a per-word agreement used in the sentence confidence. Writes
/out/stage5/<work>/<reading_id>.json.gz.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

OUT = Path("/out")
VERSION = "stage5-0.5"
ANCHOR_RUN = 4
MAX_CELLS = 4_000_000
CONTEXT = 3  # ref words either side a repetition may echo
MIN_DENSITY = 0.2   # Stage 3 suspect_source: align headers/footers only
REORDER_BELOW = 0.9  # try block reordering when fewer ref words than this are cor

DIAG, DEL, INS = 0, 1, 2
DYS_TYPES = ("repetition", "partial", "revision", "reorder", "skip", "ins", "sub", "contraction", "orth")


def kaldi_global(ref: list[str], hyp: list[str], correct: int = 1, sub: int = -1,
                 dele: int = -1, ins: int = -1) -> list[tuple[int, int]]:
    """Global alignment with Kaldi align_ctm_ref scoring and tie-breaking (diagonal,
    then deletion if strictly better, then insertion if strictly better).
    Returns [(ref_i | -1, hyp_j | -1)] in order."""
    n, m = len(ref), len(hyp)
    if n == 0 or m == 0:
        return [(i, -1) for i in range(n)] + [(-1, j) for j in range(m)]
    prev = [j * ins for j in range(m + 1)]
    bps = [bytearray([INS]) * (m + 1)]
    for i in range(1, n + 1):
        r = ref[i - 1]
        cur = [0] * (m + 1)
        bp = bytearray(m + 1)
        cur[0] = i * dele
        bp[0] = DEL
        for j in range(1, m + 1):
            best = prev[j - 1] + (correct if r == hyp[j - 1] else sub)
            b = DIAG
            v = prev[j] + dele
            if v > best:
                best, b = v, DEL
            v = cur[j - 1] + ins
            if v > best:
                best, b = v, INS
            cur[j] = best
            bp[j] = b
        bps.append(bp)
        prev = cur
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        b = bps[i][j] if i > 0 else INS
        if j == 0:
            b = DEL
        if b == DIAG:
            i, j = i - 1, j - 1
            out.append((i, j))
        elif b == DEL:
            i -= 1
            out.append((i, -1))
        else:
            j -= 1
            out.append((-1, j))
    out.reverse()
    return out


def anchors(ref: list[str], hyp: list[str], run: int = ANCHOR_RUN) -> list[tuple[int, int]]:
    """(ref_i, hyp_j) pairs inside exact-match runs of >= ``run`` words, minus each
    run's edge words so the DP decides repetitions next to a run."""
    from rapidfuzz.distance import Levenshtein

    out = []
    for op in Levenshtein.opcodes(ref, hyp):
        if op.tag == "equal" and op.src_end - op.src_start >= run:
            for d in range(1, op.src_end - op.src_start - 1):
                out.append((op.src_start + d, op.dest_start + d))
    return out


def align(ref: list[str], hyp: list[str]) -> tuple[list[tuple[int, int]], int]:
    """Anchored global alignment; returns (pairs, number of chunks too big for the DP)."""
    from rapidfuzz.distance import Levenshtein

    pairs: list[tuple[int, int]] = []
    big = 0
    pi = pj = 0
    for ai, aj in anchors(ref, hyp) + [(len(ref), len(hyp))]:
        if ai > pi or aj > pj:
            r, h = ref[pi:ai], hyp[pj:aj]
            if len(r) * len(h) <= MAX_CELLS:
                chunk = kaldi_global(r, h)
            else:
                big += 1
                chunk = []
                for op in Levenshtein.opcodes(r, h):
                    if op.tag in ("equal", "replace"):
                        k = min(op.src_end - op.src_start, op.dest_end - op.dest_start)
                        chunk += [(op.src_start + d, op.dest_start + d) for d in range(k)]
                        chunk += [(op.src_start + d, -1) for d in range(k, op.src_end - op.src_start)]
                        chunk += [(-1, op.dest_start + d) for d in range(k, op.dest_end - op.dest_start)]
                    elif op.tag == "delete":
                        chunk += [(op.src_start + d, -1) for d in range(op.src_end - op.src_start)]
                    else:
                        chunk += [(-1, op.dest_start + d) for d in range(op.dest_end - op.dest_start)]
            pairs += [(i + pi if i >= 0 else -1, j + pj if j >= 0 else -1) for i, j in chunk]
        if ai < len(ref):
            pairs.append((ai, aj))
        pi, pj = ai + 1, aj + 1
    return pairs, big


def letters(ws) -> str:
    return "".join(ws).replace("'", "")


def edit_kind(ref: list[str], hyp: list[str], i: int, j: int) -> str:
    if i < 0:
        return "ins"
    if j < 0:
        return "del"
    return "cor" if ref[i] == hyp[j] else "sub"


CONTRACTIONS = {"'s": [["is"], ["has"], ["us"]], "'ll": [["will"], ["shall"]], "'re": [["are"]],
                "'ve": [["have"]], "'d": [["would"], ["had"]], "'m": [["am"]], "n't": [["not"]]}
IRREGULAR = {"won't": ["will", "not"], "can't": ["can", "not"], "shan't": ["shall", "not"]}


def expansions(word: str) -> list[list[str]]:
    if word in IRREGULAR:
        return [IRREGULAR[word]]
    for suf, alts in CONTRACTIONS.items():
        if word.endswith(suf) and len(word) > len(suf):
            return [[word[: -len(suf)]] + a for a in alts]
    return []


def contraction(rw: list[str], hw: list[str]) -> bool:
    """One side says a contraction where the other has it in full ("it's" / "it is")."""
    def full(ws):
        out = [[]]
        for w in ws:
            out = [o + e for o in out for e in (expansions(w) or [[w]])]
        return {tuple(o) for o in out}
    return rw != hw and bool(full(rw) & full(hw))


def label_run(ref: list[str], hyp: list[str], run: list[tuple[int, int]],
              before: list[int], after: list[int], ref_kind: dict[int, str]) -> str:
    """Label one maximal non-cor run of a body region.

    before/after: indices of the ref words around the run (up to CONTEXT each side).
    ref_kind: ref index -> 'heading' | 'marker' for words not expected to be read.
    """
    r = [i for i, _ in run if i >= 0]
    h = [j for _, j in run if j >= 0]
    if r and h and letters(ref[i] for i in r) == letters(hyp[j] for j in h):
        return "orth"
    if not h:
        kinds = {ref_kind.get(i) for i in r}
        if kinds == {"marker"}:
            return "unread_marker"
        if kinds == {"heading"}:
            return "unread_heading"
        return "skip"
    if r and h and contraction([ref[i] for i in r], [hyp[j] for j in h]):
        return "contraction"
    ins = [hyp[j] for i, j in run if i < 0]
    if not ins:
        return "sub"
    pre = [ref[i] for i in before]
    post = [ref[i] for i in after]
    whole = len(ins) == len(run)
    if whole and (ins == post[:len(ins)] or (len(ins) <= len(pre) and ins == pre[len(pre) - len(ins):])):
        return "repetition"
    # a restart: the inserted material begins with what is said next
    k = 0
    while k < len(ins) and k < len(post) and ins[k] == post[k]:
        k += 1
    if k < len(ins) and k < len(post) and len(ins[k]) < len(post[k]) and post[k].startswith(ins[k]):
        rest = ins[k + 1:]
        return "partial" if whole and all(w == ins[k] for w in rest) else "revision"
    if k and whole:
        return "revision"
    return "ins" if whole else "sub"


REORDER_WORDS = 6


def mark_reorders(spans: list[dict], ref: list[str], hyp: list[str], reach: int = 2) -> None:
    """An inserted word that is deleted from the ref close by was moved, not added:
    relabel both runs ``reorder`` ("he always seemed" read "he seemed always")."""
    for a, sa in enumerate(spans):
        if sa["region"] != "body" or sa["type"] not in ("ins", "sub") or not sa["hyp"]:
            continue
        said = {hyp[j] for j in sa["hyp"]}
        for sb in spans[max(0, a - reach):a + reach + 1]:
            if sb is sa or sb["region"] != "body" or sb["type"] not in ("skip", "sub"):
                continue
            if abs(sb["pairs"][0] - sa["pairs"][0]) > REORDER_WORDS:
                continue
            if sb["type"] == "skip" and said & {ref[i] for i in sb["ref"]}:
                sa["type"] = sb["type"] = "reorder"
                break


def runs(pairs: list[tuple[int, int]], ref: list[str], hyp: list[str]) -> list[tuple[int, int]]:
    """[(p0, p1)) index ranges over ``pairs`` of maximal non-cor runs."""
    out, p0 = [], None
    for p, (i, j) in enumerate(pairs):
        bad = edit_kind(ref, hyp, i, j) != "cor"
        if bad and p0 is None:
            p0 = p
        elif not bad and p0 is not None:
            out.append((p0, p))
            p0 = None
    if p0 is not None:
        out.append((p0, len(pairs)))
    return out


def norm_word(w: str) -> str:
    return re.sub(r"[^a-z']", "", w.lower()).strip("'")


def whisper_words(path: str | None) -> list[dict] | None:
    if not path or not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text())
    ws = []
    for w in d.get("word_segments") or [x for s in d.get("segments", []) for x in s.get("words", [])]:
        if "start" in w and "end" in w:
            for part in w["word"].replace("-", " ").split():
                if norm_word(part):
                    ws.append({"w": norm_word(part), "start": w["start"], "end": w["end"]})
    return ws


def whisper_vote(wx: list[dict], t0: float, t1: float, rw: list[str], hw: list[str]) -> str:
    """Which side WhisperX's words in [t0, t1] agree with: 'ref' (the book: it smoothed
    over what wav2vec2 heard), 'w2v', or 'other'."""
    got = letters(w["w"] for w in wx if t0 - 0.05 <= (w["start"] + w["end"]) / 2 <= t1 + 0.05)
    if got == letters(rw):
        return "ref"
    if got == letters(hw):
        return "w2v"
    return "other"


def load_stage1(tid: str, cache: dict) -> dict:
    if tid not in cache:
        d = json.load(gzip.open(OUT / "stage1" / f"{tid}.json.gz", "rt"))
        tk = d["tokens"]
        kinds = [b["kind"] for b in d["blocks"]]
        cache[tid] = {"surface": tk["surface"], "sent": tk["sent"], "block": tk["block"],
                      "block_kind": kinds, "sentences": d["sentences"]}
    return cache[tid]


def build_ref(s2sec: dict, s4sec: dict) -> tuple[list[list], list[dict]]:
    """Reference rows [word, token, nsw, element] (element -1 for body) and the header/
    footer elements [{id, kind, variant, source, w0, w1}] in audio order."""
    tmpl = s2sec.get("templates", [])
    els, rows = [], []
    ms = sorted(s2sec.get("matches", []), key=lambda m: m["w0"])
    heads = [m for m in ms if m["role"] == "header"]
    feet = [m for m in ms if m["role"] == "footer"]
    for role, group in (("H", heads), ("F", feet)):
        for n, m in enumerate(group, 1):
            v = tmpl[m["element"]]["variants"][m["variant"]]
            els.append({"id": f"{role}{n:02d}", "kind": m["kind"], "element": m["element"],
                        "variant": m["variant"], "text": v["text"], "source": v["source"],
                        "score": m["score"], "hyp": [m["w0"], m["w1"]]})
    nh = len(heads)
    for e, el in enumerate(els[:nh]):
        rows += [[w, -1, -1, e] for w in el["text"].split()]
    rows += [[w, t, k, -1] for w, t, k in s4sec["ref"]]
    for e, el in enumerate(els[nh:], nh):
        rows += [[w, -1, -1, e] for w in el["text"].split()]
    return rows, els


def align_section(rows: list[list], els: list[dict], wd: list[dict], s1: dict | None,
                  wx: list[dict] | None) -> dict:
    ref = [r[0] for r in rows]
    hyp = [norm_word(w["word"]) for w in wd]
    pairs, big = align(ref, hyp)

    ref_kind = {}
    if s1:
        S, bk, kinds = s1["surface"], s1["block"], s1["block_kind"]
        for i, (_, t, _, e) in enumerate(rows):
            if e >= 0:
                continue
            if kinds[bk[t]] in ("heading", "toc"):
                ref_kind[i] = "heading"
            elif 0 < t < len(S) - 1 and S[t - 1] in "[(" and S[t + 1] in "])" and S[t].isdigit():
                ref_kind[i] = "marker"

    def time_of(p: int, side: int) -> float:
        """A time for pairs[p] even when it has no hyp word: the nearest one beside it."""
        rng = range(p, -1, -1) if side < 0 else range(p, len(pairs))
        for q in rng:
            j = pairs[q][1]
            if j >= 0:
                return wd[j]["end"] if side < 0 else wd[j]["start"]
        return 0.0 if side < 0 else (wd[-1]["end"] if wd else 0.0)

    ref_pos = [p for p, (i, _) in enumerate(pairs) if i >= 0]
    body = [p for p in ref_pos if rows[pairs[p][0]][3] < 0]
    b0, b1 = (body[0], body[-1]) if body else (len(pairs), -1)

    spans = []
    for p0, p1 in runs(pairs, ref, hyp):
        run = pairs[p0:p1]
        r = [i for i, _ in run if i >= 0]
        h = [j for _, j in run if j >= 0]
        if not (any(rows[i][3] < 0 for i in r) or (not r and b0 < p0 and p1 <= b1)):
            region = "header" if p0 < b0 else "footer"
            label = "template" if not h else ("extra" if not r else "template_sub")
        else:
            region = "body"
            ri = [i for i, _ in pairs[:p0] if i >= 0][-CONTEXT:]
            ra = [i for i, _ in pairs[p1:p1 + 4 * CONTEXT] if i >= 0][:CONTEXT]
            label = label_run(ref, hyp, run, ri, ra, ref_kind)
        sp = {"type": label, "region": region, "pairs": [p0, p1], "ref": r, "hyp": h,
              "w2v": " ".join(hyp[j] for j in h),
              "start": round(time_of(p0 - 1, -1) if not h else wd[h[0]]["start"], 3),
              "end": round(time_of(p1, +1) if not h else wd[h[-1]]["end"], 3)}
        if region == "body" and wx is not None and label not in ("unread_marker", "unread_heading"):
            sp["whisperx"] = whisper_vote(wx, sp["start"], sp["end"], [ref[i] for i in r], [hyp[j] for j in h])
        spans.append(sp)

    mark_reorders(spans, ref, hyp)

    wx_eq = set()
    if wx is not None:
        from rapidfuzz.distance import Levenshtein

        bi = [i for i, row in enumerate(rows) if row[3] < 0]
        for op in Levenshtein.opcodes([ref[i] for i in bi], [w["w"] for w in wx]):
            if op.tag == "equal":
                wx_eq.update(bi[op.src_start + d] for d in range(op.src_end - op.src_start))

    words = []
    for i, j in pairs:
        k = edit_kind(ref, hyp, i, j)
        words.append([i, j, k] + ([wd[j]["start"], wd[j]["end"], wd[j]["score"]] if j >= 0 else [None, None, None]))

    units: dict = {}
    for i, j in pairs:
        if i < 0:
            continue
        e = rows[i][3]
        key = els[e]["id"] if e >= 0 else (s1["sent"][rows[i][1]] if s1 else -1)
        u = units.setdefault(key, {"n": 0, "cor": 0, "wx": 0, "score": 0.0})
        u["n"] += 1
        if j >= 0 and ref[i] == hyp[j]:
            u["cor"] += 1
            u["score"] += wd[j]["score"]
        if i in wx_eq:
            u["wx"] += 1
    unit_stats = []
    for key, u in units.items():
        voted = wx is not None and not isinstance(key, str)  # no WhisperX vote on headers
        conf = (u["cor"] + (u["wx"] if voted else u["cor"])) / (2 * u["n"])
        conf *= u["score"] / u["cor"] if u["cor"] else 0.0
        unit_stats.append({"unit": key, "n": u["n"], "cor": u["cor"],
                           "whisperx_eq": u["wx"] if voted else None,
                           "mean_score": round(u["score"] / u["cor"], 4) if u["cor"] else None,
                           "confidence": round(conf, 4)})

    counts: dict = {}
    for k in ("cor", "sub", "ins", "del"):
        counts[k] = sum(1 for w in words if w[2] == k)
    return {"words": words, "spans": spans, "units": unit_stats, "counts": counts,
            "big_chunks": big, "ref": rows, "elements": els}


def reorder_blocks(rows: list[list], hyp: list[str], block_of_tok: list[int], k: int = 4) -> list[int] | None:
    """A row order with body blocks sorted by where they were read (median hyp position
    of their 4-grams that occur at most twice in the hyp). Blocks without hits follow
    their text-order predecessor. None if the order doesn't change."""
    pos: dict = {}
    for j in range(len(hyp) - k + 1):
        pos.setdefault(tuple(hyp[j:j + k]), []).append(j)
    blocks: list[list[int]] = []
    last = None
    for i, row in enumerate(rows):
        if row[3] >= 0:
            continue
        b = block_of_tok[row[1]]
        if b != last:
            blocks.append([])
            last = b
        blocks[-1].append(i)
    keys = []
    prev = -1.0
    for n, idx in enumerate(blocks):
        ws = [rows[i][0] for i in idx]
        hits = sorted(p for g in range(len(ws) - k + 1) for p in pos.get(tuple(ws[g:g + k]), [])
                      if len(pos[tuple(ws[g:g + k])]) <= 2)
        key = float(hits[len(hits) // 2]) if len(hits) >= 2 else prev + 1e-3
        keys.append((key, n))
        prev = key
    order = [n for _, n in sorted(keys)]
    if order == sorted(order):
        return None
    head = [i for i, row in enumerate(rows) if row[3] >= 0 and i < (blocks[0][0] if blocks else len(rows))]
    tail = [i for i, row in enumerate(rows) if row[3] >= 0 and i not in head]
    return head + [i for n in order for i in blocks[n]] + tail


def section_inputs(r: dict) -> tuple[Path, Path, Path]:
    return (OUT / "stage2" / r["work"] / f"{r['reading_id']}.json",
            OUT / "stage3" / r["work"] / f"{r['reading_id']}.json",
            OUT / "stage4" / r["work"] / f"{r['reading_id']}.json")


def out_of_order(s3sec: dict) -> bool:
    sp = sorted(s3sec.get("spans", []), key=lambda x: x["hyp_first"])
    return any(a["text_first"] > b["text_first"] for a, b in zip(sp, sp[1:]))


def process_reading(args: tuple[dict, bool]) -> dict:
    r, force = args
    p2, p3, p4 = section_inputs(r)
    dst = OUT / "stage5" / r["work"] / f"{r['reading_id']}.json.gz"
    if not (p2.exists() and p3.exists() and p4.exists()):
        return {"reading_id": r["reading_id"], "status": "no_input"}
    key = hashlib.sha256(b"".join(p.read_bytes() for p in (p2, p3, p4)) + VERSION.encode()).hexdigest()[:16]
    if dst.exists() and not force:
        try:
            if json.load(gzip.open(dst, "rt")).get("input_sha") == key:
                return {"reading_id": r["reading_id"], "status": "done"}
        except (OSError, json.JSONDecodeError):
            pass
    s2 = {s["stem"]: s for s in json.loads(p2.read_text())["sections"]}
    s3 = {s["stem"]: s for s in json.loads(p3.read_text())["sections"]}
    s4 = json.loads(p4.read_text())["sections"]
    inv = {s["file"].rsplit(".", 1)[0]: s for s in r["sections"]}
    cache: dict = {}
    out = {"version": VERSION, "input_sha": key, "reading_id": r["reading_id"], "work": r["work"],
           "sections": [], "skipped": []}
    for sec in s4:
        stem = sec["stem"]
        wd = json.loads((OUT / "w2v" / r["work"] / r["librivox_slug"] / f"{stem}.json").read_text())["words"]
        s3sec = s3.get(stem, {})
        low = s3sec.get("density", 1.0) < MIN_DENSITY
        rows, els = build_ref(s2.get(stem, {}), {**sec, "ref": []} if low else sec)
        s1 = load_stage1(sec["text"], cache)
        wxp = inv.get(stem, {}).get("asr", {}).get("whisperx", {})
        wx = whisper_words(wxp.get("path") if wxp.get("present") else None)
        a = align_section(rows, els, wd, s1, wx)
        notes = {"body_dropped": "low_density"} if low else {}
        if not low and (a["counts"]["cor"] < REORDER_BELOW * len(rows) or out_of_order(s3sec)):
            order = reorder_blocks(rows, [norm_word(w["word"]) for w in wd], s1["block"])
            if order is not None:
                b = align_section([rows[i] for i in order], els, wd, s1, wx)
                if b["counts"]["cor"] > a["counts"]["cor"]:
                    a, notes = b, {"reordered": True}
        out["sections"].append({"stem": stem, "text": sec["text"], "blocks": sec["blocks"], **notes, **a})
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(dst)
    return {"reading_id": r["reading_id"], "status": "aligned", "n": len(out["sections"])}


def write_review(out: Path) -> None:
    """summary.tsv per section; spans_sample.tsv: up to 40 spans per type with their words."""
    import random
    from collections import Counter

    rdir = out / "review" / "stage5"
    rdir.mkdir(parents=True, exist_ok=True)
    tot, wxv, rows, skipped, readings = Counter(), Counter(), [], [], {}
    sample: dict = {}
    rng = random.Random(5)
    for p in sorted(out.glob("stage5/*/*.json.gz")):
        d = json.load(gzip.open(p, "rt"))
        skipped += [(d["reading_id"], s["stem"], s["reason"]) for s in d["skipped"]]
        for sec in d["sections"]:
            c = sec["counts"]
            ty = Counter(s["type"] for s in sec["spans"])
            tot.update(ty)
            wxv.update((s["type"], s.get("whisperx")) for s in sec["spans"] if s["region"] == "body")
            body = sum(1 for row in sec["ref"] if row[3] < 0)
            n_ref = c["cor"] + c["sub"] + c["del"]
            rows.append([d["work"], d["reading_id"], sec["stem"], sec["text"], n_ref, body,
                         round(c["cor"] / max(1, n_ref), 3), c["sub"], c["ins"], c["del"], sec["big_chunks"],
                         " ".join(f"{k}:{v}" for k, v in sorted(ty.items()))])
            ref = [row[0] for row in sec["ref"]]
            at = -1
            ref_at = []
            for w in sec["words"]:
                at = w[0] if w[0] >= 0 else at
                ref_at.append(at)
            for s in sec["spans"]:
                s["at"] = s["ref"][0] if s["ref"] else ref_at[s["pairs"][0]] + 1
                sample.setdefault(s["type"], []).append((d["reading_id"], sec["stem"], s, ref))
            dys = sum(ty[k] for k in ("repetition", "partial", "revision"))
            per_reading = readings.setdefault(d["reading_id"], [d["work"], 0, 0, Counter()])
            per_reading[1] += body
            per_reading[2] += dys
            per_reading[3].update({k: v for k, v in ty.items() if k in DYS_TYPES})
    with (rdir / "sections.tsv").open("w") as f:
        f.write("work\treading_id\tstem\ttext\tn_ref\tn_body\tcor_rate\tsub\tins\tdel\tbig_chunks\tspans\n")
        for row in sorted(rows, key=lambda x: x[6]):
            f.write("\t".join(map(str, row)) + "\n")
    with (rdir / "spans_sample.tsv").open("w") as f:
        f.write("type\tregion\twhisperx\treading_id\tstem\tstart\tend\tref_context\tref\tw2v\n")
        for t, xs in sorted(sample.items()):
            for rid, stem, s, ref in rng.sample(xs, min(40, len(xs))):
                lo = max(0, s["at"] - 4)
                f.write("\t".join(map(str, [t, s["region"], s.get("whisperx"), rid, stem, s["start"], s["end"],
                                            " ".join(ref[lo:lo + 10]), " ".join(ref[i] for i in s["ref"]),
                                            s["w2v"]])) + "\n")
    with (rdir / "readings.tsv").open("w") as f:
        f.write("work\treading_id\tbody_words\tdysfl_per_1k\t" + "\t".join(DYS_TYPES) + "\n")
        for rid, (work, n, dys, c) in sorted(readings.items(), key=lambda kv: -kv[1][2] / max(1, kv[1][1])):
            f.write("\t".join(map(str, [work, rid, n, round(1000 * dys / max(1, n), 2)] + [c[k] for k in DYS_TYPES])) + "\n")
    with (rdir / "summary.txt").open("w") as f:
        f.write(f"sections aligned: {len(rows)}; skipped: {len(skipped)}\n")
        f.write("spans by type: " + ", ".join(f"{k} {v}" for k, v in tot.most_common()) + "\n")
        f.write("whisperx vote (body): " + ", ".join(f"{k[0]}/{k[1]} {v}" for k, v in sorted(wxv.items(), key=str)) + "\n")
        for s in skipped:
            f.write("skipped\t" + "\t".join(s) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", action="append")
    ap.add_argument("--reading", action="append")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--no-review", action="store_true")
    a = ap.parse_args(argv)
    inv = [json.loads(x) for x in (OUT / "inventory.jsonl").read_text().splitlines()]
    todo = [(r, a.force) for r in inv
            if not (a.work and r["work"] not in a.work)
            and not (a.reading and r["librivox_slug"] not in a.reading and r["reading_id"] not in a.reading)]
    from collections import Counter

    st = Counter()
    with Pool(a.jobs) as pool:
        for res in pool.imap_unordered(process_reading, todo):
            st[res["status"]] += 1
            if res["status"] == "aligned":
                print(res["reading_id"], res["n"], flush=True)
    print(dict(st))
    if not a.no_review:
        write_review(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
