"""Stage 4: normalise source texts to the wav2vec2 alphabet, with NeMo for NSWs.

    ssh worker 'cd /work && PYTHONPATH=src /opt/venv/bin/python -m librivox_mult.stage4_norm text [--text pg103]'
    uv run python -m librivox_mult.stage4_norm select [--work W] [--reading SLUG] [--force]

``text`` (worker, needs nemo_text_processing): per Stage 1 text, every token gets
its spoken words in the wav2vec2 alphabet (lower-case a-z and '). Non-standard words
(numbers, currency, times, roman numerals, abbreviations, symbols) become NSW spans
with NeMo candidates: ``NormalizerWithAudio`` gives the candidate set,
``Normalizer`` the default. Writes /out/stage4/text/<text_id>.json.gz.

``select`` (CPU): per section, the reference (the Stage 3 span, whole blocks) with
each NSW's default form is aligned to the wav2vec2 words; the NSW's audio window
lies between its matched neighbours, every candidate is scored by character error
rate against that window, and the best is chosen. Writes
/out/stage4/<work>/<reading_id>.json with, per section, the reference words (word,
token, nsw) and the NSW choices (candidates, choice, score, window).

A chosen form maps many words to one source span ("80" -> "eighty"): every
reference word keeps the Stage 1 token it came from, and each NSW its char span.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

OUT = Path("/out")
TEXT_VERSION = "stage4-0.1"  # the NeMo step (worker); outputs keep this unless it changes
VERSION = "stage4-0.4"       # the selection step
LIGATURES = str.maketrans({"æ": "ae", "œ": "oe", "ß": "ss", "ø": "o", "ð": "d", "þ": "th", "ł": "l",
                           "’": "'", "‘": "'"})
# abbreviations followed by "."; the ones in NEEDS_NUMBER only before a number
ABBREV = {"Mr", "Mrs", "Messrs", "Dr", "St", "Mt", "Jr", "Sr", "Capt", "Col", "Gen", "Lieut", "Rev",
          "Prof", "Hon", "Esq", "Co", "Bros", "viz", "etc", "ft", "oz", "lb", "lbs", "yd", "yds",
          "No", "Nos", "vol", "Vol", "ch", "Ch", "p", "pp", "c"}
NEEDS_NUMBER = {"No", "Nos", "vol", "Vol", "ch", "Ch", "p", "pp", "ft", "oz", "lb", "lbs", "yd", "yds"}
SYMBOLS = set("£$&%½¼¾§°")
JOINERS = set(",.:/–-")  # inside numbers: 20,000  8.45  12:30  1/2  1848-53
ROMAN = re.compile(r"^[IVXLCDM]{2,}$")


def plain_words(surface: str) -> list[str]:
    """A non-NSW token's spoken words: lower-case, diacritics folded, hyphens split."""
    t = unicodedata.normalize("NFKD", surface.lower().translate(LIGATURES))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return [w.strip("'") for w in re.split(r"[^a-z']+", t) if w.strip("'")]


def find_nsw(doc: dict) -> list[dict]:
    """NSW spans over Stage 1 tokens: [{tok_start, tok_end (excl.), kind, query, context}]."""
    tk, T = doc["tokens"], doc["text"]
    S, A, Z, P = tk["surface"], tk["start"], tk["end"], tk["punct"]
    n = len(S)
    out, i = [], 0

    def adjacent(j):  # token j+1 starts where j ends (no space)
        return j + 1 < n and A[j + 1] == Z[j]

    while i < n:
        s = S[i]
        if s == "&" and adjacent(i) and S[i + 1] == "c" and adjacent(i + 1) and S[i + 2:i + 3] == ["."]:
            out.append({"tok_start": i, "tok_end": i + 3, "kind": "abbrev"})  # &c. = et cetera
            i += 3
            continue
        if re.search(r"\d", s) or (s in SYMBOLS and adjacent(i) and re.search(r"\d", S[i + 1])):
            j = i
            # extend over digit groups and joiners with no spaces: 20,000 / 8.45 / £20,000 / 12:30
            while adjacent(j) and (S[j + 1] in JOINERS and adjacent(j + 1) and re.search(r"\d", S[j + 2])
                                   or re.search(r"\d", S[j + 1]) or S[j] in SYMBOLS):
                j += 2 if S[j + 1] in JOINERS else 1
            if adjacent(j) and S[j + 1] in ("%", "°"):
                j += 1
            out.append({"tok_start": i, "tok_end": j + 1, "kind": "number"})
            i = j + 1
            continue
        if s in SYMBOLS:
            out.append({"tok_start": i, "tok_end": i + 1, "kind": "symbol"})
        elif not P[i] and adjacent(i) and S[i + 1] == "." and s in ABBREV:
            nxt = S[i + 2] if i + 2 < n else ""
            if s not in NEEDS_NUMBER or re.match(r"\d", nxt):
                end = i + 2
                if s in NEEDS_NUMBER:  # "No. 7" is one span: "number seven"
                    end = i + 3
                    while end < n and A[end] == Z[end - 1] and (re.search(r"\d", S[end]) or S[end] in JOINERS):
                        end += 1
                out.append({"tok_start": i, "tok_end": end, "kind": "abbrev"})
                i = end
                continue
        elif not P[i] and ROMAN.match(s):
            out.append({"tok_start": i, "tok_end": i + 1, "kind": "roman"})
        i += 1
    for k, x in enumerate(out):
        a, z = A[x["tok_start"]], Z[x["tok_end"] - 1]
        x.update({"id": k, "char_start": a, "char_end": z, "surface": T[a:z]})
        # previous word as context: roman numerals ("Henry VIII" vs "Chapter XII") and
        # ordinals/abbreviations read differently depending on what precedes them
        prev = next((S[j] for j in range(x["tok_start"] - 1, max(-1, x["tok_start"] - 3), -1) if not P[j]), "")
        x["context"] = prev if x["kind"] == "roman" else ""
        x["query"] = f"{prev} {x['surface']}" if x["context"] else x["surface"]
    return out


def to_alphabet(s: str) -> str:
    return " ".join(w for w in plain_words(s))


def strip_context(cand: str, context: str) -> str:
    """Drop the normalised context word(s) from the front of a candidate."""
    c, ctx = cand.split(), to_alphabet(context).split()
    return " ".join(c[len(ctx):]) if ctx and c[:len(ctx)] == ctx else cand


def cmd_text(a) -> int:
    from nemo_text_processing.text_normalization.normalize import Normalizer
    from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

    cache = a.out / "cache" / "nemo_tn"
    det = Normalizer(input_case="cased", lang="en", cache_dir=str(cache))
    aud = NormalizerWithAudio(input_case="cased", lang="en", cache_dir=str(cache))
    memo_p = a.out / "stage4" / "nsw_cache.json"
    memo = json.loads(memo_p.read_text()) if memo_p.exists() else {}
    tids = a.text or sorted(p.name[:-8] for p in (a.out / "stage1").glob("*.json.gz"))
    for tid in tids:
        src = a.out / "stage1" / f"{tid}.json.gz"
        dst = a.out / "stage4" / "text" / f"{tid}.json.gz"
        sha = hashlib.sha256(src.read_bytes()).hexdigest()
        if dst.exists() and not a.force:
            m = json.load(gzip.open(dst, "rt"))["meta"]
            if m["stage1_sha256"] == sha and m["version"] == TEXT_VERSION:
                continue
        doc = json.load(gzip.open(src, "rt"))
        nsws = find_nsw(doc)
        for x in nsws:
            q = x["query"]
            if q not in memo:
                cands = aud.normalize(q, n_tagged=a.n_tagged, punct_post_process=True)
                memo[q] = {"default": det.normalize(q, punct_post_process=True), "candidates": sorted(cands)}
            r = memo[q]
            default = strip_context(to_alphabet(r["default"]), x["context"])
            cands = [strip_context(to_alphabet(c), x["context"]) for c in r["candidates"]]
            x["default"] = default
            x["candidates"] = [c for c in dict.fromkeys([default] + cands) if c]
        in_nsw = {}
        for x in nsws:
            for t in range(x["tok_start"], x["tok_end"]):
                in_nsw[t] = x["id"]
        words = []  # [word, token, nsw-id or -1]
        done = set()
        for t, (s, p) in enumerate(zip(doc["tokens"]["surface"], doc["tokens"]["punct"])):
            if t in in_nsw:
                k = in_nsw[t]
                if k not in done:
                    done.add(k)
                    words.extend([w, t, k] for w in nsws[k]["default"].split())
                continue
            if not p:
                words.extend([w, t, -1] for w in plain_words(s))
        out = {"meta": {"text": tid, "version": TEXT_VERSION, "stage1_sha256": sha, "n_tagged": a.n_tagged},
               "nsw": nsws, "words": words}
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as f:
            json.dump(out, f, ensure_ascii=False)
        tmp.replace(dst)
        memo_p.write_text(json.dumps(memo, ensure_ascii=False))
        print(f"{tid}: {len(nsws)} NSW, {len(words)} words", flush=True)
    return 0


ABBREV_FORMS = {"Mr.": ["mister"], "Mrs.": ["missus", "mistress"], "Messrs.": ["messieurs", "messrs"],
                "Dr.": ["doctor"], "St.": ["saint", "street"], "Mt.": ["mount"], "Jr.": ["junior"],
                "Sr.": ["senior"], "Capt.": ["captain"], "Col.": ["colonel"], "Gen.": ["general"],
                "Lieut.": ["lieutenant", "leftenant"], "Rev.": ["reverend"], "Prof.": ["professor"],
                "Hon.": ["honourable", "honorable"], "Esq.": ["esquire"], "Co.": ["company", "co"],
                "Bros.": ["brothers"], "viz.": ["namely", "videlicet", "viz"], "etc.": ["et cetera", "etcetera"],
                "&c.": ["et cetera", "etcetera", "and so forth", "and so on"],
                "No.": ["number"], "Nos.": ["numbers"], "vol.": ["volume"], "Vol.": ["volume"],
                "ch.": ["chapter"], "Ch.": ["chapter"], "p.": ["page"], "pp.": ["pages"],
                "ft.": ["feet", "foot"], "oz.": ["ounces", "ounce"], "lb.": ["pounds", "pound"],
                "lbs.": ["pounds"], "yd.": ["yards", "yard"], "yds.": ["yards"]}
CURRENCY = {"£": ["pounds", "pound"], "$": ["dollars", "dollar"]}


def rule_candidates(x: dict) -> list[str]:
    """Spoken forms NeMo misses ("twenty thousand pounds", "eight forty five", "et cetera")."""
    from num2words import num2words

    from .spoken import number_variants, roman_to_int

    def words(n: int) -> list[str]:
        vs = [v for v in number_variants(n) if not v.startswith("number ")]
        if 1100 <= n <= 2099:
            vs.append(" ".join(re.split(r"[\s,-]+", num2words(n, to="year"))))
        return vs

    sur = x["surface"]
    out: list[str] = []
    if x["kind"] == "abbrev":
        head = re.match(r"^(&c\.|\w+\.)", sur).group(1)
        forms = ABBREV_FORMS.get(head, [])
        num = re.sub(r"[^\d]", "", sur[len(head):])
        out += [f"{f} {v}" for f in forms for v in words(int(num))] if num else forms
    elif x["kind"] == "roman":
        n = roman_to_int(sur)
        if n:
            card = " ".join(re.split(r"[\s,-]+", num2words(n)))
            out += [card, "the " + " ".join(re.split(r"[\s,-]+", num2words(n, to="ordinal")))]
    elif x["kind"] == "number":
        m = re.fullmatch(r"([£$]?)(\d{1,3}(?:,\d{3})+|\d+)(?:(st|nd|rd|th|d)|([.:])(\d{2}))?(%?)", sur)
        if m:
            cur, num, ordn, sep, frac, pct = m.groups()
            n = int(num.replace(",", ""))
            if ordn:
                out.append(" ".join(re.split(r"[\s,-]+", num2words(n, to="ordinal"))))
            elif frac:  # 8.45 / 12:30 as a time; 3.14 as a decimal
                f = int(frac)
                hm = number_variants(n)[0] + " " + (number_variants(f)[0] if f else "o'clock")
                out += [hm, f"{number_variants(n)[0]} {'o ' if f < 10 else ''}{number_variants(f)[0]}"]
            else:
                ws = words(n)
                if cur:
                    out += [f"{w} {c}" for w in ws for c in CURRENCY[cur]]
                elif pct:
                    out += [f"{w} {p}" for w in ws for p in ("per cent", "percent")]
                else:
                    out += ws
    return [to_alphabet(c) for c in out]


def candidates_for(x: dict) -> list[str]:
    """NeMo candidates + rule candidates, minus 'candidates' that are just the
    unconverted surface ("mr fogg", "chapter xii")."""
    raw = set(plain_words(x["surface"]))
    keep = [c for c in x["candidates"] if c and not (set(c.split()) & raw and x["kind"] != "number"
                                                    and c.split() == plain_words(x["surface"]))]
    rules = rule_candidates(x)
    default = x["default"]
    if x["kind"] != "number" and default.split() == plain_words(x["surface"]):
        # NeMo's deterministic default left it unconverted ("st", "iv"): use a real form
        default = next((c for c in rules + keep if c and c != x["default"]), default)
    return [c for c in dict.fromkeys([default] + keep + rules) if c and c != (x["default"] if default != x["default"] else None)]


def window_cer(cand: str, window: str) -> float:
    """CER of the candidate against its best-matching run of words in the window.

    The window runs between the nearest exactly matched neighbours, so it can also
    hold misrecognised neighbouring words ("doctor libesey" for "Dr. Livesey");
    those must not count against the candidate.
    """
    from rapidfuzz.distance import Levenshtein

    a = cand.replace(" ", "")
    ws = window.split()
    if not ws:
        return 1.0 if a else 0.0
    best = Levenshtein.distance(a, "") / max(len(a), 1)
    for i in range(len(ws)):
        for j in range(i + 1, len(ws) + 1):
            best = min(best, Levenshtein.distance(a, "".join(ws[i:j])) / max(len(a), 1))
    return best


def select_section(ref: list[list], hyp: list[str], nsw: list[dict]) -> list[dict]:
    """Choose a candidate per NSW in ``ref`` by CER against its audio window."""
    from rapidfuzz.distance import Levenshtein

    ref_w = [w for w, _, _ in ref]
    matched = {}  # ref index -> hyp index for equal words
    for op in Levenshtein.opcodes(ref_w, hyp):
        if op.tag == "equal":
            for d in range(op.src_end - op.src_start):
                matched[op.src_start + d] = op.dest_start + d
    pos: dict[int, list[int]] = {}
    for i, (_, _, k) in enumerate(ref):
        if k >= 0:
            pos.setdefault(k, []).append(i)
    out = []
    for k, idx in pos.items():
        x = nsw[k]
        cands = candidates_for(x)
        i0, i1 = idx[0], idx[-1]
        left = max((j for j in matched if j < i0), default=None)
        right = min((j for j in matched if j > i1), default=None)
        h0 = matched[left] + 1 if left is not None else 0
        h1 = matched[right] if right is not None else len(hyp)
        window = " ".join(hyp[h0:h1]) if h1 - h0 <= 3 * max(1, len(x["default"].split())) + 6 else None
        scores = {c: round(window_cer(c, window), 3) for c in cands} if window is not None else {}
        choice = min(scores, key=lambda c: (scores[c], c != cands[0])) if scores else cands[0]
        out.append({"nsw": k, "surface": x["surface"], "kind": x["kind"], "default": cands[0],
                    "candidates": cands, "choice": choice,
                    "score": scores.get(choice), "window": window, "hyp_window": [h0, h1]})
    return out


def apply_choices(ref: list[list], choices: list[dict]) -> list[list]:
    by = {c["nsw"]: c["choice"] for c in choices}
    out, done = [], set()
    for w, t, k in ref:
        if k < 0:
            out.append([w, t, k])
        elif k not in done:
            done.add(k)
            out.extend([cw, t, k] for cw in by.get(k, w).split())
    return out


def cmd_select(a) -> int:
    inv = [json.loads(x) for x in (a.out / "inventory.jsonl").read_text().splitlines()]
    texts: dict[str, dict] = {}
    n_sec = n_nsw = 0
    for r in inv:
        if (a.work and r["work"] not in a.work) or (a.reading and r["librivox_slug"] not in a.reading):
            continue
        s3p = a.out / "stage3" / r["work"] / f"{r['reading_id']}.json"
        if not s3p.exists():
            continue
        s3 = json.loads(s3p.read_text())
        dst = a.out / "stage4" / r["work"] / f"{r['reading_id']}.json"
        key = hashlib.sha256((s3p.read_text() + VERSION).encode()).hexdigest()[:16]
        if dst.exists() and not a.force and json.loads(dst.read_text()).get("input_sha") == key:
            continue
        out = {"version": VERSION, "input_sha": key, "reading_id": r["reading_id"], "sections": []}
        for s in s3["sections"]:
            if "start" not in s:
                continue
            tid = s["text"]
            if tid not in texts:
                p = a.out / "stage4" / "text" / f"{tid}.json.gz"
                if not p.exists():
                    continue
                t = json.load(gzip.open(p, "rt"))
                d1 = json.load(gzip.open(a.out / "stage1" / f"{tid}.json.gz", "rt"))
                t["block_of_tok"] = d1["tokens"]["block"]
                texts[tid] = t
            t = texts[tid]
            b0, b1 = s["start"]["block"], s["end"]["block"]
            ranges = s.get("ranges") or [[b0, b1]]
            bt = t["block_of_tok"]
            ref = [x for x in t["words"] if any(r0 <= bt[x[1]] <= r1 for r0, r1 in ranges)]
            wd = json.loads((a.out / "w2v" / r["work"] / r["librivox_slug"] / f"{s['stem']}.json").read_text())["words"]
            lo, hi = s["book_words"]
            hyp = [re.sub(r"[^a-z']", "", w["word"].lower()).strip("'") for w in wd[lo:hi]]
            ch = select_section(ref, hyp, t["nsw"])
            out["sections"].append({"stem": s["stem"], "text": tid, "blocks": [b0, b1],
                                    **({"ranges": ranges} if len(ranges) > 1 or ranges[0] != [b0, b1] else {}),
                                    "ref": apply_choices(ref, ch), "nsw": ch})
            n_sec += 1
            n_nsw += len(ch)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, ensure_ascii=False))
    write_review(a.out)
    print(f"{n_sec} sections, {n_nsw} NSW choices")
    return 0


def write_review(out: Path) -> None:
    """nsw_summary.tsv: per surface form and choice, how often and how well it matched;
    nsw_lowconf.tsv: individual choices whose best candidate matched the audio badly."""
    from collections import Counter, defaultdict

    rdir = out / "review" / "stage4"
    rdir.mkdir(parents=True, exist_ok=True)
    agg: dict = defaultdict(list)
    low = []
    for p in sorted(out.glob("stage4/*/*.json")):
        d = json.loads(p.read_text())
        for sec in d["sections"]:
            for c in sec["nsw"]:
                agg[(c["kind"], c["surface"], c["choice"])].append(c["score"])
                if c["score"] is None or c["score"] > 0.5:
                    low.append((d["reading_id"], sec["stem"], c))
    with (rdir / "nsw_summary.tsv").open("w") as f:
        f.write("kind\tsurface\tchoice\tcount\tmedian_cer\n")
        for (k, sur, ch), sc in sorted(agg.items(), key=lambda kv: -len(kv[1])):
            v = sorted(x for x in sc if x is not None)
            f.write(f"{k}\t{sur}\t{ch}\t{len(sc)}\t{v[len(v) // 2] if v else ''}\n")
    with (rdir / "nsw_lowconf.tsv").open("w") as f:
        f.write("reading_id\tstem\tkind\tsurface\tchoice\tcer\taudio_window\tcandidates\n")
        for rid, stem, c in low:
            f.write(f"{rid}\t{stem}\t{c['kind']}\t{c['surface']}\t{c['choice']}\t{c['score']}\t"
                    f"{(c['window'] or '')[:80]}\t{' | '.join(c['candidates'][:6])}\n")
    n = sum(len(v) for v in agg.values())
    good = sum(1 for v in agg.values() for x in v if x is not None and x <= 0.2)
    kinds = Counter(k for (k, _, _), v in agg.items() for _ in v)
    (rdir / "summary.txt").write_text(f"nsw\t{n}\ncer_le_0.2\t{good}\nlowconf\t{len(low)}\n" +
                                      "".join(f"kind_{k}\t{v}\n" for k, v in sorted(kinds.items())))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["text", "select"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--text", action="append")
    ap.add_argument("--work", action="append")
    ap.add_argument("--reading", action="append")
    ap.add_argument("--n-tagged", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    return {"text": cmd_text, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
