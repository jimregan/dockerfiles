"""Stage 2: LibriVox metadata and header/footer templates, checked against wav2vec2.

    uv run lvm-headers [--work W] [--reading SLUG] [--force] [--jobs N]

For every section audio file, build the *expected* spoken header and footer
(LibriVox disclaimer, "Chapter N of <title>", "<title> by <author>", reader
credit, "End of chapter N", end of book, the cast list in dramatic readings) as
template elements with a few spoken variants, then fuzzy-locate each element in
the wav2vec2 transcript near the start/end of the file.

Templates are expectations, not truth: readers reorder, omit and ad-lib. This
stage records what was found where (word span, times, score), what was not,
and the unmatched words between matches (ad-libs: places, dates, dedications).
Exact alignment of header/footer segments happens in Stage 5.

Outputs:
  /out/stage2/<work>/<reading_id>.json      per reading: metadata + per-section matches
  /out/review/stage2/sections.tsv           one row per section
  /out/review/stage2/flagged.tsv            poor matches to review
  /out/review/stage2/cast.tsv               cast-list sections: entries found / missed
  /out/review/stage2/summary.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rapidfuzz.fuzz import partial_ratio_alignment

from .spoken import letters, number_variants, roman_to_int, spoken_words

VERSION = "stage2-0.12"
log = logging.getLogger("stage2")

DISCLAIMER_LONG = ("This is a LibriVox recording. All LibriVox recordings are in the public domain. "
                   "For more information or to volunteer, please visit librivox dot org.")
DISCLAIMER_LONG2 = ("This is a LibriVox recording. All LibriVox recordings are in the public domain. "
                    "For more information, and to find out how to volunteer, please contact librivox dot org.")
DISCLAIMER_SHORT = "This LibriVox recording is in the public domain."
DISCLAIMER_SHORT_VARIANTS = [DISCLAIMER_SHORT, "This is a LibriVox public domain recording.",
                             "A public domain recording for librivox dot org",
                             "This recording is in the public domain.", "This poem is in the public domain."]
FOOTER_PD = "This recording is in the public domain."
SECTION_WORDS = ["section", "chapter", "part", "book", "stave", "fable", "story", "poem", "letter"]
CREDIT_VERBS = ["recording by", "recorded by", "read by", "reading by", "this reading by", "narrated by",
                "read for librivox dot org by"]
CAST_SECTION = re.compile(r"\b(cast|dramatis|credits|characters)\b", re.I)

ACCEPT = 80.0          # partial-ratio score to accept a match
ACCEPT_SHORT = 90.0    # for needles under 15 letters (e.g. "end of chapter two")


# --------------------------------------------------------------------------- metadata
def clean_title(t: str) -> str:
    t = re.sub(r"\s*\((?:[^)]*?(?:version|dramatic|abridged|condensed|ver\.?)\s*\d*[^)]*)\)", "", t, flags=re.I)
    t = re.sub(r"\s*,?\s*\b(?:version|ver\.?)\s*\d+\b", "", t, flags=re.I)
    return t.strip(" -:,")


def clean_section_title(t: str) -> str:
    t = re.sub(r"^\s*\d+(?:\.\d+)?\s*[-–—:.]?\s*", "", t or "")      # "22 - Under..." / "12.14 The ..."
    return t.strip()


def title_variants(title: str) -> list[str]:
    """Full clean title, the part before ',', ':' or '(', and without 'Volume N'."""
    out = [title]
    core = re.split(r"\s*[,:(]\s*|\s+-\s+", title)[0]
    out.append(core)
    out.append(re.sub(r",?\s*\b(?:volume|vol\.?|book|part)\s+\w+\b.*$", "", title, flags=re.I))
    return [t for i, t in enumerate(out) if t and t not in out[:i]]


def translator(credits: str) -> str | None:
    m = re.search(r"Translated by\s+(.+?)(?:\s*\(|\s*\||$)", credits or "")
    return m.group(1).strip() if m else None


def section_word(title: str) -> list[str]:
    m = re.match(r"\s*(chapter|stave|part|book|fable|story|poem|letter|section)\b", title or "", re.I)
    return list(dict.fromkeys(([m.group(1).lower()] if m else []) + ["section", "chapter", "part"]))


def title_numbers(title: str) -> list[tuple[str, int]]:
    """Numbered units named in a section title: 'Part I, Chapter 05' -> [('part',1),('chapter',5)]."""
    out = []
    for w, num in re.findall(r"\b(chapter|stave|part|book|fable|story|letter|canto|act|scene)\s+([ivxlcdm]+|\d+)\b",
                             title or "", re.I):
        n = int(num) if num.isdigit() else roman_to_int(num)
        if n:
            out.append((w.lower(), n))
    return out


def section_number(sec: str | None) -> int | None:
    try:
        return int(re.sub(r"\D", "", sec or "")) if re.search(r"\d", sec or "") else None
    except ValueError:
        return None


# --------------------------------------------------------------------------- templates
def element(kind: str, role: str, variants: list[str], **extra) -> dict:
    vs = []
    for v in variants:
        w = spoken_words(v)
        if w and " ".join(w) not in {x["text"] for x in vs}:
            vs.append({"text": " ".join(w), "source": v})
    return {"kind": kind, "role": role, "variants": vs, **extra}


def templates_for(reading: dict, sec: dict, is_last: bool) -> list[dict]:
    title = clean_title(reading.get("title") or "")
    authors = [a for a in reading.get("authors") or [] if a]
    author = " and ".join(authors[:2])
    titles = title_variants(title)
    by_variants = [""] + [f" by {a}" for a in [author] + authors[:2] if a]
    tr = translator(reading.get("credits_line") or "")
    stitle = clean_section_title(sec.get("title") or "")
    n = section_number(sec.get("section"))
    sws = section_word(sec.get("title") or "")
    readers = [r["name"] for r in sec.get("readers") or []] or [r["name"] for r in reading.get("readers") or []][:1]
    readers = [re.sub(r"\s*\(.*?\)\s*", " ", r).strip() for r in readers]
    by = f" by {author}" if author else ""
    trs = f" translated by {tr}" if tr else ""
    els: list[dict] = []
    if n is not None:
        nv = number_variants(n)
        for w, k in title_numbers(sec.get("title") or ""):
            nv = nv + [f"{v}" for v in number_variants(k)[:2]]
        els.append(element("intro", "header", [f"{w} {k} of {t}{v}{x}" for w in sws for k in nv for t in titles
                                               for v in by_variants for x in ({trs, ""} if v else {""})]))
    els.append(element("disclaimer_long", "header", [DISCLAIMER_LONG, DISCLAIMER_LONG2]))
    els.append(element("disclaimer_short", "header", DISCLAIMER_SHORT_VARIANTS))
    if title:
        els.append(element("title_author", "header", [f"{t}{v}{x}" for t in titles for v in by_variants[1:] or [""]
                                                      for x in {trs, ""}]
                           + [t for t in titles if len(spoken_words(t)) >= 2]))
    for rd in readers[:3]:
        for role in ("header", "footer"):
            els.append(element("credit", role, [f"{v} {rd}" for v in CREDIT_VERBS]
                               + [f"read by {rd} for librivox dot org", f"recorded by {rd} for librivox dot org"],
                               reader=rd))
    if len(spoken_words(stitle)) >= 2:
        els.append(element("section_title", "header", [stitle], may_be_book_text=True))
    ends = []
    if n is not None:
        nv = number_variants(n)
        ends += [f"end of {w} {k}" for w in sws for k in nv] + [f"end of {w} {k} of {title}" for w in sws for k in nv[:2]]
    ends += [f"end of {w}" for w in sws]
    for w, k in title_numbers(sec.get("title") or ""):  # "Part I, Chapter 05" -> "end of chapter five"
        ends += [f"end of {w} {v}" for v in number_variants(k)[:3]]
    if stitle:
        ends.append(f"end of {stitle}")
    ends += ["end of poem", "end of story", "end of fable", "end of the poem", "end of the story", "end of text"]
    els.append(element("end", "footer", ends))
    els.append(element("footer_pd", "footer", [FOOTER_PD, DISCLAIMER_SHORT]))
    if is_last and title:
        els.append(element("end_of_book", "footer", [f"end of {t}{v}" for t in titles for v in by_variants]))
    if CAST_SECTION.search(sec.get("title") or ""):
        for c in reading.get("cast") or []:
            if c.get("pool"):
                continue
            for rd in c["readers"]:
                for role_name in c["roles"]:
                    # the spoken name often differs from the page's username ("SweetPea" -> "rachel"),
                    # so role + verb alone is also a variant; the spoken name is captured after it
                    els.append(element("cast", "body", [f"{role_name} read by {rd['name']}",
                                                         f"{role_name} played by {rd['name']}",
                                                         f"{role_name} {rd['name']}",
                                                         f"{role_name} read by", f"{role_name} played by",
                                                         f"{role_name} narrated by"],
                                       character=role_name, reader=rd["name"], reader_id=rd.get("id")))
    return [e for e in els if e["variants"]]


# --------------------------------------------------------------------------- matching
def locate(needle: str, words: list[dict], lo: int, hi: int) -> tuple[float, int, int] | None:
    """Best fuzzy occurrence of ``needle`` (letters) in words[lo:hi] -> (score, w0, w1)."""
    if hi <= lo or not needle:
        return None
    offs, parts, pos = [], [], 0
    for i in range(lo, hi):
        t = words[i]["l"]
        offs.append((pos, pos + len(t), i))
        parts.append(t)
        pos += len(t)
    hay = "".join(parts)
    if len(needle) > len(hay):
        return None
    r = partial_ratio_alignment(needle, hay)
    if r is None:
        return None
    # a word belongs to the match if most of its letters are inside the aligned span
    inside = [i for a, z, i in offs if min(z, r.dest_end) - max(a, r.dest_start) > (z - a) / 2]
    if not inside:
        inside = [next(i for a, z, i in offs if z > r.dest_start)]
    return r.score, inside[0], inside[-1] + 1


def match_section(els: list[dict], words: list[dict], dur: float, cast_section: bool) -> dict:
    n = len(words)
    hdr_t = min(180.0, max(60.0, 0.35 * dur))
    ftr_t = min(120.0, max(45.0, 0.35 * dur))
    hdr_hi = next((i for i, w in enumerate(words) if w["start"] > hdr_t), n)
    ftr_lo = next((i for i, w in enumerate(words) if w["end"] >= dur - ftr_t), n)
    if hdr_hi > ftr_lo:  # short file: header and footer regions would overlap
        hdr_hi = ftr_lo = n // 2
    regions = {"header": (0, hdr_hi), "footer": (ftr_lo, n), "body": (0, n if cast_section else hdr_hi)}
    found = []
    for ei, e in enumerate(els):
        lo, hi = regions[e["role"]]
        best = None
        for vi, v in enumerate(e["variants"]):
            needle = letters(v["text"].split())
            r = locate(needle, words, lo, hi)
            if r and (best is None or r[0] > best[0] or (r[0] == best[0] and len(needle) > best[4])):
                best = (*r, vi, len(needle))
        if best is None:
            continue
        score, w0, w1, vi, nlen = best
        if score >= (ACCEPT_SHORT if nlen < 15 else ACCEPT):
            found.append({"element": ei, "kind": e["kind"], "role": e["role"], "variant": vi,
                          "score": round(score, 1), "w0": w0, "w1": w1, "nlen": nlen})
    # resolve overlaps: prefer high score x length; keep non-overlapping
    # rank by score band first (a perfect short match beats a mediocre long one), then length
    found.sort(key=lambda m: (-(m["score"] // 5), -m["nlen"], m["w0"]))
    kept: list[dict] = []
    for m in found:
        ok = True
        for k in kept:
            ov = min(m["w1"], k["w1"]) - max(m["w0"], k["w0"])
            if ov <= 0:
                continue
            if ov <= max(2, 0.2 * (m["w1"] - m["w0"])):  # small overlap: trim the weaker match
                if k["w0"] >= m["w0"]:
                    m["w1"] = min(m["w1"], k["w0"])
                else:
                    m["w0"] = max(m["w0"], k["w1"])
                if m["w1"] <= m["w0"]:
                    ok = False
                    break
            else:
                ok = False
                break
        if ok:
            kept.append(m)
    kept.sort(key=lambda m: m["w0"])
    for m in kept:
        m["start"] = words[m["w0"]]["start"]
        m["end"] = words[m["w1"] - 1]["end"]
        m["hyp"] = " ".join(words[i]["w"] for i in range(m["w0"], m["w1"]))
        m.pop("nlen")
    # cast entries: the reader's name as spoken = words after the match, up to the next match (max 5)
    for j, m in enumerate(kept):
        if m["kind"] == "cast":
            nxt = kept[j + 1]["w0"] if j + 1 < len(kept) else n
            v = els[m["element"]]["variants"][m["variant"]]["text"]
            if v.endswith((" by",)):
                # approximate: the window up to the next "read/red/played/narrated by" also
                # contains the next character's name, which cannot be split off reliably
                stop = min(nxt, m["w1"] + 6)
                for i in range(m["w1"] + 1, stop):
                    if words[i]["l"] == "by" and words[i - 1]["l"] in ("read", "red", "played", "narrated"):
                        stop = i - 1
                        break
                m["after_verb"] = " ".join(words[i]["w"] for i in range(m["w1"], stop))
    hdr = [m for m in kept if m["role"] in ("header", "body") and m["kind"] != "cast"]
    ftr = [m for m in kept if m["role"] == "footer"]

    def gaps(ms: list[dict], lo: int, hi: int) -> list[dict]:
        """Unmatched word runs between matches inside [lo, hi): ad-lib candidates."""
        out, cur = [], lo
        for m in ms:
            if m["w0"] > cur:
                out.append({"w0": cur, "w1": m["w0"], "text": " ".join(words[i]["w"] for i in range(cur, m["w0"]))})
            cur = max(cur, m["w1"])
        if hi > cur:
            out.append({"w0": cur, "w1": hi, "text": " ".join(words[i]["w"] for i in range(cur, hi))})
        return out

    hdr_end = max((m["w1"] for m in hdr), default=0)
    ftr_start = min((m["w0"] for m in ftr), default=n)
    return {
        "n_words": n, "header_region_end_word": hdr_hi, "footer_region_start_word": ftr_lo,
        "matches": kept,
        "header_last_word": hdr_end, "footer_first_word": ftr_start,
        # words before/among header matches; words among/after footer matches
        "header_gaps": gaps(hdr, 0, hdr_end) if hdr else [],
        # words right after the last header match: ad-libs (place, date, dedication) or
        # already book text; Stage 3 finds where the book starts
        "header_after": " ".join(words[i]["w"] for i in range(hdr_end, min(n, hdr_end + 12))) if hdr else "",
        "footer_gaps": gaps(ftr, ftr_start, n) if ftr else [],
    }


# --------------------------------------------------------------------------- driver
def process_reading(args) -> dict:
    reading, w2v_root, out_root = args
    secs = reading["sections"]
    out = {"version": VERSION, "reading_id": reading["reading_id"], "work": reading["work"],
           "librivox_slug": reading["librivox_slug"], "librivox_id": reading["librivox_id"],
           "title": reading["title"], "title_clean": clean_title(reading["title"] or ""),
           "authors": reading["authors"], "translator": translator(reading.get("credits_line") or ""),
           "readers": reading["readers"], "cast": reading["cast"], "dramatic": reading["dramatic"],
           "sections": []}
    numbered = sorted((section_number(s["section"]) or 0, i) for i, s in enumerate(secs))
    last_idx = numbered[-1][1] if numbered else None
    listed_last = str(reading.get("n_sections_listed") or "")
    for i, s in enumerate(secs):
        is_last = i == last_idx and (not reading.get("collection")) and (
            section_number(s["section"]) == section_number(listed_last) or reading.get("n_sections_listed") == len(secs))
        els = templates_for(reading, s, is_last)
        p = w2v_root / reading["work"] / reading["librivox_slug"] / f"{s['stem']}.json"
        rec = {"section": s["section"], "stem": s["stem"], "title": s["title"], "duration": s["duration"],
               "readers": s["readers"], "is_last": is_last, "templates": els}
        if not p.exists():
            rec["error"] = "no wav2vec2 output"
        else:
            d = json.loads(p.read_text())
            words = [{"w": x["word"].lower(), "l": re.sub(r"[^a-z']", "", x["word"].lower()),
                      "start": x["start"], "end": x["end"]} for x in d["words"]]
            rec.update(match_section(els, words, s["duration"], bool(CAST_SECTION.search(s["title"] or ""))))
        out["sections"].append(rec)
    dst = out_root / reading["work"] / f"{reading['reading_id'].replace(':', '_')}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False))
    tmp.replace(dst)
    return out


def section_flags(sec: dict) -> dict:
    ms = sec.get("matches", [])
    kinds = Counter(m["kind"] for m in ms if m["kind"] != "credit" or True)
    best = {}
    for m in ms:
        k = m["kind"] + ("_footer" if m["kind"] == "credit" and m["role"] == "footer" else "")
        best[k] = max(best.get(k, 0), m["score"])
    # a credit counts as LibriVox attribution if "libri vox" (any ASR spelling) is in the
    # matched words or right after them ("read by X for libr vox dot org")
    near = " ".join(m["hyp"] for m in ms if m["kind"] == "credit") + " " + (sec.get("header_after") or "")
    lv_credit = bool(re.search(r"\bl[aeiy]\w{0,4}\s?[vo]?ox\b|\bvox\b", near))
    # where the LibriVox attribution is: long/short disclaimer up front, or the older
    # "read for librivox dot org by X ... this recording is in the public domain" format
    disc = ("long" if "disclaimer_long" in kinds else "short" if "disclaimer_short" in kinds
            else "credit+footer" if lv_credit and "footer_pd" in kinds
            else "footer" if "footer_pd" in kinds else "credit" if lv_credit else None)
    flags = []
    if "error" in sec:
        flags.append(sec["error"])
    if not disc:
        flags.append("no_disclaimer")
    if not any(k in kinds for k in ("end", "end_of_book", "footer_pd")):
        flags.append("no_footer")
    if not any(k in best for k in ("intro", "title_author")):
        flags.append("no_title")
    adlib_h = [g["text"] for g in sec.get("header_gaps", []) if g["w1"] - g["w0"] >= 3]
    adlib_f = [g["text"] for g in sec.get("footer_gaps", []) if g["w1"] - g["w0"] >= 3]
    return {"disclaimer": disc, "best": best, "flags": flags, "adlib_header": adlib_h, "adlib_footer": adlib_f}


def write_review(rdir: Path, readings: list[dict]) -> None:
    rdir.mkdir(parents=True, exist_ok=True)
    c = lambda x: re.sub(r"[\t\r\n]+", " ", str(x if x is not None else ""))  # noqa: E731
    cols = ["intro", "disclaimer_long", "disclaimer_short", "title_author", "credit", "section_title",
            "end", "end_of_book", "footer_pd", "credit_footer"]
    stats = Counter()
    with (rdir / "sections.tsv").open("w") as f, (rdir / "flagged.tsv").open("w") as g, \
            (rdir / "cast.tsv").open("w") as h:
        head = ("work\treading_id\tsection\tstem\tduration\tdisclaimer\t" + "\t".join(cols) +
                "\tflags\tadlib_header\tadlib_footer\theader_text\tfooter_text\theader_after\n")
        f.write(head)
        g.write(head)
        h.write("reading_id\tsection\tcharacter\treader\tfound\tscore\thyp\tafter_verb\n")
        for r in readings:
            for s in r["sections"]:
                fl = section_flags(s)
                stats["sections"] += 1
                stats[f"disclaimer_{fl['disclaimer']}"] += 1
                for k in fl["flags"]:
                    stats[k] += 1
                if fl["adlib_header"]:
                    stats["adlib_header"] += 1
                if fl["adlib_footer"]:
                    stats["adlib_footer"] += 1
                hw = s.get("header_region_end_word", 0)
                # header/footer text for eyeballing: first 45 / last 30 words
                ms = s.get("matches", [])
                htxt = " ".join(m["hyp"] for m in ms if m["role"] != "footer")[:300]
                ftxt = " ".join(m["hyp"] for m in ms if m["role"] == "footer")[:200]
                row = (f"{r['work']}\t{r['reading_id']}\t{s['section']}\t{s['stem']}\t{round(s['duration'])}\t"
                       f"{fl['disclaimer'] or ''}\t" + "\t".join(str(fl["best"].get(k, "")) for k in cols) +
                       f"\t{','.join(fl['flags'])}\t{c(' | '.join(fl['adlib_header']))[:300]}\t"
                       f"{c(' | '.join(fl['adlib_footer']))[:300]}\t{c(htxt)}\t{c(ftxt)}\t"
                       f"{c(s.get('header_after', ''))}\n")
                f.write(row)
                if fl["flags"]:
                    g.write(row)
                cast_els = [(i, e) for i, e in enumerate(s["templates"]) if e["kind"] == "cast"]
                if cast_els:
                    got = {m["element"]: m for m in ms if m["kind"] == "cast"}
                    for i, e in cast_els:
                        m = got.get(i)
                        stats["cast_entries"] += 1
                        stats["cast_found"] += bool(m)
                        h.write(f"{r['reading_id']}\t{s['section']}\t{c(e['character'])}\t{c(e['reader'])}\t"
                                f"{int(bool(m))}\t{m['score'] if m else ''}\t{c(m['hyp']) if m else ''}\t"
                                f"{c(m.get('after_verb')) if m else ''}\n")
    (rdir / "summary.txt").write_text("\n".join(f"{k}\t{v}" for k, v in sorted(stats.items())) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", type=Path, default=Path("/out/inventory.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("/out"))
    ap.add_argument("--work", action="append")
    ap.add_argument("--reading", action="append")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args(argv)
    (a.out / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(a.out / "logs" / "stage2.log"), logging.StreamHandler(sys.stderr)])
    out_root = a.out / "stage2"
    readings = [json.loads(x) for x in a.inventory.read_text().splitlines()]
    sel = [r for r in readings if (not a.work or r["work"] in a.work)
           and (not a.reading or r["librivox_slug"] in a.reading)]
    todo, done = [], []
    for r in sel:
        dst = out_root / r["work"] / f"{r['reading_id'].replace(':', '_')}.json"
        if dst.exists() and not a.force:
            d = json.loads(dst.read_text())
            if d.get("version") == VERSION:
                done.append(d)
                continue
        todo.append((r, a.out / "w2v", out_root))
    log.info("%d readings selected, %d done, %d to do", len(sel), len(done), len(todo))
    with ProcessPoolExecutor(a.jobs) as ex:
        for d in ex.map(process_reading, todo, chunksize=1):
            done.append(d)
    done.sort(key=lambda d: (d["work"], d["reading_id"]))
    # review tables cover everything on disk, not only this run's selection
    all_out = [json.loads(p.read_text()) for p in sorted(out_root.glob("*/*.json"))]
    write_review(a.out / "review" / "stage2", sorted(all_out, key=lambda d: (d["work"], d["reading_id"])))
    log.info("wrote %d readings", len(done))
    return 0


if __name__ == "__main__":
    sys.exit(main())
