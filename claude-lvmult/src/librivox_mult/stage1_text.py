"""Stage 1: Gutenberg HTML -> structured source text with character offsets.

    uv run lvm-text [--id N ...] [--work W] [--reading SLUG] [--force]

Input:  /out/gutenberg/pg<N>-images.html (+ manifest.jsonl for sha256)
Output: /out/stage1/pg<N>.json.gz, review tables under /out/review/stage1/

Conventions (see docs/text.md):

* **Canonical text** (``text``): the visible body between Gutenberg's
  ``#pg-start-separator`` and ``#pg-end-separator``; whitespace collapsed within a
  line, lines joined by "\\n", blocks by "\\n\\n". Page numbers (``span.pagenum``)
  and images are not text. Every offset in the output points into ``text``; every
  block also records its HTML path, so the HTML stays the ground truth.
* **Blocks** are the innermost text-bearing block elements (p, h1-h6, pre, li,
  td, div.stanza, ...). Every block is a *paragraph*: numbered 1-based within its
  division, headings, epigraphs, captions and verse stanzas included (Jim's
  prototype convention). ``kind`` says what it is, so later stages can filter.
* **Divisions** (chapters): a run of consecutive heading blocks opens a new
  division; anything before the first heading is division 0.
* **Verse**: div.poem/div.stanza, <pre>, and multi-line (<br>) p/td blocks that
  look like verse. A blank line splits a stanza. Consecutive verse blocks in a
  division form a poem. Lines are stored per block.
* **Sentences**: Moses sentence splitter + rejoin of splits ending in ";"
  (Jim's prototype). Prose: per block. Verse: per poem, so sentences may cross
  lines and stanzas.
* **Tokens**: words ``[^\\W_]+(?:['’-][^\\W_]+)*`` and single punctuation chars;
  surface is ``text[start:end]``.
* **Quotes**: spans of “…”, ‘…’, "…" per block, nesting level, ``closed`` false
  when a quote runs to the end of the block (multi-paragraph speech).
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
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

VERSION = "stage1-0.5"
log = logging.getLogger("stage1")

BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "li", "dt", "dd",
              "td", "th", "figcaption", "caption"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
SKIP_CLASS = re.compile(r"\b(pagenum|pageno|page-num|pagenumber)\b", re.I)
VERSE_CLASS = re.compile(r"\b(poem|stanza|verse|poetry|tei-lg)\b", re.I)
HEADING_CLASS = re.compile(r"\b(tei-head|secthead|chapterhead|chapter-head|chaptertitle)\b", re.I)
NUMWORD = r"(?:[IVXLCDM]+|\d+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|First|Second|Third|Fourth|Fifth|Last|the\s+\w+)"
HEADING_TEXT = re.compile(
    r"^(?:(?:CHAPTER|STAVE|BOOK|PART|ACT|SCENE|LETTER|CANTO)\s+(?:[IVXLCDM]+|\d+|[A-Z][A-Z-]+)\b"
    r"|(?:Chapter|Stave|Book|Part|Act|Scene|Letter|Canto)\s+" + NUMWORD + r"\b"
    r"|[IVXLCDM]+\.?$)"
)
# Letter sign-offs and imprints; only applied to short (<=4 line) blocks, where
# they are far likelier than verse ("Your friend, / Dracula.").
SHORT_CLOSING = re.compile(
    r"^[\W_]*(?:your\b|yours\b|ever\b|believe\s+me|we\s+are\b|i\s+am\b|with\s+(?:much\s+)?love|copyright)",
    re.I)
PG_EDITIONS_NOTE = re.compile(r"^There are several editions of this ebook in the Project Gutenberg collection")
PSEUDO_HEADING_CLASS = re.compile(r"\b(ph\d|chapter)\b")
LETTER_LINE = re.compile(
    r"^[\W_]*(?:yours|your\s+(?:loving|affectionate|afecksionate|true|sincere|obedient|devoted|faithful)"
    r"|ever\s+yours|dear(?:est)?\b|my\s+dear|sincerely|faithfully|affectionately|respectfully)", re.I)
CAPTION_CLASS = re.compile(r"\b(caption|figcenter|figure|figleft|figright|illus)\b", re.I)
FOOTNOTE_CLASS = re.compile(r"\b(footnotes?|fnote)\b", re.I)
TOC_CLASS = re.compile(r"\b(toc|contents)\b", re.I)
INDENT_CLASS = re.compile(r"\bi(\d+)\b")
SEPARATOR_TEXT = re.compile(r"^[\s*·•.⁂—–-]*$")
TOC_HEADING = re.compile(r"^\s*(table of )?contents\.?\s*$|^\s*(list of )?illustrations\.?\s*$", re.I)
WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")


def classes(el: Tag) -> str:
    c = el.get("class") or []
    return " ".join(c) if isinstance(c, list) else str(c)


def is_skipped(el: Tag) -> bool:
    return el.name in ("script", "style", "img") or bool(SKIP_CLASS.search(classes(el)))


def html_path(el: Tag) -> str:
    parts = []
    while el is not None and isinstance(el, Tag) and el.name not in ("[document]", "html"):
        sib = 1 + sum(1 for s in el.previous_siblings if isinstance(s, Tag) and s.name == el.name)
        parts.append(f"{el.name}[{sib}]")
        el = el.parent
    return "/" + "/".join(reversed(parts))


def ancestors_match(el: Tag, rx: re.Pattern, stop: Tag | None = None) -> bool:
    for a in el.parents:
        if a is stop or not isinstance(a, Tag):
            break
        if rx.search(classes(a)) or rx.search(a.get("id") or ""):
            return True
    return False


def extract_lines(el: Tag, pre: bool = False) -> list[dict]:
    """Visible text of ``el`` as lines (split at <br>, or newlines inside <pre>).

    Returns [{"text": str, "indent": str|None}] with empty strings kept as
    stanza-break markers (stripped later).
    """
    lines: list[list] = [[[], None]]

    def walk(node, indent):
        for ch in node.children:
            if isinstance(ch, NavigableString):
                if ch.__class__.__name__ in ("Comment", "Doctype", "Declaration", "ProcessingInstruction"):
                    continue
                s = str(ch)
                if pre:
                    parts = s.split("\n")
                    for k, part in enumerate(parts):
                        if k:
                            lines.append([[], None])
                        if part.strip() and lines[-1][1] is None:
                            lines[-1][1] = f"pre:{len(part) - len(part.lstrip(' '))}"
                        lines[-1][0].append(part)
                else:
                    lines[-1][0].append(s)
                    if s.strip() and lines[-1][1] is None and indent:
                        lines[-1][1] = indent
            elif isinstance(ch, Tag):
                if is_skipped(ch):
                    continue
                if ch.name == "br":
                    lines.append([[], None])
                    continue
                ind = indent
                m = INDENT_CLASS.search(classes(ch))
                if m:
                    ind = f"i{m.group(1)}"
                elif "margin-left" in (ch.get("style") or ""):
                    ind = re.sub(r"\s+", "", ch["style"])
                block_child = ch.name in ("div", "p") and lines[-1][0] and "".join(lines[-1][0]).strip()
                if block_child:  # e.g. div.stanza > div.line: each child its own line
                    lines.append([[], None])
                walk(ch, ind)
                if ch.name in ("div", "p"):
                    lines.append([[], None])

    walk(el, None)
    out = []
    for parts, ind in lines:
        t = re.sub(r"\s+", " ", "".join(parts)).strip()
        out.append({"text": t, "indent": ind})
    # trim leading/trailing empties, collapse runs of empties to one
    res: list[dict] = []
    for ln in out:
        if not ln["text"] and (not res or not res[-1]["text"]):
            continue
        res.append(ln)
    while res and not res[-1]["text"]:
        res.pop()
    return res


@dataclass
class RawBlock:
    el: Tag
    kind: str
    lines: list[dict]          # non-empty lines only
    source: str                # markup pattern, for review
    verse: bool = False
    flags: list[str] = field(default_factory=list)
    confidence: str | None = None   # verse: "markup" | "heuristic" | "override"


def body_range(soup: BeautifulSoup) -> tuple[list, int, int]:
    nodes = list(soup.descendants)
    pos = {id(n): i for i, n in enumerate(nodes)}
    start = soup.find(id="pg-start-separator")
    end = soup.find(id="pg-end-separator") or soup.find(id="pg-footer")
    if start is None or end is None:
        raise ValueError("Gutenberg start/end separator not found")
    last = start
    for d in start.descendants:
        last = d
    return nodes, pos[id(last)] + 1, pos[id(end)]


def looks_like_verse(lines: list[str]) -> bool:
    """Heuristic for <br>-broken blocks without verse markup.

    Rejects title pages/imprints (mostly capitals), page references and
    digit-heavy lines (tables, dates, addresses), then requires short lines.
    """
    ls = [l for l in lines if l]
    if len(ls) < 2:
        return False
    letters = [c for c in "".join(ls) if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.5:
        return False
    if any(re.search(r"\[?Page \d+", l) for l in ls):
        return False
    if LETTER_LINE.match(ls[0]) or any(HEADING_TEXT.match(l) for l in ls):
        return False  # letter openings/closings; contents rows ("CHAPTER V. / Title")
    if len(ls) <= 4 and SHORT_CLOSING.match(ls[0]):
        return False  # letter openings/closings; contents rows ("CHAPTER V. / Title")
    if re.match(r"^\W*\[?\s*Transcriber", ls[0], re.I):
        return False
    if sum(c.isdigit() for c in "".join(ls)) > 0.1 * len("".join(ls)):
        return False
    lens = sorted(len(l) for l in ls)
    med = lens[len(lens) // 2]
    return med <= 60 and lens[-1] <= 90


def find_blocks(soup: BeautifulSoup, overrides: dict | None = None) -> tuple[list[RawBlock], dict]:
    overrides = overrides or {}
    nodes, lo, hi = body_range(soup)
    in_range = nodes[lo:hi]
    pos = {id(n): i for i, n in enumerate(nodes)}

    def has_text(el: Tag) -> bool:
        return any(isinstance(t, NavigableString) and t.__class__.__name__ == "NavigableString"
                   and str(t).strip() and not any(is_skipped(a) for a in t.parents if a is not el)
                   for t in el.descendants)

    # Stanza-level verse containers: verse-like divs with no verse-like div inside.
    # Everything inside one belongs to that stanza (lines), not to its own block.
    verse_like = [n for n in in_range if isinstance(n, Tag) and n.name == "div"
                  and VERSE_CLASS.search(classes(n))]
    vset = {id(v) for v in verse_like}
    stanza_level = [v for v in verse_like
                    if not any(isinstance(d, Tag) and (id(d) in vset or d.name in BLOCK_TAGS)
                               for d in v.descendants)]
    inside_stanza = {id(d) for v in stanza_level for d in v.descendants}

    cand = []
    for n in in_range:
        if not isinstance(n, Tag) or is_skipped(n) or id(n) in inside_stanza:
            continue
        cl = classes(n)
        if n.name in BLOCK_TAGS or (n.name == "div" and id(n) in {id(v) for v in stanza_level}):
            cand.append(n)
        elif n.name == "div" and (HEADING_CLASS.search(cl) or has_text(n)):
            cand.append(n)  # divs with text: leaf/inline-only divs, TEI heads, letters
    cset = {id(c) for c in cand}
    # innermost wins: drop candidates that contain other candidates
    inner = [c for c in cand if not any(id(d) in cset for d in c.descendants if isinstance(d, Tag))]
    iset = {id(c) for c in inner}

    # Safety net: text not inside any block (e.g. bare spans next to divs) is
    # attached to its largest ancestor that contains no block.
    for n in in_range:
        if not (isinstance(n, NavigableString) and n.__class__.__name__ == "NavigableString" and str(n).strip()):
            continue
        if any(isinstance(a, Tag) and (is_skipped(a) or id(a) in iset) for a in n.parents):
            continue
        el = n.parent
        while (el.parent is not None and pos.get(id(el.parent), -1) >= lo
               and not any(id(d) in iset for d in el.parent.descendants if isinstance(d, Tag))):
            el = el.parent
        if id(el) not in iset:
            inner.append(el)
            iset.add(id(el))
    inner.sort(key=lambda e: pos[id(e)])

    blocks: list[RawBlock] = []
    for el in inner:
        pre = el.name == "pre"
        lines = extract_lines(el, pre=pre)
        texts = [l["text"] for l in lines]
        if not any(texts):
            continue
        joined = " ".join(t for t in texts if t)
        cl = classes(el)
        short = len(joined) <= 80 and el.name in ("p", "div")
        in_table = any(isinstance(x, Tag) and x.name in ("table", "td", "li") for x in el.parents)
        if el.name in HEADING_TAGS or HEADING_CLASS.search(cl):
            kind, source = "heading", el.name if el.name in HEADING_TAGS else f"{el.name}.{HEADING_CLASS.search(cl).group(1)}"
        elif short and not in_table and PSEUDO_HEADING_CLASS.search(cl):
            kind, source = "heading", f"{el.name}.{PSEUDO_HEADING_CLASS.search(cl).group(1)}"
        elif short and not in_table and HEADING_TEXT.match(joined) and not ancestors_match(el, TOC_CLASS):
            kind, source = "heading", f"{el.name}~text"
        elif SEPARATOR_TEXT.match(joined) or "asterism" in cl:
            kind, source = "separator", el.name
        elif ancestors_match(el, FOOTNOTE_CLASS) or FOOTNOTE_CLASS.search(cl):
            kind, source = "footnote", el.name
        elif el.name in ("figcaption", "caption") or CAPTION_CLASS.search(cl) or ancestors_match(el, CAPTION_CLASS):
            kind, source = "caption", el.name
        elif ancestors_match(el, TOC_CLASS) or TOC_CLASS.search(cl):
            kind, source = "toc", el.name
        else:
            kind, source = "para", el.name
        verse, conf = False, None
        multi = len([t for t in texts if t]) >= 2
        override = overrides.get(html_path(el))
        if kind == "para":
            if override in ("verse", "lines", "para"):
                verse, conf, source = override == "verse", "override", f"{el.name}:override"
                if override == "lines":
                    kind = "lines"
            elif VERSE_CLASS.search(cl) or ancestors_match(el, VERSE_CLASS):
                verse, conf = True, "markup"
                source = f"{el.name}.{VERSE_CLASS.search(cl + ' ' + ' '.join(classes(a) for a in el.parents if isinstance(a, Tag))).group(1)}"
            elif pre:
                verse, conf, source = True, "markup", "pre"
            elif multi and looks_like_verse(texts):
                verse, conf, source = True, "heuristic", f"{el.name}+br"
            elif multi:
                kind = "lines"  # line-broken prose: letters, addresses, lists, title pages
        elif override:
            kind = override
        flags = []
        if ancestors_match(el, re.compile(r"blockquote", re.I)) or any(
            isinstance(a, Tag) and a.name == "blockquote" for a in el.parents
        ):
            flags.append("blockquote")
        if verse:
            # split at blank lines into stanzas (one block per stanza)
            cur: list[dict] = []
            for ln in lines + [{"text": "", "indent": None}]:
                if ln["text"]:
                    cur.append(ln)
                elif cur:
                    blocks.append(RawBlock(el, "verse", cur, source, True, list(flags), conf))
                    cur = []
        else:
            nonempty = [l for l in lines if l["text"]]
            if len(nonempty) > 1 and kind not in ("lines", "heading"):
                flags.append("multiline")
            blocks.append(RawBlock(el, kind, nonempty, source, False, flags))

    # coverage: visible text in range not inside any kept block element
    kept = {id(b.el) for b in blocks}
    uncovered = []
    total = covered = 0
    for n in in_range:
        if not isinstance(n, NavigableString) or n.__class__.__name__ != "NavigableString":
            continue
        s = str(n).strip()
        if not s:
            continue
        if any(isinstance(a, Tag) and is_skipped(a) for a in n.parents):
            continue
        k = len(re.sub(r"\s", "", s))
        total += k
        if any(id(a) in kept for a in n.parents):
            covered += k
        else:
            uncovered.append(s[:80])
    return blocks, {"chars_total": total, "chars_covered": covered, "uncovered": uncovered}


class Splitter:
    """Moses sentence splitter + rejoin of splits ending in ';' (Jim's prototype)."""

    def __init__(self):
        from mosestokenizer import MosesSentenceSplitter

        self._ms = MosesSentenceSplitter("en")

    def __call__(self, text: str) -> list[str]:
        t = text.replace("\n", " ").strip()
        if not t:
            return []
        out, buf = [], ""
        for s in self._ms([t]):
            buf = f"{buf} {s}" if buf else s
            if not buf.endswith(";"):
                out.append(buf)
                buf = ""
        if buf:
            out.append(buf)
        return out

    def close(self):
        self._ms.close()


def locate(text: str, start: int, end: int, pieces: list[str]) -> list[tuple[int, int]] | None:
    """Map splitter output back to char spans in text[start:end], ignoring whitespace."""
    spans = []
    i = start
    for p in pieces:
        q = re.sub(r"\s+", "", p)
        while i < end and text[i].isspace():
            i += 1
        s0 = i
        k = 0
        while k < len(q) and i < end:
            if text[i].isspace():
                i += 1
                continue
            if text[i] != q[k]:
                return None
            i += 1
            k += 1
        if k < len(q):
            return None
        spans.append((s0, i))
    while i < end and text[i].isspace():
        i += 1
    return spans if i == end else None


OPEN_D, CLOSE_D = "“", "”"
OPEN_S, CLOSE_S = "‘", "’"


def find_quotes(text: str, start: int, end: int) -> list[dict]:
    """Quote spans in text[start:end]; nesting via a stack; ``’`` is a closing
    quote only if a ‘ is open and it is not word-internal (don’t, o’)."""
    out: list[dict] = []
    stack: list[tuple[str, int]] = []
    straight_open: int | None = None
    for i in range(start, end):
        c = text[i]
        prev = text[i - 1] if i > start else " "
        nxt = text[i + 1] if i + 1 < end else " "
        if c == OPEN_D or (c == OPEN_S and not prev.isalnum()):
            stack.append((c, i))
        elif c == CLOSE_D or (c == CLOSE_S and not (prev.isalnum() and nxt.isalnum())
                              and any(m == OPEN_S for m, _ in stack)):
            want = OPEN_D if c == CLOSE_D else OPEN_S
            if any(m == want for m, _ in stack):
                while stack:
                    m, s = stack.pop()
                    if m == want:
                        out.append({"start": s, "end": i + 1, "mark": m, "level": len(stack), "closed": True})
                        break
                    out.append({"start": s, "end": i, "mark": m, "level": len(stack), "closed": False})
            elif c == CLOSE_D:
                # closing without an opening in this block (continued quote)
                out.append({"start": start, "end": i + 1, "mark": "”", "level": 0, "closed": True, "opened": False})
        elif c == '"':
            if straight_open is None:
                straight_open = i
            else:
                out.append({"start": straight_open, "end": i + 1, "mark": '"', "level": 0, "closed": True})
                straight_open = None
    while stack:  # unclosed at block end: multi-paragraph speech continues
        m, s = stack.pop()
        out.append({"start": s, "end": end, "mark": m, "level": len(stack), "closed": False})
    if straight_open is not None:
        out.append({"start": straight_open, "end": end, "mark": '"', "level": 0, "closed": False})
    return sorted(out, key=lambda q: (q["start"], -q["end"]))


def build(html: str, splitter: Splitter, overrides: dict | None = None) -> dict:
    soup = BeautifulSoup(html, "lxml")
    raw, coverage = find_blocks(soup, overrides)
    # Gutenberg's own "There are several editions of this ebook..." note (an
    # h4 + table of editions at the top of the body) is catalogue boilerplate:
    # strip it like the licence, up to the next heading.
    stripped = 0
    if raw and raw[0].kind == "heading" and PG_EDITIONS_NOTE.match(" ".join(l["text"] for l in raw[0].lines)):
        k = 1
        while k < len(raw) and raw[k].kind != "heading":
            k += 1
        stripped, raw = k, raw[k:]

    # canonical text + block records
    parts: list[str] = []
    pos = 0
    blocks = []
    for b in raw:
        if parts:
            parts.append("\n\n")
            pos += 2
        line_spans = []
        for k, ln in enumerate(b.lines):
            if k:
                parts.append("\n")
                pos += 1
            line_spans.append({"char_start": pos, "char_end": pos + len(ln["text"]), "indent": ln["indent"]})
            parts.append(ln["text"])
            pos += len(ln["text"])
        blocks.append({
            "kind": b.kind, "source": b.source, "html": html_path(b.el),
            "html_id": b.el.get("id"), "flags": b.flags, "verse_confidence": b.confidence,
            "char_start": line_spans[0]["char_start"], "char_end": line_spans[-1]["char_end"],
            "lines": line_spans if (b.verse or len(line_spans) > 1) else None,
        })
    text = "".join(parts)

    # divisions: a run of consecutive headings opens a new division
    # Headed divisions are numbered from 1; division 0 holds any text before the
    # first heading (and exists only if there is some).
    divisions: list[dict] = []
    para = 0
    prev_heading = False
    for i, b in enumerate(blocks):
        # Inside a CONTENTS division, text-pattern "headings" ("Chapter I. Start")
        # are contents entries while they come in a run of short blocks; a real
        # chapter heading is followed by prose.
        if (b["kind"] == "heading" and b["source"].endswith("~text") and divisions
                and any(TOC_HEADING.match(h) for h in divisions[-1]["heading"])):
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            if nxt is None or nxt["char_end"] - nxt["char_start"] < 100:
                b["kind"] = "toc"
        is_h = b["kind"] == "heading"
        if is_h and not prev_heading:
            num = divisions[-1]["division"] + 1 if divisions else 1
            divisions.append({"division": num, "heading": [], "first_block": i})
            para = 0
        elif not divisions:
            divisions.append({"division": 0, "heading": [], "first_block": i})
            para = 0
        if is_h:
            divisions[-1]["heading"].append(text[b["char_start"]:b["char_end"]])
        prev_heading = is_h
        para += 1
        b["division"] = divisions[-1]["division"]
        b["para"] = para
    for d in divisions:
        d["n_blocks"] = sum(1 for b in blocks if b["division"] == d["division"])
    # TOC-like divisions
    for d in divisions:
        if any(TOC_HEADING.match(h) for h in d["heading"]):
            for b in blocks:
                if b["division"] == d["division"] and b["kind"] in ("para", "lines", "verse"):
                    b["kind"] = "toc"
                    b["verse_confidence"] = None

    # poems: consecutive verse blocks within a division
    poems = []
    for i, b in enumerate(blocks):
        if b["kind"] != "verse":
            continue
        prev = blocks[i - 1] if i else None
        if prev is not None and prev["kind"] == "verse" and prev["division"] == b["division"]:
            poems[-1]["blocks"].append(i)
        else:
            title = None
            if prev is not None and prev["kind"] == "heading" and prev["division"] == b["division"]:
                title = text[prev["char_start"]:prev["char_end"]]
            poems.append({"poem": len(poems) + 1, "division": b["division"], "title": title, "blocks": [i]})
        b["poem"] = len(poems)
        b["stanza"] = len(poems[-1]["blocks"])
    for p in poems:
        p["char_start"] = blocks[p["blocks"][0]]["char_start"]
        p["char_end"] = blocks[p["blocks"][-1]]["char_end"]
        p["n_lines"] = sum(len(blocks[i]["lines"]) for i in p["blocks"])
        p["source"] = sorted({raw[i].source for i in p["blocks"]})
        p["confidence"] = min((raw[i].confidence for i in p["blocks"]),
                              key=["heuristic", "markup", "override"].index)

    # sentences: prose per block, verse per poem
    units = []
    done_poems = set()
    for i, b in enumerate(blocks):
        if b["kind"] == "verse":
            if b["poem"] in done_poems:
                continue
            done_poems.add(b["poem"])
            p = poems[b["poem"] - 1]
            units.append((p["char_start"], p["char_end"], "poem", b["poem"]))
        else:
            units.append((b["char_start"], b["char_end"], "block", i))
    sentences = []
    split_failures = 0
    for s0, s1, ukind, uid in units:
        pieces = splitter(text[s0:s1])
        spans = locate(text, s0, s1, pieces) if pieces else None
        if spans is None:
            split_failures += 1
            spans = [(s0, s1)]
        for a, z in spans:
            sentences.append({"char_start": a, "char_end": z, "unit": ukind,
                              "poem" if ukind == "poem" else "block": uid})
    block_starts = [b["char_start"] for b in blocks]
    for si, s in enumerate(sentences):
        bi = bisect.bisect_right(block_starts, s["char_start"]) - 1
        s["block"] = bi
        s["division"] = blocks[bi]["division"]
        s["para"] = blocks[bi]["para"]
    # sentence index within paragraph (prose) / within poem (verse)
    counter: dict = {}
    for s in sentences:
        key = ("poem", s["poem"]) if s["unit"] == "poem" else ("block", s["block"])
        counter[key] = counter.get(key, 0) + 1
        s["n"] = counter[key]

    # tokens
    sent_starts = [s["char_start"] for s in sentences]
    toks = {"surface": [], "start": [], "end": [], "punct": [], "sent": [], "block": [], "line": []}
    for bi, b in enumerate(blocks):
        lines = b["lines"] or [{"char_start": b["char_start"], "char_end": b["char_end"]}]
        for li, ln in enumerate(lines):
            seg = text[ln["char_start"]:ln["char_end"]]
            j = 0
            while j < len(seg):
                if seg[j].isspace():
                    j += 1
                    continue
                m = WORD.match(seg, j)
                k = m.end() if m else j + 1
                a, z = ln["char_start"] + j, ln["char_start"] + k
                toks["surface"].append(text[a:z])
                toks["start"].append(a)
                toks["end"].append(z)
                toks["punct"].append(0 if m else 1)
                toks["sent"].append(bisect.bisect_right(sent_starts, a) - 1)
                toks["block"].append(bi)
                toks["line"].append(li + 1 if b["lines"] else None)
                j = k

    quotes = []
    for bi, b in enumerate(blocks):
        for q in find_quotes(text, b["char_start"], b["char_end"]):
            q["block"] = bi
            quotes.append(q)

    return {
        "text": text, "divisions": divisions, "blocks": blocks, "poems": poems,
        "sentences": sentences, "tokens": toks, "quotes": quotes,
        "stats": {
            **{k: v for k, v in coverage.items() if k != "uncovered"},
            "uncovered_samples": coverage["uncovered"][:50],
            "n_uncovered": len(coverage["uncovered"]),
            "split_failures": split_failures, "stripped_pg_note_blocks": stripped,
            "n_blocks": len(blocks), "n_divisions": len(divisions), "n_poems": len(poems),
            "n_sentences": len(sentences), "n_tokens": len(toks["start"]),
            "n_words": len(toks["start"]) - sum(toks["punct"]),
            "n_quotes": len(quotes), "n_quotes_unclosed": sum(1 for q in quotes if not q["closed"]),
            "kinds": {k: sum(1 for b in blocks if b["kind"] == k) for k in sorted({b["kind"] for b in blocks})},
        },
    }


def _c(x) -> str:
    """TSV-safe cell."""
    return re.sub(r"[\t\r\n]+", " / ", str(x if x is not None else ""))


def write_review(rdir: Path, results: dict[int, dict]) -> None:
    rdir.mkdir(parents=True, exist_ok=True)
    with (rdir / "divisions.tsv").open("w") as f:
        f.write("text\tdivision\tn_blocks\theading\tfirst_text\n")
        for n, d in sorted(results.items()):
            for dv in d["divisions"]:
                b = d["blocks"][dv["first_block"]]
                first = next((d["text"][x["char_start"]:x["char_end"]] for x in d["blocks"][dv["first_block"]:]
                              if x["kind"] not in ("heading",)), "")[:80]
                f.write(f"{n}\t{dv['division']}\t{dv['n_blocks']}\t{_c(' | '.join(dv['heading'])[:100])}\t{_c(first)}\n")
    with (rdir / "poems.tsv").open("w") as f:
        f.write("text\tpoem\tdivision\tconfidence\tsource\tstanzas\tlines\ttitle\tfirst_line\tlast_line\thtml\n")
        for n, d in sorted(results.items()):
            for p in d["poems"]:
                fb, lb = d["blocks"][p["blocks"][0]], d["blocks"][p["blocks"][-1]]
                fl = d["text"][fb["lines"][0]["char_start"]:fb["lines"][0]["char_end"]]
                ll = d["text"][lb["lines"][-1]["char_start"]:lb["lines"][-1]["char_end"]]
                f.write(f"{n}\t{p['poem']}\t{p['division']}\t{p['confidence']}\t{','.join(p['source'])}\t"
                        f"{len(p['blocks'])}\t{p['n_lines']}\t{_c((p['title'] or '')[:60])}\t{_c(fl[:70])}\t"
                        f"{_c(ll[:70])}\t{fb['html']}\n")
    with (rdir / "lines_blocks.tsv").open("w") as f:
        f.write("text\tdivision\tpara\tkind\tn_lines\tlines\thtml\n")
        for n, d in sorted(results.items()):
            for b in d["blocks"]:
                if b["kind"] == "lines" or "multiline" in b["flags"]:
                    ls = [d["text"][l["char_start"]:l["char_end"]] for l in b["lines"]]
                    f.write(f"{n}\t{b['division']}\t{b['para']}\t{b['kind']}\t{len(ls)}\t{_c(' / '.join(ls)[:200])}\t{b['html']}\n")
    write_poems_html(rdir / "poems.html", results)
    with (rdir / "summary.tsv").open("w") as f:
        cols = ["n_divisions", "n_blocks", "n_poems", "n_sentences", "n_words", "n_quotes",
                "n_quotes_unclosed", "split_failures", "chars_total", "chars_covered", "n_uncovered"]
        f.write("text\t" + "\t".join(cols) + "\tkinds\tuncovered_samples\n")
        for n, d in sorted(results.items()):
            st = d["stats"]
            f.write(f"{n}\t" + "\t".join(str(st[c]) for c in cols) + "\t" +
                    json.dumps(st["kinds"]) + "\t" + _c(" | ".join(st["uncovered_samples"][:5])) + "\n")


def write_poems_html(path: Path, results: dict[int, dict]) -> None:
    """Every poem rendered as parsed (stanzas, lines, indents); heuristic ones
    first within each text. The html path is what goes into the overrides file."""
    import html as H

    out = ["<!doctype html><meta charset=utf-8><title>Stage 1 poems</title><style>",
           "body{font:14px sans-serif;max-width:70em;margin:auto}"
           ".poem{border:1px solid #ccc;margin:.6em 0;padding:.4em .8em}"
           ".heuristic{border-color:#d80;background:#fff8ee}.markup{border-color:#8a8}"
           ".st{margin:.3em 0 .5em}.ln{white-space:pre-wrap}.meta{color:#666;font-size:12px}"
           "code{font-size:11px}</style>",
           "<h1>Stage 1 poem parses</h1><p>Orange = verse by &lt;br&gt; heuristic (check these), "
           "green = verse markup. To override, add <code>{\"pgN\": {\"html path\": \"lines\"}}</code> "
           "to <code>data/stage1_overrides.json</code>.</p><ul>"]
    for n, d in sorted(results.items()):
        nh = sum(1 for p in d["poems"] if p["confidence"] == "heuristic")
        out.append(f"<li><a href='#{n}'>{n}</a>: {len(d['poems'])} poems ({nh} heuristic)</li>")
    out.append("</ul>")
    for n, d in sorted(results.items()):
        if not d["poems"]:
            continue
        T = d["text"]
        out.append(f"<h2 id='{n}'>{n}</h2>")
        for p in sorted(d["poems"], key=lambda p: (p["confidence"] != "heuristic", p["poem"])):
            out.append(f"<div class='poem {p['confidence']}'><div class=meta>poem {p['poem']} · division "
                       f"{p['division']} · {p['confidence']} · {','.join(p['source'])} · "
                       f"{len(p['blocks'])} stanzas, {p['n_lines']} lines"
                       + (f" · title: <b>{H.escape(p['title'])}</b>" if p["title"] else "") + "</div>")
            for bi in p["blocks"]:
                b = d["blocks"][bi]
                out.append(f"<div class=st><div class=meta><code>{H.escape(b['html'])}</code></div>")
                for ln in b["lines"]:
                    ind = ln.get("indent") or ""
                    m = re.match(r"i(\d+)|pre:(\d+)", ind)
                    em = (int(m.group(1) or 0) / 2 if m and m.group(1) else int(m.group(2)) / 2 if m else 0)
                    out.append(f"<div class=ln style='margin-left:{min(em, 12)}em'>"
                               f"{H.escape(T[ln['char_start']:ln['char_end']])}</div>")
                out.append("</div>")
            out.append("</div>")
    path.write_text("\n".join(out), encoding="utf-8")


PLAIN_DIR = Path(__file__).resolve().parents[2] / "data" / "texts"


def select_sources(a, gdir: Path) -> list[tuple[str, Path]]:
    """(text_id, source file): Gutenberg HTML for pg<N>, data/texts/<id>.txt otherwise."""
    if a.id:
        ids = {f"pg{n}" for n in a.id}
    else:
        ids = set()
        for line in a.inventory.read_text().splitlines():
            r = json.loads(line)
            if a.work and r["work"] not in a.work:
                continue
            if a.reading and r["librivox_slug"] not in a.reading:
                continue
            ids.update(r.get("texts") or [f"pg{g}" for g in r["gutenberg_ids"]])
    files = plain_files()
    return [(t, gdir / f"{t}-images.html" if t.startswith("pg") else files.get(t, PLAIN_DIR / f"{t}.txt"))
            for t in sorted(ids, key=lambda t: (not t.startswith("pg"), int(t[2:]) if t.startswith("pg") else 0, t))]


def plain_files() -> dict[str, Path]:
    """text_id -> plain-text file, from data/source_overrides.json (`file`)."""
    p = Path(__file__).resolve().parents[2] / "data" / "source_overrides.json"
    if not p.exists():
        return {}
    o = json.loads(p.read_text())
    entries = list(o.get("works", {}).values()) + list(o.get("readings", {}).values())
    return {e["text_id"]: Path(e["file"]) for e in entries if e.get("text_id") and e.get("file")}


def plaintext_to_html(txt: str) -> str:
    """A plain-text poem as Gutenberg-style HTML, so it goes through the same parser.

    A first block of one short line is the title; every other blank-line-separated
    block is a stanza (verse markup, so confidence 'markup').
    """
    import html as H

    blocks = [b.strip("\n") for b in re.split(r"\n\s*\n", txt.strip()) if b.strip()]
    body = []
    if blocks and "\n" not in blocks[0] and len(blocks[0]) <= 80 and len(blocks) > 1:
        body.append(f"<h2>{H.escape(blocks[0].strip())}</h2>")
        blocks = blocks[1:]
    body.append('<div class="poem">')
    for b in blocks:
        lines = "".join(f'<span class="i0">{H.escape(l.strip())}<br></span>\n' for l in b.splitlines() if l.strip())
        body.append(f'<div class="stanza">\n{lines}</div>')
    body.append("</div>")
    return ('<html><body><div id="pg-start-separator"></div>\n' + "\n".join(body) +
            '\n<div id="pg-end-separator"></div></body></html>')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", type=Path, default=Path("/out/inventory.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("/out"))
    ap.add_argument("--id", type=int, action="append")
    ap.add_argument("--work", action="append")
    ap.add_argument("--reading", action="append")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    (a.out / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(a.out / "logs" / "stage1.log"), logging.StreamHandler(sys.stderr)],
    )
    gdir = a.out / "gutenberg"
    manifest = {json.loads(l)["id"]: json.loads(l) for l in (gdir / "manifest.jsonl").read_text().splitlines()}
    odir = a.out / "stage1"
    odir.mkdir(parents=True, exist_ok=True)
    ov_path = Path(__file__).resolve().parents[2] / "data" / "stage1_overrides.json"
    overrides_all = json.loads(ov_path.read_text()) if ov_path.exists() else {}
    ov_sha = hashlib.sha256(ov_path.read_bytes()).hexdigest()[:12] if ov_path.exists() else None
    splitter = Splitter()
    results = {}
    try:
        for tid, src in select_sources(a, gdir):
            dst = odir / f"{tid}.json.gz"
            if not src.exists():
                log.error("%s: no source at %s", tid, src)
                continue
            sha = hashlib.sha256(src.read_bytes()).hexdigest()
            if dst.exists() and not a.force:
                with gzip.open(dst, "rt") as f:
                    d = json.load(f)
                if (d["meta"]["source_sha256"] == sha and d["meta"]["version"] == VERSION
                        and d["meta"].get("overrides_sha") == ov_sha):
                    results[tid] = d
                    continue
            raw = src.read_text(encoding="utf-8")
            html = raw if src.suffix == ".html" else plaintext_to_html(raw)
            d = build(html, splitter, overrides_all.get(tid, {}))
            n = int(tid[2:]) if tid.startswith("pg") else None
            d["meta"] = {"text": tid, "gutenberg_id": n, "source": str(src), "source_sha256": sha,
                         "manifest_sha256": manifest.get(n, {}).get("sha256") if n else None, "version": VERSION,
                         "splitter": "moses-en+rejoin-semicolon", "overrides_sha": ov_sha}
            tmp = dst.with_suffix(".tmp")
            with gzip.open(tmp, "wt", compresslevel=6) as f:
                json.dump(d, f, ensure_ascii=False)
            tmp.replace(dst)
            st = d["stats"]
            log.info("%s: %d div, %d blocks, %d poems, %d sents, %d words, coverage %.4f, uncovered %d, split fail %d",
                     tid, st["n_divisions"], st["n_blocks"], st["n_poems"], st["n_sentences"], st["n_words"],
                     st["chars_covered"] / max(st["chars_total"], 1), st["n_uncovered"], st["split_failures"])
            results[tid] = d
    finally:
        splitter.close()
    write_review(a.out / "review" / "stage1", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
