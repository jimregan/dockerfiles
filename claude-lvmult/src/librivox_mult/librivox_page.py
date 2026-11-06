"""Parse a saved LibriVox project page (``index.html``).

Pure function of the HTML; no network. Extracts project metadata, the
section table, readers, the online-text (usually Gutenberg) link and, for
dramatic readings, the cast list from the description.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from urllib.parse import unquote, urlparse

from bs4 import BeautifulSoup, NavigableString, Tag

READER_RE = re.compile(r"librivox\.org/reader/(\d+)")
GUTENBERG_RE = re.compile(
    r"gutenberg\.(?:org|net|ca|net\.au)/(?:etext/|ebooks/|files/|cache/epub/)?(\d+)",
    re.I,
)
RSS_RE = re.compile(r"librivox\.org/rss/(\d+)")
ARCHIVE_RE = re.compile(r"archive\.org/details/([^/?#]+)")


def _text(node) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip() if node else ""


def _reader(a: Tag) -> dict:
    m = READER_RE.search(a.get("href", ""))
    return {"name": _text(a), "id": int(m.group(1)) if m else None}


def _gutenberg_id(url: str) -> int | None:
    m = GUTENBERG_RE.search(url)
    return int(m.group(1)) if m else None


def mp3_basename(url: str) -> str:
    return PurePosixPath(unquote(urlparse(url).path)).name


def parse_cast(desc: Tag) -> list[dict]:
    """Parse a ``Cast List:`` block from the description paragraph.

    LibriVox writes it as ``Role A, Role B: <a reader>`` lines separated by
    ``<br/>``. Returns ``[{"roles": [...], "readers": [{name,id}], "raw": str}]``.
    Lines after the cast list that credit editors/proof-listeners are skipped.
    """
    # Flatten the description into lines split on <br>, keeping reader links.
    lines: list[list] = [[]]
    for node in desc.descendants:
        if isinstance(node, Tag) and node.name == "br":
            lines.append([])
        elif isinstance(node, Tag) and node.name == "a":
            lines[-1].append(node)
        elif isinstance(node, NavigableString) and not (
            node.parent is not None and node.parent.name == "a"
        ):
            lines[-1].append(str(node))
    header_re = re.compile(r"(?i)^\s*cast(?: list(?:s|ing)?)?\s*:\s*")
    skip_re = re.compile(
        r"(?i)^(audio )?(edited|editing|proof|dramatic reading coordinat|coordinat|"
        r"produced|music|sound|meta|book coordinator|bc\b|cover|summary|"
        r"(?:\w+ )?listened|note\b)"
    )
    group = None
    has_header = any(
        header_re.match("".join(p if isinstance(p, str) else _text(p) for p in parts))
        for parts in lines
    )
    in_cast = not has_header  # headerless: accept any "Roles: <a>Reader</a>" line
    cast: list[dict] = []
    for parts in lines:
        raw = "".join(p if isinstance(p, str) else _text(p) for p in parts).strip()
        if not raw:
            continue
        if header_re.match(raw):
            in_cast = True
            raw = header_re.sub("", raw)
            if not raw:
                continue
        if not in_cast:
            continue
        # numbered sub-casts in collections: "1. The Masked Marriage from ..."
        m = re.match(r"^(\d+)\.\s+(.*)", raw)
        if m and not any(isinstance(p, Tag) for p in parts):
            group = raw
            continue
        if skip_re.match(raw) or ":" not in raw:
            continue
        roles_part, reader_part = raw.split(":", 1)
        roles = [r.strip() for r in re.split(r",|/|;|\band\b", roles_part) if r.strip()]
        readers = [_reader(p) for p in parts if isinstance(p, Tag) and _text(p)]
        if not readers:
            if not has_header:
                continue  # headerless mode needs a link to count as a cast line
            name = reader_part.strip()
            readers = [{"name": name, "id": None}] if name else []
        entry = {"roles": roles, "readers": readers, "raw": raw}
        if group:
            entry["group"] = group
        cast.append(entry)
    # headerless mode is a heuristic: require several lines to avoid stray hits
    if not has_header and len(cast) < 3:
        return []
    return cast


def parse_page(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    out: dict = {}
    page = soup.select_one("div.page.book-page") or soup
    out["title"] = _text(page.find("h1"))
    authors = []
    for p in page.select("p.book-page-author"):
        for a in p.find_all("a"):
            authors.append(re.sub(r"\s*\(.*?\)\s*$", "", _text(a)))
    out["authors"] = authors
    # translators etc. appear as "Translated by X" text in book-page-author lines
    out["credits_line"] = " | ".join(_text(p) for p in page.select("p.book-page-author"))

    genres, language = [], None
    for p in page.select("p.book-page-genre"):
        t = _text(p)
        if t.lower().startswith("genre"):
            genres = [g.strip() for g in t.split(":", 1)[1].split(",") if g.strip()]
        elif t.lower().startswith("language"):
            language = t.split(":", 1)[1].strip()
    out["genres"] = genres
    out["language"] = language

    desc = page.select_one("p.description") or page.select_one("div.description")
    out["description"] = _text(desc)
    out["cast"] = parse_cast(desc) if desc is not None else []

    # product details: <dl class="product-details"><dt>..</dt><dd>..</dd>
    details: dict = {}
    for dl in soup.select("dl.product-details"):
        for dt in dl.find_all("dt"):
            dd = dt.find_next_sibling("dd")
            key = _text(dt).rstrip(":").strip().lower().replace(" ", "_")
            links = [_reader(a) for a in dd.find_all("a")] if dd else []
            details[key] = {
                "text": _text(dd),
                "people": [r for r in links if r["id"] is not None],
            }
    out["running_time"] = details.get("running_time", {}).get("text")
    out["catalog_date"] = details.get("catalog_date", {}).get("text")
    out["read_by"] = details.get("read_by", {}).get("people", [])
    out["read_by_text"] = details.get("read_by", {}).get("text")
    out["book_coordinator"] = details.get("book_coordinator", {}).get("people", [])

    hrefs = [a.get("href", "") for a in soup.find_all("a")]
    online = [
        {"text": _text(a), "url": a.get("href", "")}
        for a in soup.find_all("a")
        if _text(a).lower().startswith("online text")
    ]
    out["online_text"] = online
    gids = []
    for o in online:
        g = _gutenberg_id(o["url"])
        if g is not None and g not in gids:
            gids.append(g)
    out["gutenberg_ids"] = gids
    m = next((RSS_RE.search(h) for h in hrefs if RSS_RE.search(h)), None)
    out["librivox_id"] = int(m.group(1)) if m else None
    m = next((ARCHIVE_RE.search(h) for h in hrefs if ARCHIVE_RE.search(h)), None)
    out["archive_id"] = m.group(1) if m else None

    sections = []
    table = soup.select_one("table.chapter-download")
    if table is not None:
        heads = [_text(th).lower() for th in table.select("thead th")]
        for tr in table.select("tbody tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            cells = dict(zip(heads, tds))
            sec_td = cells.get("section", tds[0])
            num = re.sub(r"(?i)\bplay\b", "", _text(sec_td)).strip()
            chap_td = cells.get("chapter", tds[1] if len(tds) > 1 else None)
            name_a = chap_td.find("a", class_="chapter-name") if chap_td else None
            mp3_urls = [
                a.get("href") for a in tr.find_all("a") if a.get("href", "").endswith(".mp3")
            ]
            reader_td = cells.get("reader")
            readers = [
                _reader(a) for a in (reader_td.find_all("a") if reader_td else [])
            ]
            sec = {
                "section": num,
                "title": _text(name_a or chap_td),
                "mp3_urls": mp3_urls,
                "mp3_names": [mp3_basename(u) for u in mp3_urls],
                "readers": [r for r in readers if r["id"] is not None or r["name"]],
                "reader_text": _text(reader_td) if reader_td else None,
                "time": _text(cells.get("time")) or None,
            }
            # collections have Author / Source columns
            for extra in ("author", "source"):
                if extra in cells:
                    td = cells[extra]
                    sec[extra] = _text(td)
                    if extra == "source":
                        links = [a.get("href", "") for a in td.find_all("a")]
                        sec["source_urls"] = links
                        sec["gutenberg_ids"] = [
                            g for g in (_gutenberg_id(u) for u in links) if g is not None
                        ]
            sections.append(sec)
        out["section_columns"] = heads
    out["sections"] = sections
    return out
