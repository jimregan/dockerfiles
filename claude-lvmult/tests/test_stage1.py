import difflib
import re
from pathlib import Path

import pytest

from librivox_mult import stage1_text as s1

FIX = Path(__file__).parent / "fixtures" / "text"


def page(body: str) -> str:
    return f"""<html><head><title>x</title></head><body>
<section id="pg-header"><p>Project Gutenberg licence header</p>
<div id="pg-start-separator">*** START OF THE PROJECT GUTENBERG EBOOK X ***</div></section>
{body}
<section id="pg-footer"><div id="pg-end-separator">*** END OF THE PROJECT GUTENBERG EBOOK X ***</div>
<p>Licence footer text.</p></section></body></html>"""


@pytest.fixture(scope="module")
def splitter():
    sp = s1.Splitter()
    yield sp
    sp.close()


def blocks_text(d):
    return [(b["kind"], d["text"][b["char_start"]:b["char_end"]]) for b in d["blocks"]]


def test_boilerplate_divisions_paragraphs(splitter):
    d = s1.build(page("""
<p>A Novel</p>
<h2>CHAPTER I.</h2><h3>The Beginning</h3>
<p>Mr. Smith arrived at 3 p.m. on the 4th. He sat down.</p>
<p><span class="pagenum">[12]</span>“Hello,” said he.</p>
<h2>CHAPTER II.</h2>
<p>The end.</p>"""), splitter)
    assert "licence" not in d["text"].lower() and "START OF" not in d["text"]
    assert [dv["division"] for dv in d["divisions"]] == [0, 1, 2]
    assert d["divisions"][1]["heading"] == ["CHAPTER I.", "The Beginning"]
    paras = [(b["division"], b["para"], b["kind"]) for b in d["blocks"]]
    assert paras == [(0, 1, "para"), (1, 1, "heading"), (1, 2, "heading"), (1, 3, "para"),
                     (1, 4, "para"), (2, 1, "heading"), (2, 2, "para")]
    assert "[12]" not in d["text"]  # page numbers are not text
    sents = [d["text"][s["char_start"]:s["char_end"]] for s in d["sentences"]]
    assert "Mr. Smith arrived at 3 p.m. on the 4th." in sents  # abbreviations
    assert d["stats"]["chars_total"] == d["stats"]["chars_covered"]


def test_invariants_offsets_and_sentence_coverage(splitter):
    d = s1.build((FIX / "pg40745-images.html").read_text(encoding="utf-8"), splitter)
    T = d["text"]
    tk = d["tokens"]
    assert all(T[a:z] == s for s, a, z in zip(tk["surface"], tk["start"], tk["end"]))
    # every non-space character is in exactly one sentence
    cover = [0] * len(T)
    for s in d["sentences"]:
        for i in range(s["char_start"], s["char_end"]):
            cover[i] += 1
    assert all((c == 1) if not T[i].isspace() else c <= 1 for i, c in enumerate(cover))
    # every non-space character is in exactly one token
    tcov = [0] * len(T)
    for a, z in zip(tk["start"], tk["end"]):
        for i in range(a, z):
            tcov[i] += 1
    assert all((c == 1) if not T[i].isspace() else c == 0 for i, c in enumerate(tcov))
    assert d["stats"]["split_failures"] == 0


def test_crocodile_matches_jims_prototype(splitter):
    """Regression oracle: Jim's Moses+rejoin splits of The Crocodile (pg40745)."""
    d = s1.build((FIX / "pg40745-images.html").read_text(encoding="utf-8"), splitter)
    norm = lambda x: re.sub(r"\s+", " ", x).strip()  # noqa: E731
    ours = [norm(d["text"][s["char_start"]:s["char_end"]]) for s in d["sentences"]]
    oracle = [norm(line.rstrip("\n").split("\t")[2])
              for f in sorted(FIX.glob("output_0*.tsv"))
              for line in list(f.open(encoding="utf-8"))[1:]]
    st = ours.index(oracle[0])
    sm = difflib.SequenceMatcher(None, oracle, ours[st:st + len(oracle) + 10], autojunk=False)
    assert sum(b.size for b in sm.get_matching_blocks()) == len(oracle)


def test_verse_markup_and_stanzas(splitter):
    d = s1.build(page("""
<h2>A POEM</h2>
<div class="poem"><div class="stanza">
<span class="i0">The first line is here,<br></span>
<span class="i2">The second line is near.<br></span></div>
<div class="stanza"><span class="i0">Another stanza starts;<br></span>
<span class="i0">and ends the poem here.<br></span></div></div>
<pre>
   Pre verse one,
   pre verse two.

   Pre stanza two
</pre>"""), splitter)
    verse = [b for b in d["blocks"] if b["kind"] == "verse"]
    assert len(verse) == 4 and len(d["poems"]) == 1
    p = d["poems"][0]
    assert p["title"] == "A POEM" and p["n_lines"] == 7 and p["confidence"] == "markup"
    assert [b["stanza"] for b in verse] == [1, 2, 3, 4]
    assert verse[0]["lines"][1]["indent"] == "i2"
    T = d["text"]
    assert T[verse[0]["lines"][0]["char_start"]:verse[0]["lines"][0]["char_end"]] == "The first line is here,"
    # verse sentences cross line and stanza boundaries
    sents = [T[s["char_start"]:s["char_end"]] for s in d["sentences"] if s["unit"] == "poem"]
    assert sents[0] == "The first line is here,\nThe second line is near."
    assert sents[1].startswith("Another stanza starts;") and sents[1].endswith("and ends the poem here.")


def test_br_blocks_verse_vs_lines(splitter):
    d = s1.build(page("""
<p>Twinkle, twinkle, little star,<br>How I wonder what you are!<br><br>Up above the world so high,<br>Like a diamond in the sky.</p>
<p>“Yours respectfully,<br>Anne Shirley.”</p>
<p>BOSTON<br>LITTLE, BROWN, AND COMPANY</p>"""), splitter)
    kinds = [(b["kind"], b.get("verse_confidence")) for b in d["blocks"]]
    assert kinds == [("verse", "heuristic"), ("verse", "heuristic"), ("lines", None), ("lines", None)]


def test_overrides(splitter):
    html = page("<p>One line here,<br>another line there.</p>")
    d = s1.build(html, splitter)
    assert d["blocks"][0]["kind"] == "verse"
    path = d["blocks"][0]["html"]
    d2 = s1.build(html, splitter, {path: "lines"})
    assert d2["blocks"][0]["kind"] == "lines" and not d2["poems"]


def test_tei_and_text_headings(splitter):
    d = s1.build(page("""
<div class="tei tei-head">WELAND'S SWORD</div>
<div class="tei tei-lg"><div class="tei tei-l">Line one of verse,</div><div class="tei tei-l">line two.</div></div>
<p class="ph2">1. The Horror in Clay.</p>
<p>Text.</p>
<p>CHAPTER IV</p><p>Part of it was true.</p>"""), splitter)
    k = blocks_text(d)
    assert k[0] == ("heading", "WELAND'S SWORD")
    assert k[1] == ("verse", "Line one of verse,\nline two.")
    assert k[2][0] == "heading" and k[4] == ("heading", "CHAPTER IV")
    assert k[5] == ("para", "Part of it was true.")


def test_quotes():
    T = "“What do you mean by ‘once’? Don’t,” he said; “and then—"
    q = s1.find_quotes(T, 0, len(T))
    spans = [(T[x["start"]:x["end"]], x["level"], x["closed"]) for x in q]
    assert ("“What do you mean by ‘once’? Don’t,”", 0, True) in spans
    assert ("‘once’", 1, True) in spans
    assert ("“and then—", 0, False) in spans
    assert len(q) == 3  # the apostrophe in Don’t is not a quote


def test_locate_whitespace_insensitive():
    T = "Hello  there.\nGeneral   Kenobi!"
    assert s1.locate(T, 0, len(T), ["Hello there.", "General Kenobi!"]) == [(0, 13), (14, len(T))]
    assert s1.locate(T, 0, len(T), ["Hello there."]) is None  # must consume everything


def test_pg_editions_note_stripped_and_toc(splitter):
    d = s1.build(page("""
<h4>There are several editions of this ebook in the Project Gutenberg collection. Various characteristics.</h4>
<table><tr><td><a href="https://www.gutenberg.org/ebooks/46">46</a></td><td>(First edition)</td></tr></table>
<h1>A BOOK</h1><h2>Contents</h2><p>Chapter I. Start</p>
<h2>CHAPTER I.</h2><p>Text.</p>"""), splitter)
    assert "several editions" not in d["text"]
    assert d["stats"]["stripped_pg_note_blocks"] == 3  # heading + 2 table cells
    assert d["divisions"][0]["heading"] == ["A BOOK", "Contents"]
    assert [b["kind"] for b in d["blocks"]] == ["heading", "heading", "toc", "heading", "para"]


@pytest.mark.parametrize("lines", [
    ["“Your friend,", "“Dracula.”"], ["“Ever your loving", "“Lucy."], ["“Believe me,", "“Sister Agatha."],
    ["“We are, dear Sirs,", "“Samuel F. Billington & Son.”"], ["Copyright, 1897, in the United States", "THE PRESS"]])
def test_letter_closings_are_not_verse(lines):
    assert not s1.looks_like_verse(lines)


def test_short_verse_still_verse():
    assert s1.looks_like_verse(["God bless you, merry gentleman!", "May nothing you dismay!"])


def test_plaintext_poem(splitter):
    txt = "No Second Troy\n\nWhy should I blame her that she filled my days\nWith misery, or that she would of late\n\nWhy, what could she have done, being what she is?\nWas there another Troy for her to burn?\n"
    d = s1.build(s1.plaintext_to_html(txt), splitter)
    assert [b["kind"] for b in d["blocks"]] == ["heading", "verse", "verse"]
    assert len(d["poems"]) == 1 and d["poems"][0]["title"] == "No Second Troy"
    assert d["poems"][0]["n_lines"] == 4 and d["poems"][0]["confidence"] == "markup"
