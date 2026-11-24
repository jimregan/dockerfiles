from pathlib import Path

from librivox_mult import stage2_headers as s2
from librivox_mult.librivox_page import parse_page
from librivox_mult.spoken import number_variants, roman_to_int, spoken_words

FIX = Path(__file__).parent / "fixtures" / "librivox_pages"


def test_spoken_words():
    assert spoken_words("Chapter XII. Mr. Fogg & co") == ["chapter", "twelve", "mister", "fogg", "and", "co"]
    assert spoken_words("Volume 3, 1848") == ["volume", "three", "eighteen", "forty", "eight"]
    assert spoken_words("I am") == ["i", "am"]  # a bare I is not a numeral
    assert spoken_words("Æsop’s Fables") == ["aesop's", "fables"]
    assert roman_to_int("XIV") == 14 and roman_to_int("hello") is None


def test_number_variants():
    assert number_variants(102) == ["one hundred and two", "one hundred two", "one o two",
                                    "number one hundred and two", "number one hundred two"]
    assert number_variants(16) == ["sixteen", "number sixteen"]


def test_titles_and_sections():
    assert s2.clean_title("Around the World in Eighty Days (Dramatic Reading)") == "Around the World in Eighty Days"
    assert s2.clean_title("The Secret Garden (version 3)") == "The Secret Garden"
    assert s2.title_variants("Aesop's Fables, Volume 07 (Fables 151-175)")[:2] == [
        "Aesop's Fables, Volume 07 (Fables 151-175)", "Aesop's Fables"]
    assert s2.title_numbers("Part I, Chapter 05") == [("part", 1), ("chapter", 5)]
    assert s2.section_word("Stave 4: The Last of the Spirits")[0] == "stave"
    assert s2.clean_section_title("22 - Under the Willow-Tree") == "Under the Willow-Tree"
    assert s2.translator("Jules Verne (1828 - 1905) | Translated by George Makepeace Towle (1841 - 1893)") \
        == "George Makepeace Towle"


def words(text: str, t0: float = 0.0, step: float = 0.4):
    return [{"w": w, "l": w, "start": t0 + i * step, "end": t0 + i * step + 0.3}
            for i, w in enumerate(text.split())]


READING = {"title": "Around the World in Eighty Days", "authors": ["Jules Verne"],
           "credits_line": "", "readers": [{"name": "Kim Reader", "id": 1}], "cast": []}


def test_match_header_footer_with_asr_quirks():
    body = " ".join(["word"] * 400)
    hyp = ("chapter fifteen of around the world in eighty days by jules verne this is a libri vox recording "
           "all libri vox recordings are in the public domain for more information or to volunteer please visit "
           "libri vox dot org recording by kim reeder in dublin " + body +
           " end of chapter fifteen recorded by kim reader")
    w = words(hyp)
    sec = {"section": "15", "title": "Chapter 15", "readers": [{"name": "Kim Reader", "id": 1}]}
    els = s2.templates_for(READING, sec, is_last=False)
    r = s2.match_section(els, w, w[-1]["end"] + 1, cast_section=False)
    kinds = [(m["kind"], m["role"]) for m in r["matches"]]
    assert ("intro", "header") in kinds and ("disclaimer_long", "header") in kinds
    assert ("credit", "header") in kinds and ("end", "footer") in kinds and ("credit", "footer") in kinds
    intro = next(m for m in r["matches"] if m["kind"] == "intro")
    assert intro["w0"] == 0 and intro["hyp"].endswith("jules verne")  # not swallowed by the disclaimer
    # the ad-lib after the credit ("in dublin") is not a template match; it is shown in
    # header_after (Stage 3 decides where the book starts)
    assert r["header_after"].startswith("in dublin")
    fl = s2.section_flags({"matches": r["matches"], "templates": els, **r})
    assert fl["disclaimer"] == "long" and not fl["flags"]


def test_older_format_credit_and_footer():
    hyp = ("if by rudyard kipling read for libri vox dot org by beth thomas " + " ".join(["word"] * 300) +
           " end of poem this libri vox recording is in the public domain")
    w = words(hyp)
    sec = {"section": "01", "title": "If - Read by BT", "readers": [{"name": "Beth Thomas", "id": 2}]}
    reading = {**READING, "title": "If", "authors": ["Rudyard Kipling"]}
    els = s2.templates_for(reading, sec, is_last=False)
    r = s2.match_section(els, w, w[-1]["end"] + 1, cast_section=False)
    fl = s2.section_flags({"matches": r["matches"], "templates": els, **r})
    assert fl["disclaimer"] == "credit+footer" and "no_disclaimer" not in fl["flags"]


def test_cast_role_and_spoken_name():
    reading = {**READING, "cast": [{"roles": ["Oliver Twist"], "readers": [{"name": "SweetPea", "id": 7356}]},
                                   {"roles": ["Fagin"], "readers": [{"name": "Adamski", "id": 7326}]}]}
    hyp = ("cast of characters this is a libri vox recording oliver twist read by rachel fagin read by adam "
           + " ".join(["word"] * 50))
    w = words(hyp)
    sec = {"section": "00", "title": "Cast of Characters", "readers": []}
    els = s2.templates_for(reading, sec, is_last=False)
    r = s2.match_section(els, w, w[-1]["end"] + 1, cast_section=True)
    cast = {els[m["element"]]["character"]: m for m in r["matches"] if m["kind"] == "cast"}
    assert set(cast) == {"Oliver Twist", "Fagin"}
    assert cast["Oliver Twist"]["after_verb"] == "rachel"


def test_reader_pool_not_a_character():
    p = parse_page((FIX / "lv10387_oliver_dramatic_headerless_cast.html").read_text(encoding="utf-8"))
    pools = [c for c in p["cast"] if c.get("pool")]
    assert len(pools) == 1 and len(pools[0]["readers"]) > 20
    els = s2.templates_for({**READING, "cast": p["cast"]}, {"section": "00", "title": "Cast of Characters",
                                                            "readers": []}, False)
    assert not any(e["kind"] == "cast" and e["character"].lower().startswith("minor") for e in els)
