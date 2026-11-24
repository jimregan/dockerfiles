import pytest

from librivox_mult import stage4_norm as s4
from librivox_mult.stage1_text import WORD


def doc(text):
    S, A, Z, P = [], [], [], []
    j = 0
    while j < len(text):
        if text[j].isspace():
            j += 1
            continue
        m = WORD.match(text, j)
        k = m.end() if m else j + 1
        S.append(text[j:k]); A.append(j); Z.append(k); P.append(0 if m else 1)  # noqa: E702
        j = k
    return {"text": text, "tokens": {"surface": S, "start": A, "end": Z, "punct": P}}


@pytest.mark.parametrize("text,expected", [
    ("Mr. Fogg paid £20,000 at 8.45 on the 2nd.", [("Mr.", "abbrev"), ("£20,000", "number"), ("8.45", "number"),
                                                  ("2nd", "number")]),
    ("Chapter XII. Henry VIII came.", [("XII", "roman"), ("VIII", "roman")]),
    ("No. 7 Savile Row, &c. in 1872, 50% off", [("No. 7", "abbrev"), ("&c.", "abbrev"), ("1872", "number"),
                                                 ("50%", "number")]),
    ("I said No. Then 1848-53 and 12:30.", [("1848-53", "number"), ("12:30", "number")]),
])
def test_find_nsw(text, expected):
    assert [(x["surface"], x["kind"]) for x in s4.find_nsw(doc(text))] == expected


def test_roman_gets_context_word():
    q = {x["surface"]: x["query"] for x in s4.find_nsw(doc("Chapter XII. Henry VIII came."))}
    assert q == {"XII": "Chapter XII", "VIII": "Henry VIII"}
    assert s4.strip_context("henry the eighth", "Henry") == "the eighth"


def test_plain_words():
    assert s4.plain_words("Half-past") == ["half", "past"]
    assert s4.plain_words("’tis") == ["tis"] and s4.plain_words("don’t") == ["don't"]
    assert s4.plain_words("Déjà-vu") == ["deja", "vu"]


def test_select_by_audio_window():
    nsw = [{"id": 0, "surface": "1872", "kind": "number", "default": "one thousand eight hundred seventy two",
            "candidates": ["one thousand eight hundred seventy two", "eighteen seventy two", "eighteen hundred and seventy two"]}]
    ref = [["lived", 1, -1], ["in", 2, -1]] + [[w, 3, 0] for w in nsw[0]["default"].split()] + [["at", 4, -1], ["number", 5, -1]]
    hyp = "lived in eighteen seventy two at number".split()
    ch = s4.select_section(ref, hyp, nsw)
    assert ch[0]["choice"] == "eighteen seventy two" and ch[0]["score"] == 0.0
    assert ch[0]["window"] == "eighteen seventy two"
    out = s4.apply_choices(ref, ch)
    assert [w for w, _, _ in out] == hyp and all(t == 3 for w, t, k in out if k == 0)


def test_select_keeps_default_when_reader_skipped_it():
    nsw = [{"id": 0, "surface": "7", "kind": "number", "default": "seven", "candidates": ["seven", "seventh"]}]
    ref = [["at", 1, -1], ["number", 2, -1], ["seven", 3, 0], ["savile", 4, -1], ["row", 5, -1]]
    hyp = "at number savile row".split()
    ch = s4.select_section(ref, hyp, nsw)
    assert ch[0]["choice"] == "seven" and ch[0]["window"] == ""


def test_window_score_ignores_misheard_neighbours():
    assert s4.window_cer("doctor", "doctor libesey") == 0.0
    assert s4.window_cer("twenty", "minutes twenty seconds") == 0.0
    assert s4.window_cer("two zero", "minutes twenty seconds") > 0.3


def test_rule_candidates():
    c = lambda sur, kind: s4.rule_candidates({"surface": sur, "kind": kind})  # noqa: E731
    assert "twenty thousand pounds" in c("£20,000", "number")
    assert "eight forty five" in c("8.45", "number")
    assert "twenty third" in c("23d", "number") and "second" in c("2nd", "number")
    assert "eighteen seventy two" in c("1872", "number")
    assert "et cetera" in c("&c.", "abbrev") and "number seven" in c("No. 7", "abbrev")
    assert c("XII", "roman") == ["twelve", "the twelfth"]
