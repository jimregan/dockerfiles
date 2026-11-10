"""Light spoken-form normalisation into the wav2vec2 alphabet (a-z, ', space).

Used for header/footer templates (titles, names, section numbers), where a
rough spoken form is enough because matching is fuzzy. Book text gets the
NeMo-based treatment in Stage 4; this is not that.
"""

from __future__ import annotations

import re
import unicodedata

from num2words import num2words

ABBREV = {
    "mr": "mister", "mrs": "missus", "dr": "doctor", "st": "saint", "vol": "volume",
    "no": "number", "ch": "chapter", "jr": "junior", "sr": "senior", "&": "and",
}
LIGATURES = str.maketrans({"æ": "ae", "œ": "oe", "ß": "ss", "ø": "o", "ð": "d", "þ": "th", "ł": "l"})
ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
# words after which a roman numeral is read as a number ("Part II", "Volume III")
NUMBERED = {"chapter", "part", "book", "volume", "section", "stave", "act", "scene",
            "canto", "fable", "story", "letter", "song", "number", "vol"}


def roman_to_int(s: str) -> int | None:
    s = s.lower()
    if not s or any(c not in ROMAN for c in s):
        return None
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = ROMAN[a]
        total += -v if b != " " and ROMAN.get(b, 0) > v else v
    return total if 0 < total < 400 else None


def number_words(tok: str, year_ok: bool = True) -> str:
    n = int(tok)
    if year_ok and len(tok) == 4 and 1100 <= n <= 2099:
        return num2words(n, to="year")
    return num2words(n)


def spoken_words(text: str) -> list[str]:
    """'Chapter XII. Mr. Fogg & 1872' -> ['chapter','twelve','mister','fogg','and','eighteen','seventy','two']"""
    t = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
    t = re.sub(r"(\d)(st|nd|rd|th)\b", lambda m: m.group(1) + "\x00", t)  # ordinals marker
    raw = re.findall(r"[^\W_]+(?:'[^\W_]+)*\x00?|&", t)
    out: list[str] = []
    prev = ""
    for w in raw:
        ordinal = w.endswith("\x00")
        w = w.rstrip("\x00")
        lw = w.lower()
        if w.isdigit():
            words = num2words(int(w), to="ordinal") if ordinal else number_words(w)
            out.extend(re.split(r"[\s,-]+", words.replace(" and ", " and ")))
        elif prev in NUMBERED and w.isupper() and roman_to_int(w):
            out.extend(re.split(r"[\s,-]+", num2words(roman_to_int(w))))
        elif lw in ABBREV:
            out.append(ABBREV[lw])
        else:
            lw = lw.translate(LIGATURES)
            lw = "".join(c for c in unicodedata.normalize("NFKD", lw) if not unicodedata.combining(c))
            lw = re.sub(r"[^a-z']", "", lw)
            if lw:
                out.append(lw)
        prev = lw
    return [w for w in out if w]


def letters(words: list[str]) -> str:
    """Space-free letter string used for fuzzy matching (tolerates word splits)."""
    return "".join(words)


DIGITS = ["o", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def number_variants(n: int) -> list[str]:
    """Ways readers say a section number: 'one hundred and two', 'one hundred two',
    'one o two', 'number one hundred and two'."""
    base = " ".join(re.split(r"[\s,-]+", num2words(n)))
    vs = [base, base.replace(" and ", " ")]
    if 100 <= n <= 999:
        h, rest = divmod(n, 100)
        vs.append(f"{DIGITS[h]} " + (" ".join(re.split(r"[\s-]+", num2words(rest))) if rest >= 10
                                     else f"o {DIGITS[rest] if rest else 'o'}"))
    vs += [f"number {v}" for v in vs[:2]]
    return list(dict.fromkeys(vs))
