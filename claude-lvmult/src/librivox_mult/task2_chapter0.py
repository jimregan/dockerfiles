"""Task 2: chapter-0 (cast list) review table for dramatic readings.

`uv run python -m librivox_mult.task2_chapter0`

Finds the sections of each dramatic reading whose wav2vec2 transcript says "read by" (or
"played by", "voiced by", "as") at least MIN_HITS times. For each cast entry on the LibriVox
page it shows the spoken snippet that follows the first mention of the role, so that a person
can fill in the reader actually named (usernames, extra roles, coordinators reading the list).

Writes `/out/review/task2/chapter0.tsv` (one row per cast entry) and
`/out/review/task2/chapter0_transcripts.txt` (full transcripts of those sections).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

OUT = Path("/out")
MIN_HITS = 3
CUE = re.compile(r"\b(read|played|voiced|performed) by\b")
SNIPPET = 14


def transcript(r: dict, s: dict) -> str | None:
    p = OUT / "w2v" / r["work"] / r["librivox_slug"] / f"{s['stem']}.json"
    if not p.exists():
        return None
    return " ".join(w["word"] for w in json.load(open(p))["words"]).lower()


def snippet(text: str, role: str) -> str:
    words = text.split()
    key = [w for w in re.findall(r"[a-z']+", role.lower()) if w not in {"the", "mr", "mrs", "miss", "of"}]
    if not key:
        return ""
    for i, w in enumerate(words):
        if w == key[0] and all(i + j < len(words) and words[i + j] == k for j, k in enumerate(key[: 2])):
            return " ".join(words[i:i + SNIPPET])
    return ""


def main(argv: list[str] | None = None) -> int:
    inv = [json.loads(x) for x in (OUT / "inventory.jsonl").read_text().splitlines()]
    dst = OUT / "review" / "task2"
    dst.mkdir(parents=True, exist_ok=True)
    rows = ["reading_id\twork\tsection\tduration\troles\tpage_reader\treader_id\tspoken_snippet\tcorrected_reader"]
    full = []
    for r in inv:
        if not r.get("dramatic"):
            continue
        found = False
        for s in r["sections"]:
            text = transcript(r, s)
            if not text or len(CUE.findall(text)) < MIN_HITS:
                continue
            found = True
            full.append(f"## {r['reading_id']} {r['work']} section {s['section']} ({s['duration']:.0f} s)\n{text}\n")
            for c in r["cast"]:
                names = "; ".join(x["name"] for x in c["readers"])
                ids = "; ".join(str(x["id"] or "") for x in c["readers"])
                snip = next((sn for role in c["roles"] if (sn := snippet(text, role))), "")
                rows.append(f"{r['reading_id']}\t{r['work']}\t{s['section']}\t{s['duration']:.0f}\t"
                            f"{' / '.join(c['roles'])}\t{names}\t{ids}\t{snip}\t")
        if not found:
            rows.append(f"{r['reading_id']}\t{r['work']}\t-\t\t(no cast-list section found)\t\t\t\t")
    (dst / "chapter0.tsv").write_text("\n".join(rows) + "\n")
    (dst / "chapter0_transcripts.txt").write_text("\n".join(full))
    print(dst / "chapter0.tsv", len(rows) - 1, "rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
