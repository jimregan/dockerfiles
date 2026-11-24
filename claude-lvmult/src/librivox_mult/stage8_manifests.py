"""Stage 8: lhotse-schema manifests and validation.

`uv run python -m librivox_mult.stage8_manifests [--work W] [--reading ID] [--force]`

Per reading, from Stage 7 (segments), Stage 5 (reference rows) and the Stage 0 inventory,
writes `/out/stage8/<work>/<reading_id>/`:

- `recordings.jsonl.gz`: one Recording per section audio file that has segments.
- `supervisions_S.jsonl.gz` (and `supervisions_L.jsonl.gz` for readings with verse): one
  SupervisionSegment per spoken segment.
- `unspoken_S.jsonl.gz` (`_L`): reference units with no speech (unread contents, skipped
  sentences). Same fields, but no audio extent, so they are not supervisions.
- `validation.json`: the checks below.

Then concatenates all readings into `/out/manifests/` in inventory order. lhotse itself is
not used here (it is a test dependency); `tests/test_stage8.py` loads the output with it.

Checks: required fields and types; no overlap within a segmentation of a recording;
supervisions and their alignment items inside the recording; every Stage 5 reference row
covered exactly once per segmentation (supervisions + unspoken).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

OUT = Path("/out")
VERSION = "stage8-0.2"
EPS = 0.011  # seconds of slack for rounded times


def recording_id(work: str, rid: str, section: str) -> str:
    return f"{work}_{rid}_{section}"


def speaker_of(sec_inv: dict, reading: dict) -> tuple[str | None, list[dict]]:
    """`librivox:<reader id>` of the section's reader; the reading's sole reader when the
    section lists none. Several readers: the first, with all of them returned for custom."""
    rs = sec_inv.get("readers") or []
    if not rs and len(reading.get("readers") or []) == 1:
        rs = reading["readers"]
    rs = [r for r in rs if r.get("id") is not None]
    return (f"librivox:{rs[0]['id']}" if rs else None), rs


def recording(work: str, rid: str, s: dict) -> dict:
    ch = list(range(s["channels"]))
    return {"id": recording_id(work, rid, s["section"]),
            "sources": [{"type": "file", "channels": ch, "source": s["path"]}],
            "sampling_rate": s["sampling_rate"], "num_samples": s["num_samples"],
            "duration": round(s["num_samples"] / s["sampling_rate"], 6), "channel_ids": ch}


def supervision(seg: dict, g: str, work: str, rid: str, sec7: dict, rec: dict, rows: list,
                speaker: str | None, readers: list[dict]) -> dict:
    edits = []
    for kind, ri, ai in seg["edits"]:
        row = rows[ri] if ri is not None and ri >= 0 else None
        edits.append([kind, row[0] if row else None,
                      row[1] if row and row[1] >= 0 else None,
                      ai if ai is not None and ai >= 0 else None])
    custom = {"segmentation": g, "unit": seg["unit"], "text_id": seg["text_id"],
              "gutenberg": sec7["text"], "section": sec7["section"],
              "verbatim": seg["verbatim"], "src_span": seg["src_span"],
              "boundary": seg["boundary"], "pause_before": seg["pause_before"],
              "pause_after": seg["pause_after"], "dysfluencies": seg["dysfluencies"],
              "nsw": seg["nsw"], "confidence": seg["confidence"],
              "header_footer": seg["header_footer"], "edits": edits}
    if seg["unit"] == "X":
        custom["unreferenced"] = True
    if len(readers) > 1:
        custom["readers"] = [f"librivox:{r['id']}" for r in readers]
    ch = rec["channel_ids"]
    return {"id": f"{work}/{rid}/{seg['id']}", "recording_id": rec["id"],
            "start": seg["start"], "duration": seg["duration"],
            "channel": ch[0] if len(ch) == 1 else ch,
            "text": seg["text"], "language": "en", "speaker": speaker,
            "alignment": {"word": seg["alignment"]}, "custom": custom}


REQUIRED = {"id": str, "recording_id": str, "start": (int, float), "duration": (int, float),
            "text": str, "language": str}


def validate(recs: dict, sups: dict, unspoken: dict, n_rows: dict) -> list[str]:
    errs = []
    for g, ss in sups.items():
        ids: Counter = Counter()
        by_rec: dict = {}
        for s in ss:
            ids[s["id"]] += 1
            for k, t in REQUIRED.items():
                if not isinstance(s.get(k), t):
                    errs.append(f"schema:{s['id']}:{k}")
            r = recs.get(s["recording_id"])
            if r is None:
                errs.append(f"no_recording:{s['id']}")
                continue
            if s["duration"] <= 0 or s["start"] < 0 or s["start"] + s["duration"] > r["duration"] + EPS:
                errs.append(f"outside:{s['id']}")
            for a in s["alignment"]["word"]:
                if a["start"] < s["start"] - EPS or a["start"] + a["duration"] > s["start"] + s["duration"] + EPS:
                    errs.append(f"word_outside:{s['id']}")
                    break
            by_rec.setdefault(s["recording_id"], []).append(s)
        for rid, xs in by_rec.items():
            xs.sort(key=lambda s: s["start"])
            for a, b in zip(xs, xs[1:]):
                if b["start"] < a["start"] + a["duration"] - EPS:
                    errs.append(f"overlap:{g}:{a['id']}|{b['id']}")
        for u in unspoken.get(g, []):
            ids[u["id"]] += 1
        cover: dict = {}
        for s in ss + unspoken.get(g, []):
            for e in s["custom"]["edits_rows"]:
                cover.setdefault(s["recording_id"], Counter())[e] += 1
        for rid, n in n_rows.items():
            c = cover.get(rid, Counter())
            bad = [i for i in range(n) if c[i] != 1]
            if bad:
                errs.append(f"coverage:{g}:{rid}:{len(bad)}")
        errs += [f"duplicate_id:{g}:{i}" for i, n in ids.items() if n > 1]
    return errs


def write_jsonl(path: Path, items: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    tmp.replace(path)


def process(args: tuple[dict, bool]) -> dict:
    r, force = args
    rid, work = r["reading_id"], r["work"]
    p7 = OUT / "stage7" / work / f"{rid}.json.gz"
    p5 = OUT / "stage5" / work / f"{rid}.json.gz"
    if not (p7.exists() and p5.exists()):
        return {"reading_id": rid, "status": "no_input"}
    dst = OUT / "stage8" / work / rid
    key = hashlib.sha256(p7.read_bytes() + p5.read_bytes() + json.dumps(r, sort_keys=True).encode()
                         + VERSION.encode()).hexdigest()[:16]
    done = dst / "validation.json"
    if done.exists() and not force and json.loads(done.read_text()).get("input_sha") == key:
        return {"reading_id": rid, "status": "done", **json.loads(done.read_text())}
    d7 = json.load(gzip.open(p7, "rt"))
    d5 = {s["stem"]: s for s in json.load(gzip.open(p5, "rt"))["sections"]}
    inv = {s["stem"]: s for s in r["sections"]}
    recs, sups, unsp, n_rows = {}, {}, {}, {}
    no_speaker = 0
    for g, secs in d7["segmentations"].items():
        sups[g], unsp[g] = [], []
        for sec7 in secs:
            s = inv[sec7["stem"]]
            rec = recs.setdefault(recording_id(work, rid, s["section"]), recording(work, rid, s))
            rows = d5[sec7["stem"]]["ref"]
            n_rows[rec["id"]] = len(rows)
            speaker, readers = speaker_of(s, r)
            for seg in sec7["segments"]:
                sup = supervision(seg, g, work, rid, sec7, rec, rows, speaker, readers)
                sup["custom"]["edits_rows"] = [ri for _, ri, _ in seg["edits"] if ri is not None and ri >= 0]
                if seg["spoken"]:
                    sups[g].append(sup)
                    no_speaker += speaker is None and g == "S"
                else:
                    sup["start"], sup["duration"] = seg["start"], 0.0
                    unsp[g].append(sup)
    errs = validate(recs, sups, unsp, n_rows)
    for xs in list(sups.values()) + list(unsp.values()):
        for x in xs:
            del x["custom"]["edits_rows"]
    dst.mkdir(parents=True, exist_ok=True)
    write_jsonl(dst / "recordings.jsonl.gz", list(recs.values()))
    for g in sups:
        write_jsonl(dst / f"supervisions_{g}.jsonl.gz", sups[g])
        write_jsonl(dst / f"unspoken_{g}.jsonl.gz", unsp[g])
    v = {"version": VERSION, "input_sha": key, "errors": errs[:200], "n_errors": len(errs),
         "recordings": len(recs), "supervisions": {g: len(x) for g, x in sups.items()},
         "unspoken": {g: len(x) for g, x in unsp.items()}, "no_speaker": no_speaker,
         "hours": round(sum(x["duration"] for x in recs.values()) / 3600, 3)}
    done.write_text(json.dumps(v))
    return {"reading_id": rid, "status": "written", **v}


def combine(inv: list[dict]) -> dict:
    dst = OUT / "manifests"
    dst.mkdir(parents=True, exist_ok=True)
    names = ["recordings", "supervisions_S", "supervisions_L", "unspoken_S", "unspoken_L"]
    counts = Counter()
    outs = {n: gzip.open(dst / f"{n}.jsonl.tmp", "wt") for n in names}
    for r in inv:
        d = OUT / "stage8" / r["work"] / r["reading_id"]
        for n in names:
            p = d / f"{n}.jsonl.gz"
            if p.exists():
                with gzip.open(p, "rt") as f:
                    for line in f:
                        outs[n].write(line)
                        counts[n] += 1
    for n, f in outs.items():
        f.close()
        (dst / f"{n}.jsonl.tmp").replace(dst / f"{n}.jsonl.gz")
    return dict(counts)


def write_review(results: list[dict], counts: dict) -> None:
    rdir = OUT / "review" / "stage8"
    rdir.mkdir(parents=True, exist_ok=True)
    with (rdir / "readings.tsv").open("w") as f:
        f.write("reading_id\tstatus\trecordings\thours\tsup_S\tsup_L\tunspoken_S\tno_speaker\tn_errors\tfirst_error\n")
        for x in sorted(results, key=lambda x: x["reading_id"]):
            f.write("\t".join(map(str, [x["reading_id"], x["status"], x.get("recordings"), x.get("hours"),
                                        x.get("supervisions", {}).get("S"), x.get("supervisions", {}).get("L"),
                                        x.get("unspoken", {}).get("S"), x.get("no_speaker"), x.get("n_errors"),
                                        (x.get("errors") or [""])[0]])) + "\n")
    kinds = Counter(e.split(":")[0] for x in results for e in x.get("errors", []))
    with (rdir / "summary.txt").open("w") as f:
        f.write(f"readings: {Counter(x['status'] for x in results)}\n")
        f.write(f"manifest lines: {counts}\n")
        f.write(f"hours: {round(sum(x.get('hours', 0) for x in results), 1)}\n")
        f.write(f"supervisions without speaker: {sum(x.get('no_speaker', 0) for x in results)}\n")
        f.write(f"validation errors: {sum(x.get('n_errors', 0) for x in results)} {dict(kinds)}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", action="append")
    ap.add_argument("--reading", action="append")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args(argv)
    inv = [json.loads(x) for x in (OUT / "inventory.jsonl").read_text().splitlines()]
    sel = [(r, a.force) for r in inv if not (a.work and r["work"] not in a.work)
           and not (a.reading and r["reading_id"] not in a.reading)]
    with Pool(a.jobs) as pool:
        results = list(pool.imap_unordered(process, sel))
    for x in results:
        if x.get("n_errors"):
            print(x["reading_id"], x["n_errors"], x["errors"][:3], flush=True)
    counts = combine(inv)
    if not a.work and not a.reading:
        write_review(results, counts)
    print(dict(Counter(x["status"] for x in results)), counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
