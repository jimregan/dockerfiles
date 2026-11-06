"""Stage 0: survey /data and write /out/inventory.jsonl (one row per reading).

A *reading* is the set of local mp3s from one LibriVox project directory,
except for multi-version projects (several readers each reading the same
short text, e.g. weekly poetry), where each section is its own reading.

Outputs (all under --out, default /out):
  inventory.jsonl              one row per reading
  stage0/ffprobe.jsonl         probe cache, keyed by path/size/mtime
  stage0/summary.json          aggregate counts for docs/inputs.md
  logs/stage0.log

Deterministic: rows sorted by (work, reading_id); probe results cached.
Rerunning reuses the cache unless --force.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .librivox_page import parse_page

log = logging.getLogger("stage0")

ASR_SUBDIRS = {"whisperx": "whisperx", "wav2vec2": "wav2vec2"}
NON_WORK_DIRS = set(ASR_SUBDIRS.values())
VERSE_GENRES = {"poetry", "sonnets", "multi-version (weekly and fortnightly poetry)"}
PLAY_GENRES = {"plays", "drama", "comedy", "tragedy"}


def ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,bit_rate:stream=codec_name,sample_rate,channels",
        "-of", "json", str(path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
        j = json.loads(r.stdout)
        st = (j.get("streams") or [{}])[0]
        fmt = j.get("format", {})
        sr = int(st["sample_rate"]) if "sample_rate" in st else None
        # Header durations of LibriVox mp3s can be off by several seconds
        # (e.g. 3200.8 s header vs 3208.1 s decoded); count decoded samples.
        n = decoded_samples(path)
        return {
            "duration_header": round(float(fmt["duration"]), 3) if "duration" in fmt else None,
            "num_samples": n,
            "duration": round(n / sr, 6) if n is not None and sr else None,
            "sampling_rate": sr,
            "channels": st.get("channels"),
            "codec": st.get("codec_name"),
            "bit_rate": int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 - recorded, not fatal
        return {"duration": None, "error": f"{type(e).__name__}: {e}"[:300]}


def decoded_samples(path: Path) -> int | None:
    """Exact sample count at native rate: decode to mono u8, count bytes."""
    p = subprocess.run(
        f"ffmpeg -nostdin -v error -i {shlex.quote(str(path))} -map 0:a:0 -ac 1 -f u8 - | wc -c",
        shell=True, capture_output=True, text=True, timeout=1800,
    )
    if p.returncode != 0 or p.stderr.strip():
        log.warning("decode %s: rc=%s %s", path, p.returncode, p.stderr.strip()[:200])
    try:
        return int(p.stdout.strip())
    except ValueError:
        return None


class ProbeCache:
    def __init__(self, path: Path, force: bool):
        self.path = path
        self.data: dict[str, dict] = {}
        if path.exists() and not force:
            for line in path.read_text().splitlines():
                r = json.loads(line)
                self.data[r["key"]] = r

    @staticmethod
    def key(p: Path) -> str:
        s = p.stat()
        return f"{p}|{s.st_size}|{int(s.st_mtime)}"

    def probe_all(self, paths: list[Path], jobs: int) -> dict[Path, dict]:
        todo = [p for p in paths if self.key(p) not in self.data]
        log.info("ffprobe: %d cached, %d to probe", len(paths) - len(todo), len(todo))
        with ThreadPoolExecutor(jobs) as ex:
            for p, res in zip(todo, ex.map(ffprobe, todo)):
                self.data[self.key(p)] = {"key": self.key(p), "path": str(p), **res}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w") as f:
            for k in sorted(self.data):
                f.write(json.dumps(self.data[k], sort_keys=True) + "\n")
        tmp.replace(self.path)
        return {p: self.data[self.key(p)] for p in paths}


def vibevoice_info(path: Path) -> dict:
    """VibeVoice output: JSON list of {Start, End, Content, Speaker}."""
    if not path.exists():
        return {"present": False}
    try:
        segs = json.loads(path.read_text())
        assert isinstance(segs, list)
        return {
            "present": True,
            "path": str(path),
            "n_segments": len(segs),
            # non-speech segments ("[Silence]", "[Music]"...) have no Speaker key
            "n_speakers": len({s["Speaker"] for s in segs if s.get("Speaker") is not None}),
            "nonspeech_tags": dict(Counter(
                (s.get("Content") or "").strip() for s in segs if s.get("Speaker") is None
            )),
            "end": max((s.get("End") or 0 for s in segs), default=0.0),
            "n_chars": sum(len(s.get("Content") or "") for s in segs),
        }
    except Exception as e:  # noqa: BLE001
        return {"present": True, "path": str(path), "error": str(e)[:200]}


def whisperx_info(path: Path) -> dict:
    """WhisperX output: {segments:[{start,end,text,words:[{word,start,end,score}]}],
    word_segments, language}; siblings .srt/.tsv/.txt/.vtt."""
    if not path.exists():
        return {"present": False}
    try:
        j = json.loads(path.read_text())
        words = j.get("word_segments") or [w for s in j["segments"] for w in s.get("words", [])]
        untimed = sum(1 for w in words if "start" not in w)
        return {
            "present": True,
            "path": str(path),
            "language": j.get("language"),
            "n_segments": len(j.get("segments", [])),
            "n_words": len(words),
            "n_words_untimed": untimed,
            "end": max((s.get("end") or 0 for s in j.get("segments", [])), default=0.0),
            "has_speaker": any("speaker" in s for s in j.get("segments", [])),
        }
    except Exception as e:  # noqa: BLE001
        return {"present": True, "path": str(path), "error": str(e)[:200]}


def respiro_info(path: Path) -> dict:
    """Respiro breath detector: one JSON line {audio_file, breaths:[[start,end],...]}."""
    if not path.exists():
        return {"present": False}
    try:
        d = json.loads(path.read_text().splitlines()[0])
        b = d["breaths"]
        return {
            "present": True,
            "path": str(path),
            "n_breaths": len(b),
            "breath_time": round(sum(e - s for s, e in b), 3),
            "first_starts_at_zero": bool(b and b[0][0] == 0.0),
        }
    except Exception as e:  # noqa: BLE001
        return {"present": True, "path": str(path), "error": str(e)[:200]}


EXCLUSIONS_FILE = Path(__file__).resolve().parents[2] / "data" / "exclusions.json"
EXCLUSIONS: dict[str, str] = (json.loads(EXCLUSIONS_FILE.read_text())["files"]
                              if EXCLUSIONS_FILE.exists() else {})

RESPIRO_ROOT = Path("/src-projects/librivox-mult-work-files/respiro")


def find_asr(data: Path, work: str, reading_dir: str, stem: str,
             respiro_root: Path = RESPIRO_ROOT, staging: Path | None = None) -> dict:
    vv = data / work / reading_dir / f"{stem}.json"
    if not vv.exists() and staging is not None and (staging / work / reading_dir / f"{stem}.json").exists():
        vv = staging / work / reading_dir / f"{stem}.json"
    out = {"vibevoice": vibevoice_info(vv)}
    out["respiro"] = respiro_info(respiro_root / work / reading_dir / f"{stem}.jsonl")
    out["whisperx"] = whisperx_info(data / "whisperx" / work / reading_dir / f"{stem}.json")
    # wav2vec2: layout unknown (directory empty at time of survey); look for any
    # file with this stem under /data/wav2vec2.
    w2v = sorted((data / "wav2vec2").rglob(f"{stem}.*")) if (data / "wav2vec2").exists() else []
    out["wav2vec2"] = {"present": bool(w2v), "paths": [str(p) for p in w2v]}
    return out


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def is_multiversion(page: dict) -> bool:
    if any("multi-version" in g.lower() for g in page["genres"]):
        return True
    secs = page["sections"]
    if len(secs) < 2:
        return False
    # "IF - Read by ADR" / "If (read by X)": the reader credit is not part of the title
    titles = Counter(norm_title(re.sub(r"\s*[-–—:(]*\s*read by\b.*$|\(.*?\)", "", s["title"], flags=re.I))
                     for s in secs)
    readers = {tuple(r["id"] or r["name"] for r in s["readers"]) for s in secs}
    # >=80% sharing one title tolerates page typos ("IIF - Read by ...")
    return titles.most_common(1)[0][1] >= 0.8 * len(secs) and len(readers) == len(secs)


def classify(page: dict, slug: str, n_readers: int, collection: bool = False) -> dict:
    genres = {g.lower() for g in page["genres"]}
    # NB: not the free-text title: "Condensed by the Author for his Dramatic
    # Readings" (Dickens's public-reading text) is a solo reading.
    slug_dr = "dramatic-reading" in slug
    dramatic = bool(page["cast"]) or "dramatic readings" in genres or slug_dr
    play = bool(genres & PLAY_GENRES)
    verse = bool(genres & VERSE_GENRES)
    return {
        # Collection genres describe the whole collection (a "Poetry" collection
        # can contain a prose tale), so leave form for Stage 1 to decide.
        "verse": None if collection else verse,
        "form": "unknown" if collection else ("verse" if verse else "prose"),
        "dramatic": dramatic,
        "play": play,
        "solo": n_readers <= 1,
        "evidence": {
            "genres": page["genres"],
            "cast_entries": len(page["cast"]),
            "slug_says_dramatic": slug_dr,
            "section_reader_text": sorted({s.get("reader_text") or "" for s in page["sections"]})[:5],
        },
    }


def match_section(mp3: str, sections: list[dict]) -> dict | None:
    def base(n: str) -> str:
        return re.sub(r"_(64|128)kb(?=\.mp3$)", "", n.lower())

    for s in sections:
        if mp3 in s["mp3_names"]:
            return s
    for s in sections:
        if base(mp3) in {base(n) for n in s["mp3_names"]}:
            return s
    return None


def reader_key(r: dict) -> str:
    return f"librivox:{r['id']}" if r.get("id") else f"name:{r['name']}"


def survey_reading(data: Path, work: str, rdir: Path, probes: dict[Path, dict],
                   respiro_root: Path = RESPIRO_ROOT, staging: Path | None = None) -> list[dict]:
    index = rdir / "index.html"
    index_source = "data"
    if not index.exists() and staging is not None:
        # pages fetched later (e.g. from librivox.org) and staged for copying into /data
        alt = staging / work / rdir.name / "index.html"
        if alt.exists():
            index, index_source = alt, f"staging:{alt}"
    page = parse_page(index.read_text(encoding="utf-8", errors="replace")) if index.exists() else None
    mp3s = sorted(rdir.glob("*.mp3"))
    stray = sorted(p.name for p in rdir.iterdir() if p.is_file() and re.search(r"\.mp3\.\d+$", p.name))
    files_list = rdir / "files"
    listed = (
        [ln.strip().rsplit("/", 1)[-1] for ln in files_list.read_text().splitlines() if ln.strip()]
        if files_list.exists() else None
    )
    issues = []
    if page is None:
        issues.append("no index.html")
    if stray:
        issues.append(f"stray duplicate downloads: {stray}")
    if listed is not None:
        missing = sorted(set(listed) - {p.name for p in mp3s})
        if missing:
            issues.append(f"in 'files' but not downloaded: {missing}")

    sections_local = []
    for p in mp3s:
        sec = match_section(p.name, page["sections"]) if page else None
        pr = probes.get(p, {})
        row = {
            "file": p.name,
            "path": str(p),
            "stem": p.stem,
            "section": sec["section"] if sec else None,
            "title": sec["title"] if sec else None,
            # e.g. Monte Cristo "..., alternate version": a second reader's
            # recording of the same chapter inside one project
            "alternate": bool(sec and re.search(r"(?i)\balternate\b", sec["title"])),
            # data/exclusions.json: files judged unusable (later stages skip them)
            "excluded": EXCLUSIONS.get(str(p.relative_to(data))),
            "readers": sec["readers"] if sec else [],
            "listed_time": sec["time"] if sec else None,
            "duration": pr.get("duration"),
            "num_samples": pr.get("num_samples"),
            "duration_header": pr.get("duration_header"),
            "sampling_rate": pr.get("sampling_rate"),
            "channels": pr.get("channels"),
            "bit_rate": pr.get("bit_rate"),
            "probe_error": pr.get("error"),
            "asr": find_asr(data, work, rdir.name, p.stem, respiro_root, staging),
        }
        for k in ("author", "source", "source_urls", "gutenberg_ids"):
            if sec and k in sec:
                row[f"section_{k}"] = sec[k]
        if page and sec is None:
            issues.append(f"mp3 not in section table: {p.name}")
        sections_local.append(row)

    def build(rows: list[dict], reading_id: str, extra_issues: list[str]) -> dict:
        readers: dict[str, dict] = {}
        for s in rows:
            for r in s["readers"]:
                readers.setdefault(reader_key(r), r)
        # dramatic readings list the section reader as "Group": use the cast
        for c in (page["cast"] if page else []):
            for r in c["readers"]:
                readers.setdefault(reader_key(r), r)
        if page and not readers:
            for r in page["read_by"]:
                readers.setdefault(reader_key(r), r)
        collection = bool(page and "source" in page.get("section_columns", []))
        # Collections: only the section's own Source link describes our text;
        # project-level links belong to the collection as a whole.
        gids = [] if collection else (list(page["gutenberg_ids"]) if page else [])
        for s in rows:
            for g in s.get("section_gutenberg_ids", []):
                if g not in gids:
                    gids.append(g)
        asr_counts = {
            k: sum(1 for s in rows if s["asr"][k].get("present")) for k in ("wav2vec2", "whisperx", "vibevoice", "respiro")
        }
        durs = [s["duration"] for s in rows if s["duration"] is not None]
        cls = classify(page, rdir.name, len(readers), collection) if page else {
            "verse": None, "form": None, "dramatic": None, "play": None,
            "solo": None, "evidence": {},
        }
        return {
            "work": work,
            "reading_id": reading_id,
            "reading_dir": str(rdir),
            "index_source": index_source if page else None,
            "librivox_slug": rdir.name,
            "librivox_id": page["librivox_id"] if page else None,
            "archive_id": page["archive_id"] if page else None,
            "title": page["title"] if page else None,
            "authors": page["authors"] if page else [],
            "credits_line": page["credits_line"] if page else None,
            "language": page["language"] if page else None,
            "catalog_date": page["catalog_date"] if page else None,
            "readers": sorted(readers.values(), key=lambda r: (r["id"] is None, r["id"] or 0, r["name"])),
            "n_readers": len(readers),
            "coordinators": page["book_coordinator"] if page else [],
            "gutenberg_ids": gids,
            "online_text": page["online_text"] if page else [],
            "source_kind": (
                "gutenberg" if gids else ("other" if page and page["online_text"] else "none")
            ),
            "collection": collection,
            "section_source_urls": sorted({u for s in rows for u in s.get("section_source_urls", [])}),
            "multiversion": False,
            "n_sections_listed": len(page["sections"]) if page else None,
            "n_sections_local": len(rows),
            "duration": round(sum(durs), 3),
            **cls,
            "cast": page["cast"] if page else [],
            "asr_coverage": {**asr_counts, "n_files": len(rows)},
            "sections": rows,
            "issues": issues + extra_issues,
        }

    lvid = page["librivox_id"] if page and page["librivox_id"] else None
    base_id = f"lv{lvid}" if lvid else f"dir:{rdir.name}"
    if page and is_multiversion(page) and len(sections_local) > 1:
        out = []
        for s in sections_local:
            sid = (s["section"] or s["stem"]).zfill(2)
            r = build([s], f"{base_id}-s{sid}", [])
            r["multiversion"] = True
            out.append(r)
        return out
    if (page and "source" in page.get("section_columns", []) and sections_local
            and len(sections_local) < len(page["sections"])):
        # Collections: one LibriVox project appears under several works with a
        # different downloaded section each, so the section is part of the id.
        secs = "+".join(sorted({(x["section"] or x["stem"]).zfill(2) for x in sections_local}))
        base_id = f"{base_id}-s{secs}"
    return [build(sections_local, base_id, [])]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("/data"))
    ap.add_argument("--out", type=Path, default=Path("/out"))
    ap.add_argument("--work", action="append", help="limit to these work dirs (repeatable)")
    ap.add_argument("--reading", action="append", help="limit to these LibriVox slugs (repeatable)")
    ap.add_argument("--force", action="store_true", help="re-probe audio, ignore cache")
    ap.add_argument("--staging", type=Path, default=Path("/out/staging/data"),
                    help="fallback for files not yet copied into /data (same layout)")
    ap.add_argument("--respiro", type=Path, default=RESPIRO_ROOT,
                    help="root of Respiro breath outputs (<work>/<slug>/<stem>.jsonl)")
    ap.add_argument("--jobs", type=int, default=min(16, (os.cpu_count() or 4)))
    a = ap.parse_args(argv)

    (a.out / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(a.out / "logs" / "stage0.log"), logging.StreamHandler(sys.stderr)],
    )
    works = sorted(
        d.name for d in a.data.iterdir() if d.is_dir() and d.name not in NON_WORK_DIRS
    )
    if a.work:
        works = [w for w in works if w in a.work]
    rdirs = [
        (w, r) for w in works for r in sorted((a.data / w).iterdir())
        if r.is_dir() and (not a.reading or r.name in a.reading)
    ]
    mp3s = [p for _, r in rdirs for p in sorted(r.glob("*.mp3"))]
    probes = ProbeCache(a.out / "stage0" / "ffprobe.jsonl", a.force).probe_all(mp3s, a.jobs)

    rows = []
    for w, r in rdirs:
        rows.extend(survey_reading(a.data, w, r, probes, a.respiro, a.staging))
    rows.sort(key=lambda x: (x["work"], x["reading_id"]))

    # partial runs update only their own rows in an existing inventory
    inv = a.out / "inventory.jsonl"
    if (a.work or a.reading) and inv.exists():
        keep = [
            json.loads(ln) for ln in inv.read_text().splitlines()
            if not any(json.loads(ln)["reading_dir"] == str(a.data / w / r.name) for w, r in rdirs)
        ]
        rows = sorted(keep + rows, key=lambda x: (x["work"], x["reading_id"]))
    tmp = inv.with_suffix(".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(inv)

    summary = summarise(rows)
    (a.out / "stage0" / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    log.info("wrote %d readings to %s", len(rows), inv)
    return 0


def summarise(rows: list[dict]) -> dict:
    secs = [s for r in rows for s in r["sections"]]
    return {
        "n_works": len({r["work"] for r in rows}),
        "n_readings": len(rows),
        "n_files": len(secs),
        "hours": round(sum(r["duration"] for r in rows) / 3600, 1),
        "asr_files": {
            k: sum(1 for s in secs if s["asr"][k].get("present"))
            for k in ("wav2vec2", "whisperx", "vibevoice", "respiro")
        },
        "form": Counter(r.get("form") for r in rows),
        "dramatic": sum(1 for r in rows if r.get("dramatic")),
        "dramatic_multi_reader": sum(1 for r in rows if r.get("dramatic") and not r.get("solo")),
        "dramatic_single_reader": sum(1 for r in rows if r.get("dramatic") and r.get("solo")),
        "collections": sum(1 for r in rows if r["collection"]),
        "multiversion_readings": sum(1 for r in rows if r["multiversion"]),
        "source_kind": Counter(r["source_kind"] for r in rows),
        "sampling_rates": Counter(s["sampling_rate"] for s in secs),
        "channels": Counter(s["channels"] for s in secs),
        "probe_errors": [s["path"] for s in secs if s["probe_error"]],
        "excluded_files": {s["path"]: s["excluded"] for s in secs if s.get("excluded")},
        "readings_with_issues": {r["reading_id"]: r["issues"] for r in rows if r["issues"]},
    }


if __name__ == "__main__":
    sys.exit(main())
