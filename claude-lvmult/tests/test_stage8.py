import gzip
import json
from pathlib import Path

import pytest

from librivox_mult import stage8_manifests as s8



def seg(sid, unit, start, dur, rows, words, spoken=True, text="One two."):
    return {"id": sid, "text_id": None if unit in "HX" else f"pg1/{sid[2:]}", "unit": unit,
            "start": start, "duration": dur, "spoken": spoken, "text": text, "src_span": None,
            "verbatim": " ".join(w["symbol"] for w in words), "alignment": words,
            "edits": [["cor", r, k] for k, r in enumerate(rows)],
            "boundary": {"start": "pause", "end": "pause"}, "pause_before": None, "pause_after": None,
            "dysfluencies": [], "nsw": [], "confidence": 0.9, "header_footer": None}


def w(sym, t):
    return {"symbol": sym, "start": t, "duration": 0.2, "score": 0.9}


def fixture(root: Path) -> dict:
    r = {"reading_id": "lv1", "work": "w", "readers": [{"id": 7, "name": "A"}],
         "sections": [{"stem": "w_01", "section": "01", "path": "/data/w_01.mp3", "channels": 1,
                       "sampling_rate": 16000, "num_samples": 16000 * 10, "readers": []}]}
    segs = [seg("01/H01", "H", 0.0, 1.0, [0], [w("chapter", 0.2)]),
            seg("S/001/0001/01", "S", 1.0, 2.0, [1, 2], [w("one", 1.2), w("two", 1.6)]),
            seg("S/001/0001/02", "S", 3.0, 0.0, [3], [], spoken=False),
            seg("01/X01", "X", 3.0, 4.0, [], [w("hm", 3.5)], text="")]
    segs[3]["edits"] = [["ins", -1, 0]]
    (root / "stage7" / "w").mkdir(parents=True)
    (root / "stage5" / "w").mkdir(parents=True)
    with gzip.open(root / "stage7" / "w" / "lv1.json.gz", "wt") as f:
        json.dump({"segmentations": {"S": [{"stem": "w_01", "section": "01", "text": "pg1",
                                            "segments": segs}]}}, f)
    with gzip.open(root / "stage5" / "w" / "lv1.json.gz", "wt") as f:
        json.dump({"sections": [{"stem": "w_01", "ref": [["chapter", -1, -1, 0], ["one", 0, -1, -1],
                                                          ["two", 1, -1, -1], ["three", 3, -1, -1]]}]}, f)
    return r


def test_manifests_load_in_lhotse(tmp_path, monkeypatch):
    lhotse = pytest.importorskip("lhotse")
    monkeypatch.setattr(s8, "OUT", tmp_path)
    res = s8.process((fixture(tmp_path), False))
    assert res["n_errors"] == 0 and res["supervisions"] == {"S": 3} and res["unspoken"] == {"S": 1}
    d = tmp_path / "stage8" / "w" / "lv1"
    recs = lhotse.RecordingSet.from_file(d / "recordings.jsonl.gz")
    sups = lhotse.SupervisionSet.from_file(d / "supervisions_S.jsonl.gz")
    lhotse.validate_recordings_and_supervisions(recs, sups)
    by = {x.id: x for x in sups}
    s = by["w/lv1/S/001/0001/01"]
    assert s.recording_id == "w_lv1_01" and s.speaker == "librivox:7"
    assert s.alignment["word"][1].symbol == "two" and s.custom["edits"][0] == ["cor", "one", 0, 0]
    assert by["w/lv1/01/X01"].custom["unreferenced"] is True


def test_validation_catches_overlap_and_gaps(tmp_path, monkeypatch):
    monkeypatch.setattr(s8, "OUT", tmp_path)
    r = fixture(tmp_path)
    p = tmp_path / "stage7" / "w" / "lv1.json.gz"
    d = json.load(gzip.open(p, "rt"))
    segs = d["segmentations"]["S"][0]["segments"]
    segs[1]["start"] = 0.5
    segs[2]["edits"] = []
    with gzip.open(p, "wt") as f:
        json.dump(d, f)
    errs = s8.process((r, True))["errors"]
    assert any(e.startswith("overlap:") for e in errs) and any(e.startswith("coverage:") for e in errs)


REAL = Path("/out/stage8/dracula/lv12966")


@pytest.mark.skipif(not (REAL / "supervisions_S.jsonl.gz").exists(), reason="no Stage 8 output")
def test_real_reading_loads():
    lhotse = pytest.importorskip("lhotse")
    recs = lhotse.RecordingSet.from_file(REAL / "recordings.jsonl.gz")
    sups = lhotse.SupervisionSet.from_file(REAL / "supervisions_S.jsonl.gz")
    lhotse.validate_recordings_and_supervisions(recs, sups)
    assert len(sups) > 1000
