import json
from pathlib import Path

import pytest

from librivox_mult import stage0_survey as s0
from librivox_mult.librivox_page import parse_page

FIX = Path(__file__).parent / "fixtures" / "librivox_pages"


def page(name: str) -> dict:
    return parse_page((FIX / name).read_text(encoding="utf-8"))


def test_solo_prose_page():
    p = page("lv714_around80days.html")
    assert p["librivox_id"] == 714
    assert p["gutenberg_ids"] == [103]
    assert p["title"] == "Around the World in Eighty Days"
    assert len(p["sections"]) == 37
    s1 = p["sections"][0]
    assert s1["section"] == "01"
    assert s1["mp3_names"][-1] == "around80days_01_verne.mp3"
    assert s1["readers"] == [{"name": "TBOL3", "id": 492}]
    assert s1["time"] == "00:12:23"
    assert p["cast"] == []
    assert "Translated by George Makepeace Towle" in p["credits_line"]


def test_dramatic_cast_list():
    p = page("lv9611_around80days_dramatic.html")
    assert "Dramatic Readings" in p["genres"]
    roles = {r for c in p["cast"] for r in c["roles"]}
    assert {"Narrator", "Phileas Fogg", "Passapartout", "Aouda", "Porter"} <= roles
    fogg = next(c for c in p["cast"] if "Phileas Fogg" in c["roles"])
    assert fogg["readers"] == [{"name": "BrassRhino", "id": 9822}]
    # editor / proof-listener credits after the cast are not cast entries
    assert not any("edited" in c["raw"].lower() or "proof" in c["raw"].lower() for c in p["cast"])


def test_headerless_cast_list():
    p = page("lv10387_oliver_dramatic_headerless_cast.html")
    assert len(p["cast"]) >= 10
    assert p["cast"][0]["roles"] == ["Narrator"]


def test_cast_reader_with_non_librivox_link():
    p = page("lv6889_treasure_dramatic_external_links.html")
    bd = next(c for c in p["cast"] if "Black Dog" in c["roles"])
    assert bd["readers"] == [{"name": "Phillip J Mather", "id": None}]


def test_group_reader_cell():
    p = page("lv5213_carol_dramatic_group.html")
    assert all(s["reader_text"] == "Group" and s["readers"] == [] for s in p["sections"])
    assert len(p["cast"]) == 27


def test_collection_section_source():
    p = page("lv9296_first_chapter_collection.html")
    assert "source" in p["section_columns"]
    sg = next(s for s in p["sections"] if s["title"] == "The Secret Garden")
    assert sg["gutenberg_ids"] == [113]
    assert sg["author"].startswith("Frances Hodgson Burnett")


def test_multiversion_detection():
    assert s0.is_multiversion(page("if_multiversion.html"))
    assert not s0.is_multiversion(page("lv714_around80days.html"))


def test_classify():
    c = s0.classify(page("lv9611_around80days_dramatic.html"), "x-dramatic-reading-y", 19)
    assert c["dramatic"] and not c["solo"] and c["form"] == "prose"
    c = s0.classify(page("if_multiversion.html"), "if-by-rudyard-kipling", 1)
    assert c["form"] == "verse" and not c["dramatic"]
    c = s0.classify(page("lv9296_first_chapter_collection.html"), "coll", 1, collection=True)
    assert c["form"] == "unknown"


def test_match_section_bitrate_suffix():
    secs = [{"mp3_names": ["a_01_x_64kb.mp3", "a_01_x.mp3"], "section": "01"}]
    assert s0.match_section("a_01_x.mp3", secs)["section"] == "01"
    assert s0.match_section("a_01_x_128kb.mp3", secs)["section"] == "01"
    assert s0.match_section("b.mp3", secs) is None


def test_respiro_info(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"audio_file": "a/b/x.mp3", "breaths": [[0.0, 0.5], [3.0, 3.25]]}) + "\n")
    i = s0.respiro_info(p)
    assert i["n_breaths"] == 2 and i["breath_time"] == 0.75 and i["first_starts_at_zero"]
    assert s0.respiro_info(tmp_path / "nope.jsonl") == {"present": False}


def test_asr_readers(tmp_path):
    vv = tmp_path / "x.json"
    vv.write_text(json.dumps([
        {"Start": 0, "End": 2.5, "Content": "This is a LibriVox recording.", "Speaker": 0},
        {"Start": 2.5, "End": 4.0, "Content": "Chapter one.", "Speaker": 1},
        {"Start": 4.0, "End": 5.0, "Content": "[Silence]"},
    ]))
    i = s0.vibevoice_info(vv)
    assert i["present"] and i["n_segments"] == 3 and i["n_speakers"] == 2 and i["end"] == 5.0
    assert i["nonspeech_tags"] == {"[Silence]": 1}
    assert s0.vibevoice_info(tmp_path / "missing.json") == {"present": False}

    wx = tmp_path / "w.json"
    wx.write_text(json.dumps({"language": "en", "segments": [
        {"start": 0.1, "end": 0.9, "text": " Hi there", "words": [
            {"word": "Hi", "start": 0.1, "end": 0.3, "score": 0.9},
            {"word": "there", "start": 0.4, "end": 0.9, "score": 0.8}]}],
        "word_segments": [{"word": "Hi", "start": 0.1, "end": 0.3},
                          {"word": "there", "start": 0.4, "end": 0.9}]}))
    i = s0.whisperx_info(wx)
    assert i["n_words"] == 2 and i["n_words_untimed"] == 0 and i["language"] == "en"


@pytest.mark.skipif(not Path("/data/around-the-world-in-80-days").exists(), reason="needs /data")
def test_survey_one_reading_end_to_end(tmp_path):
    rd = "around-the-world-in-80-days-by-jules-verne"
    assert s0.main(["--out", str(tmp_path), "--reading", rd]) == 0
    rows = [json.loads(x) for x in (tmp_path / "inventory.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    r = rows[0]
    assert r["reading_id"] == "lv714" and r["gutenberg_ids"] == [103]
    assert r["n_sections_local"] == r["n_sections_listed"] == 37
    s = r["sections"][0]
    assert s["sampling_rate"] == 44100 and s["num_samples"] > 0
    assert abs(s["duration"] - 743.2) < 2.0
    assert s["asr"]["whisperx"]["present"] and s["asr"]["vibevoice"]["present"]
    if s0.RESPIRO_ROOT.exists():
        assert s["asr"]["respiro"]["present"]
    # second run reuses the probe cache and is byte-identical (deterministic)
    first = (tmp_path / "inventory.jsonl").read_bytes()
    assert s0.main(["--out", str(tmp_path), "--reading", rd]) == 0
    assert (tmp_path / "inventory.jsonl").read_bytes() == first
