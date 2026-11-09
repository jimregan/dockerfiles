from librivox_mult import stage7_segment as s7


def doc():
    text = "One two. Three four."
    return {"text": text,
            "tokens": {"start": [0, 4, 9, 15], "end": [3, 8, 14, 20], "sent": [0, 0, 1, 1],
                       "line": [None] * 4, "block": [0, 0, 0, 0]},
            "sentences": [{"unit": "para", "division": 1, "para": 1, "n": 1},
                          {"unit": "para", "division": 1, "para": 1, "n": 2}],
            "blocks": [{"kind": "para"}]}


def sec5():
    rows = [["one", 0, -1, -1], ["two", 1, -1, -1], ["three", 2, -1, -1], ["four", 3, -1, -1]]
    words = [[0, 0, "cor", 1.0, 1.2, 0.9], [1, 1, "cor", 1.3, 1.5, 0.9],
             [-1, 2, "ins", 2.0, 2.1, 0.5], [2, 3, "cor", 2.2, 2.5, 0.9], [3, 4, "cor", 2.6, 2.9, 0.9]]
    return {"words": words, "ref": rows, "elements": [], "text": "pg1", "units": [],
            "spans": [{"type": "partial", "pairs": [2, 3], "hyp": [2], "ref": [], "whisperx": "ref"}]}


HYP = ["one", "two", "thr", "three", "four"]


def test_pause_cut_and_insertion_goes_to_next():
    b6 = {"boundaries": [{"p": 1, "prev_end": 1.5, "next_start": 2.0, "silence": [1.55, 1.95], "pause": 0.4}]}
    segs = s7.segment_section(sec5(), b6, None, HYP, doc(), "S", "01", 10.0, False)
    a, b = segs
    assert a["id"] == "S/001/0001/01" and b["text_id"] == "pg1/001/0001/02"
    assert b["verbatim"] == "thr- three four"
    assert b["dysfluencies"][0]["type"] == "partial" and b["dysfluencies"][0]["tokens"] == [0]
    assert a["boundary"]["end"] == "pause" and abs(a["start"] + a["duration"] - 1.75) < 1e-6
    assert b["start"] == 1.75 and a["text"] == "One two." and b["src_span"] == [9, 20]
    assert s7.check(segs, 4, 10.0) == []


def test_forced_cut_without_silence():
    b6 = {"boundaries": [{"p": 1, "prev_end": 1.5, "next_start": 2.0, "silence": None, "pause": 0.0}]}
    a, b = s7.segment_section(sec5(), b6, None, HYP, doc(), "S", "01", 10.0, False)
    assert a["boundary"]["end"] == "forced" and b["start"] == 1.75


def test_long_insertion_is_its_own_segment():
    d = sec5()
    extra = [[-1, 5 + k, "ins", 1.55 + 0.02 * k, 1.56 + 0.02 * k, 0.9] for k in range(s7.X_MIN_WORDS)]
    d["words"] = d["words"][:2] + extra + d["words"][2:]
    d["spans"] = []
    hyp = HYP + ["x"] * s7.X_MIN_WORDS
    segs = s7.segment_section(d, None, None, hyp, doc(), "S", "01", 100.0, False,
                              measure=lambda p: None)
    assert [s["unit"] for s in segs] == ["S", "X", "S"]
    assert segs[1]["id"] == "01/X01" and segs[1]["text_id"] is None and len(segs[1]["alignment"]) == s7.X_MIN_WORDS + 1
    assert s7.check(segs, 4, 100.0) == []
