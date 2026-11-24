import numpy as np

from librivox_mult import stage6_pauses as s6


def test_silence_in_longest_run_and_extent():
    db = np.full(300, -10.0, dtype=np.float16)
    db[100:130] = -60  # 0.30 s quiet
    db[130:135] = -20  # a breath
    db[135:150] = -60
    (s0, s1), pause = s6.silence_in(db, -40.0, 0.95, 1.6)
    assert (s0, s1) == (1.0, 1.3)
    assert pause == 0.5
    assert s6.silence_in(db, -40.0, 0.2, 0.5) is None


def test_boundaries_attach_insertions_to_next_unit():
    s1 = {"sent": [0, 0, 1, 1], "line": [None] * 4, "block": [0, 0, 0, 0], "in_poem": [False]}
    sec = {"elements": [],
           "ref": [["a", 0, -1, -1], ["b", 1, -1, -1], ["c", 2, -1, -1], ["d", 3, -1, -1]],
           "words": [[0, 0, "cor", 0.0, 0.2, 1], [1, 1, "cor", 0.3, 0.5, 1],
                     [-1, 2, "ins", 1.5, 1.6, 1], [2, 3, "cor", 1.7, 1.9, 1], [3, 4, "cor", 2.0, 2.2, 1]]}
    db = np.full(300, -10.0, dtype=np.float16)
    db[52:148] = -60
    bs, base = s6.boundaries(sec, s1, db, -40.0, [])
    assert len(bs) == 1 and bs[0]["kinds"] == ["S"]
    assert bs[0]["prev_end"] == 0.5 and bs[0]["next_start"] == 1.5  # the inserted word starts unit 2
    assert bs[0]["silence"] == (0.52, 1.48)
    assert len(base) == 2


def test_classify():
    assert s6.classify({"sent_end": 0.9, "line_end_only": 0.8}) == "both"
    assert s6.classify({"sent_end": 0.9, "line_end_only": 0.1}) == "sentences"
    assert s6.classify({"sent_end": 0.2, "line_end_only": 0.7}) == "lines"
    assert s6.classify({"sent_end": 0.2, "line_end_only": None}) == "neither"
