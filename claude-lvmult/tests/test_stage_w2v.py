import numpy as np
import pytest

from librivox_mult import stage_w2v as w


@pytest.mark.parametrize("secs", [0.01, 0.5, 29.99, 30.0, 41.3, 95.0, 743.2045])
@pytest.mark.parametrize("chunk,ctx", [(30.0, 5.0), (10.0, 2.0)])
def test_windows_tile_all_frames_once(secs, chunk, ctx):
    n = int(secs * w.SR)
    total = w.n_frames(n)
    cover = np.zeros(total, dtype=int)
    for win in w.plan_windows(n, chunk, ctx):
        assert 0 <= win["w0"] < win["w1"] <= n
        assert win["w0"] % w.HOP == 0
        k = win["keep_to"] - win["keep_from"]
        assert k >= 0 and win["keep_to"] <= w.n_frames(win["w1"] - win["w0"])
        cover[win["dst"] : win["dst"] + k] += 1
        # kept frames of the window line up with global frame times
        assert win["w0"] // w.HOP + win["keep_from"] == win["dst"]
    assert (cover == 1).all()


def test_windows_have_context():
    n = 120 * w.SR
    wins = w.plan_windows(n, 30.0, 5.0)
    assert len(wins) == 4
    assert wins[0]["keep_from"] == 0 and wins[1]["keep_from"] == 5 * w.SR // w.HOP


def _logp(seq, vocab):
    idx = {t: i for i, t in enumerate(vocab)}
    lp = np.full((len(seq), len(vocab)), np.log(0.01), dtype=np.float32)
    for t, s in enumerate(seq):
        lp[t, idx[s]] = np.log(0.9)
    return lp


def test_greedy_ctc_collapse_and_words():
    vocab = ["<pad>", "|", "F", "O", "G", "H", "I"]
    id2tok = dict(enumerate(vocab))
    # "FO" partial, then "FOGG" (GG needs a blank between), then "HI"
    seq = ["F", "F", "O", "|", "|", "F", "O", "G", "<pad>", "G", "|", "<pad>", "H", "I", "I"]
    tokens, words = w.greedy_ctc(_logp(seq, vocab), id2tok, blank=0)
    assert [x["word"] for x in words] == ["FO", "FOGG", "HI"]
    assert tokens[0][:3] == ["F", 0, 1]
    assert words[0]["start"] == 0.0 and words[0]["end"] == round(3 * w.FRAME_SHIFT, 3)
    assert words[2]["end"] == round(15 * w.FRAME_SHIFT, 3)
    assert 0.8 < words[1]["score"] <= 1.0


def test_greedy_ctc_empty():
    assert w.greedy_ctc(np.zeros((0, 4), dtype=np.float32), {}, 0) == ([], [])


def test_is_done_requires_both_outputs(tmp_path):
    j = tmp_path / "a.b_128kb.json"
    assert w.logp_path(j).name == "a.b_128kb.logp.npy"
    assert not w.is_done(j)
    j.write_text("{}")
    assert not w.is_done(j)  # json alone (should not happen; logp is written first)
    w.logp_path(j).write_bytes(b"x")
    assert w.is_done(j)
    j.unlink()
    assert not w.is_done(j)  # interrupted after logp, before json: redo
