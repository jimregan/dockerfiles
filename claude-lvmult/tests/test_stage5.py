import importlib.util
import random
from pathlib import Path

from librivox_mult import stage5_align as s5

_spec = importlib.util.spec_from_file_location(
    "kaldi_sw", Path(__file__).parents[1] / "vendor/sync_asr/align_ctm_ref.py")
kaldi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kaldi)


def kaldi_pairs(ref, hyp):
    sim = lambda a, b: 1 if a == b else -1  # noqa: E731
    raw = kaldi.smith_waterman_alignment(ref, hyp, sim, -1, -1)[0]
    return [(r1 - 1 if r1 > r0 else -1, h1 - 1 if h1 > h0 else -1) for _, _, r0, h0, r1, h1 in raw]


def test_matches_vendored_kaldi_when_hyp_covers_ref():
    rng = random.Random(0)
    vocab = list("abcdef")
    n = 0
    for _ in range(300):
        ref = [rng.choice(vocab) for _ in range(rng.randint(2, 12))]
        hyp = list(ref)
        for _ in range(rng.randint(0, 3)):
            k = rng.randrange(1, len(hyp))
            op = rng.choice("sid")
            if op == "s":
                hyp[k] = rng.choice(vocab)
            elif op == "i":
                hyp.insert(k, rng.choice(vocab))
            elif len(hyp) > 2:
                del hyp[k]
        ours = s5.kaldi_global(ref, hyp)
        theirs = kaldi_pairs(ref, hyp)
        # the vendored DP is local on the ref; compare when it covers all of it
        if [i for i, _ in theirs if i >= 0] == list(range(len(ref))):
            assert ours == theirs, (ref, hyp)
            n += 1
    assert n > 200


def test_global_keeps_flanking_ref_as_del():
    ref = "this is a librivox recording chapter one phileas fogg".split()
    hyp = "chapter one phileas fogg".split()
    pairs = s5.kaldi_global(ref, hyp)
    assert [i for i, _ in pairs] == list(range(len(ref)))
    assert sum(j < 0 for _, j in pairs) == 5


def test_repeated_word_first_copy_is_inserted():
    pairs = s5.kaldi_global(["the", "cat"], ["the", "the", "cat"])
    assert pairs == [(-1, 0), (0, 1), (1, 2)]


def test_anchored_alignment_equals_full_dp():
    rng = random.Random(1)
    ref = [f"w{rng.randrange(400)}" for _ in range(600)]
    hyp = list(ref)
    for k in sorted(rng.sample(range(len(hyp)), 40), reverse=True):
        hyp[k:k] = ["uh"] if k % 2 else []
        if k % 3 == 0:
            hyp[k] = "x"
    full = s5.kaldi_global(ref, hyp)
    anchored, big = s5.align(ref, hyp)
    assert big == 0
    assert [p for p in anchored if p[0] >= 0 and p[1] >= 0 and ref[p[0]] == hyp[p[1]]] == \
           [p for p in full if p[0] >= 0 and p[1] >= 0 and ref[p[0]] == hyp[p[1]]]


def lab(ref, hyp, ref_kind=None):
    ref, hyp = ref.split(), hyp.split()
    pairs, _ = s5.align(ref, hyp)
    out = []
    for p0, p1 in s5.runs(pairs, ref, hyp):
        before = [i for i, _ in pairs[:p0] if i >= 0][-3:]
        after = [i for i, _ in pairs[p1:] if i >= 0][:3]
        out.append(s5.label_run(ref, hyp, pairs[p0:p1], before, after, ref_kind or {}))
    return out


def test_labels():
    assert lab("he went to the shop", "he went to to the shop") == ["repetition"]
    assert lab("phileas fogg had won", "phileas fo fogg had won") == ["partial"]
    assert lab("he went to the shop", "he walked he went to the shop") == ["revision"]
    assert lab("he went to the shop", "he went the shop") == ["skip"]
    assert lab("he went to the shop", "he went into the shop") == ["sub"]
    assert lab("he went to the shop", "he went um to the shop") == ["ins"]
    assert lab("at the bedpost there", "at the bed post there") == ["orth"]
    assert lab("see note one here", "see note here", {2: "marker"}) == ["unread_marker"]
    assert lab("is that it's a good", "is that it is a good") == ["contraction"]
    assert lab("i won't go there", "i will not go there") == ["contraction"]
    assert lab("the fogg had", "the fo the fogg had") == ["partial"]
    assert lab("of the inns of court either", "of the inns of the court either") == ["ins"]
    # a prefix of a word that isn't then said in full is not a partial
    assert lab("you only asked a hundred", "you only ask for a hundred") == ["sub"]


def test_reorder_blocks_follows_reading_order():
    paras = [[f"p{b}w{i}" for i in range(12)] for b in range(4)]
    rows = [["chapter", -1, -1, 0]]
    tok = []
    for b, p in enumerate(paras):
        for w in p:
            rows.append([w, len(tok), -1, -1])
            tok.append(b)
    hyp = ["chapter"] + paras[2] + paras[0] + paras[1] + paras[3]
    order = s5.reorder_blocks(rows, hyp, tok)
    assert [rows[i][0] for i in order] == hyp
    assert s5.reorder_blocks(rows, ["chapter"] + sum(paras, []), tok) is None


def test_reorder():
    ref = "the man though he always seemed to know".split()
    hyp = "the man though he seemed always to know".split()
    pairs, _ = s5.align(ref, hyp)
    spans = [{"region": "body", "pairs": [a, b], "ref": [i for i, _ in pairs[a:b] if i >= 0], "hyp": [j for _, j in pairs[a:b] if j >= 0],
              "type": "ins" if all(i < 0 for i, _ in pairs[a:b]) else "skip"} for a, b in s5.runs(pairs, ref, hyp)]
    s5.mark_reorders(spans, ref, hyp)
    assert [x["type"] for x in spans] == ["reorder", "reorder"]
