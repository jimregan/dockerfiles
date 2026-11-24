from librivox_mult import stage3_map as s3


def make_text(paras: list[str], kinds: list[str] | None = None) -> dict:
    """Minimal stand-in for a Stage 1 doc + the word index load_text builds."""
    text, blocks, tk = "", [], {"surface": [], "start": [], "punct": [], "block": []}
    for bi, p in enumerate(paras):
        if text:
            text += "\n\n"
        start = len(text)
        for w in p.split():
            tk["surface"].append(w)
            tk["start"].append(len(text))
            tk["punct"].append(0)
            tk["block"].append(bi)
            text += w + " "
        text = text.rstrip()
        blocks.append({"kind": (kinds or ["para"] * len(paras))[bi], "division": 1, "para": bi + 1,
                       "char_start": start, "char_end": len(text)})
    doc = {"text": text, "blocks": blocks, "tokens": tk, "divisions": [{"division": 1, "heading": ["X"]}]}
    words = [w.lower() for w in tk["surface"]]
    index = {}
    for j in range(len(words) - s3.K + 1):
        index.setdefault(tuple(words[j:j + s3.K]), []).append(j)
    return {"id": "pg1", "doc": doc, "words": words, "tok_of": list(range(len(words))), "index": index}


def test_lis_chain():
    pts = [(0, 10), (1, 11), (2, 5), (3, 12), (4, 13), (5, 1)]
    assert s3.lis_chain(pts) == [(0, 10), (1, 11), (3, 12), (4, 13)]


PARAS = [" ".join(f"a{i}w{j}" for j in range(30)) for i in range(10)]


def test_map_section_ignores_stray_hits():
    t = make_text(PARAS)
    # audio reads paragraphs 3-5, plus a stray 4-gram from paragraph 9 at the start
    hyp = PARAS[9].split()[:4] + " ".join(PARAS[3:6]).split()
    m = s3.map_section(hyp, t)
    assert m["n_spans"] == 1
    assert t["doc"]["tokens"]["block"][t["tok_of"][m["text_first"]]] == 3
    assert t["doc"]["tokens"]["block"][t["tok_of"][m["text_last"]]] == 5
    assert m["density"] > 0.9


def test_map_section_out_of_order_spans():
    paras = [" ".join(f"f{i}w{j}" for j in range(80)) for i in range(40)]
    t = make_text(paras)
    hyp = paras[30].split() + paras[2].split()  # fables read in a different order
    m = s3.map_section(hyp, t)
    assert m["n_spans"] == 2 and m["density"] > 0.9


def sec(section, b0, b1, before=0, after=0):
    return {"section": section, "stem": section, "text": "pg1", "n_spans": 1,
            "start": {"block": b0}, "end": {"block": b1},
            "hyp_unanchored_before": before, "hyp_unanchored_after": after}


def test_snap_edges_assigns_short_paragraphs():
    paras = PARAS[:5] + ["Stas' head swam."] + PARAS[5:]
    t = make_text(paras)
    a, b = sec("01", 0, 4, after=4), sec("02", 6, 9)
    s3.snap_edges([a, b], {"pg1": t})
    assert a["end"]["block"] == 5 and a["snapped_end_blocks"] == 1
    assert "flags" not in b


def test_snap_edges_flags_real_gaps():
    t = make_text(PARAS)
    a, b = sec("01", 0, 2), sec("02", 6, 9)  # paragraphs 3-5 (90 words) not read
    s3.snap_edges([a, b], {"pg1": t})
    assert b["gap_unaccounted_words"] == 90 and b["flags"] == ["gap_3_blocks_90_words"]


def test_same_span():
    assert s3.same_span(sec("01", 10, 20), sec("02", 10, 21))
    assert not s3.same_span(sec("01", 10, 20), sec("02", 18, 40))
