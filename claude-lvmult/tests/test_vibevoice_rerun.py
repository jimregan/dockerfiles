from librivox_mult import vibevoice_rerun as vr


def words_with_gaps(duration, gap_at):
    """A word every 0.5 s (0.4 s long) with a 2 s pause at each time in gap_at."""
    out, t = [], 0.0
    while t < duration - 1:
        if any(abs(t - g) < 0.25 for g in gap_at):
            t += 2.0
            continue
        out.append({"start": t, "end": t + 0.4})
        t += 0.5
    return out


def test_short_file_single_piece():
    assert vr.plan_cuts(1500, [], 1800) == [{"index": 1, "start": 0.0, "end": 1500, "gap": None}]


def test_cuts_snap_to_pauses_and_respect_cap():
    dur = 4400.0  # 73 min -> 3 pieces at <=30 min
    w = words_with_gaps(dur, gap_at=[1500.0, 2950.0])
    ps = vr.plan_cuts(dur, w, 1800)
    assert len(ps) == 3
    assert all(p["end"] - p["start"] <= 1800 for p in ps)
    assert ps[0]["end"] == ps[1]["start"] and ps[-1]["end"] == dur
    # cuts land inside the long pauses, not between ordinary words
    assert 1500 < ps[0]["end"] < 1502.1 and ps[0]["gap"] >= 1.5
    assert 2950 < ps[1]["end"] < 2952.1


def test_stitch_offsets_and_speakers():
    pieces = [{"index": 1, "start": 0.0, "end": 100.0}, {"index": 2, "start": 100.0, "end": 180.0}]
    outs = [[{"Start": 0.0, "End": 5.0, "Speaker": 0, "Content": "a"}],
            [{"Start": 1.0, "End": 4.0, "Speaker": 0, "Content": "b"}, {"Start": 4.0, "End": 6.0, "Content": "[Music]"}]]
    s = vr.stitch(pieces, outs)
    assert [(x["Start"], x["End"], x.get("Speaker"), x["Piece"]) for x in s] == [
        (0.0, 5.0, "1:0", 1), (101.0, 104.0, "2:0", 2), (104.0, 106.0, None, 2)]
