# Stage 6: pauses

`src/librivox_mult/stage6_pauses.py`, CPU, two steps:

1. `energy`: ffmpeg decode to 16 kHz mono, 10 ms frame RMS in dB, saved as float16 in
   `/out/stage6/energy/<work>/<slug>/<stem>.npy`. Existing files are skipped.
2. `pauses`: writes `/out/stage6/<work>/<reading_id>.json.gz` plus a
   `<reading_id>.summary.json` per reading. Recomputed when the Stage 5 output or
   `VERSION` changes.

## Boundaries

A boundary is a point between consecutive aligned ref words that changes one or more units:

| Kind | Unit |
|---|---|
| `H` | header/footer element, or header ↔ body |
| `S` | Stage 1 sentence |
| `L` | verse line (block, line) |
| `P` | block |

Each boundary records:
- `prev_end`: the end of the last wav2vec2 word up to the unit's last ref word.
- `next_start`: the start of the next wav2vec2 word, so inserted material belongs to
  the next unit (Stage 7's rule).
- `gap`.
- `silence`: the longest quiet run inside [prev_end, next_start + 20 ms]. This is where
  Stage 7 cuts.
- `pause`: the first-to-last quiet frame extent, so a breath inside a pause stays in it.
- `breath`: whether a Respiro breath overlaps `silence`. This is recorded only; nothing
  depends on it (per Jim, Respiro is only used to agree with a sentence boundary).

A frame is quiet below floor + 0.2 × (speech − floor), where floor and speech are the
file's 10th and 90th energy percentiles. Gaps between `cor` word pairs inside a unit are
measured the same way, as each reading's baseline (`p_word`).

wav2vec2 CTC word times are peaky (a word's end is its last emitted letter), so `gap`
overstates the silence. That's why the pause comes from energy, not from `gap`.

## Behaviour per reading

Rates = share of boundaries with a pause ≥ 0.15 s. Header boundaries are excluded.
- `p_sent_end`: sentence ends.
- `p_line_end_only`: verse line ends that aren't sentence ends.
- `p_sent_mid_line`: sentence ends inside a verse line.
- `p_block`: block ends that are neither.
- `p_word`: baseline.

`behaviour` is the reading's class:
- `both`: `p_sent_end` ≥ 0.5 and `p_line_end_only` ≥ 0.5.
- `sentences`: only `p_sent_end` ≥ 0.5.
- `lines`: only `p_line_end_only` ≥ 0.5.
- `neither`: neither rate reaches 0.5.

Readings with no verse lines can only be `sentences` or `neither`.
