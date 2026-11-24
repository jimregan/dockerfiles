# Stage 7: segmentation (committed)

`uv run python -m librivox_mult.stage7_segment [--work W] [--reading ID] [--force]` writes
`/out/stage7/<work>/<reading_id>.json.gz` from Stage 5 (alignment) and Stage 6 (pauses).

## Segmentations

- **`S`**: one segment per Stage 1 sentence. Prose sentences use `{chapter:03d}/{para:04d}/{sent:02d}`;
  verse sentences use the poem hierarchy, `{poem:03d}/s{n:02d}`.
- **`L`**: one segment per verse line, `{poem:03d}/{stanza:02d}/{line:02d}`. It is made only
  for readings whose text contains verse. Non-verse material in those readings falls back to
  sentences, so `L` still covers the whole reading.
- **Headers and footers**: each matched element (`H01`…, `F01`…) is its own segment in both.

Every reference row is in exactly one segment per segmentation. The `edits` list keeps every
aligned pair `[kind, ref_row, alignment_index]`, so `del` rows are covered too.

## Assigning words

- Segments are maximal runs of aligned pairs with the same unit, in spoken order.
- Inserted words take the unit of the **next** reference word. Before the first unit they
  go to the first segment, and after the last, to the last one.
- **Unreferenced speech**: a run of 15 or more consecutive inserted words becomes its own
  segment, `X01`… (`unit: "X"`, no `text_id`, empty `text`). The cut is measured from
  energy, as in Stage 6. Examples: a story read that isn't in the mapped text, long
  ad-libs, and the 5 sections whose body Stage 5 dropped.
- If a unit comes back after another unit has started, it gets a second segment with id
  suffix `.2`. This only happens with reordered reading. A unit already used by an earlier
  section of the reading (alternate versions, overlapping sections) also gets the next `.N`.

## Boundaries

The cut between segments A and B uses the Stage 6 measurement at A's last reference word:

- **`pause`**: the longest quiet run in the gap is at least 0.1 s. A ends at
  `silence_start + pad` and B starts at `silence_end − pad`, with
  `pad = min(0.25 s, silence / 2)`. Part of a long silence therefore belongs to neither
  segment.
- **`forced`**: no such silence. A and B meet at the midpoint between A's last word end
  and B's first word start.
- **`edge`**: a section's first or last segment. It extends 0.25 s beyond its first or
  last word, clipped to the recording.

A segment is never cut inside one of its own words: start and end are widened to its first
and last word if needed. `pause_before` and `pause_after` are the Stage 6 `pause` (quiet
extent, breaths included).

A unit with no spoken words (a skipped sentence, the unread contents) is kept with
`spoken: false` and zero duration, at the cut point.

## Fields per segment

`id` (suffix after `{work}/{reader}/`), `text_id`, `unit` (`S`/`L`/`H`/`F`), `start`,
`duration`, `spoken`, `text` (source slice with punctuation; the template source text for
headers), `src_span`, `verbatim`, `alignment` (spoken words, partials marked `-`), `edits`,
`boundary`, `pause_before`/`pause_after`, `dysfluencies` (`tokens` index into `alignment`,
`ref_tokens` are Stage 1 tokens), `nsw`, `confidence`, `header_footer`.

`confidence`:
- An `S` segment holding its whole sentence takes Stage 5's sentence confidence.
- Otherwise: cor/n × mean wav2vec2 score.

## IDs (decided 2026-11-09; rules in `ids.md`)

- `{reader}` in supervision ids = the Stage 0 `reading_id` (`lv<id>[-s<NN>]`), which is unique
  per reading.
- In a reading that covers several texts (collections), body ids carry the text:
  `S/{text}/{chapter}/…`. Without it, two texts' `001/0001/01` would collide.
- Poem ordinal padded to 3 digits, like chapters.
