# Stage 3: section → text span

`uv run lvm-map [--work W] [--reading SLUG] [--force] [--jobs N]`
(`src/librivox_mult/stage3_map.py`)

- **Output:** `/out/stage3/<work>/<reading_id>.json`, plus review tables in
  `/out/review/stage3/`.
- **Runtime:** about 45 s for the corpus.
- **Rerunning:** a reading is recomputed when its inventory row changes, and outputs of
  readings that no longer exist are pruned.

## Method

1. **Audio words:** the wav2vec2 words between the Stage 2 header (last header match)
   and footer (first footer match), lower-cased in the model alphabet.
2. **Text words:** the Stage 1 word tokens of the reading's source text through
   `spoken.spoken_words` ("1872" → "eighteen seventy two"), each pointing back to its
   token. Contents and separator blocks are not indexed, because a contents entry
   would anchor the chapter title.
3. **Anchors:** word 4-grams shared by audio and text, using only 4-grams that occur at
   most 8 times in the text.
4. **Runs:** the longest chain of anchors increasing in both audio and text position
   (LIS), split wherever the text jumps more than 5× the audio gap + 300 words.
   - Runs are extracted repeatedly: take the best, remove points in its audio range,
     repeat. A section that reads pieces in a different order from the book (fable
     collections) therefore gets several **spans**.
   - Runs with at least 10 anchors (and 10% of the largest) count.
5. **Span:** first/last anchor → token, block, division, paragraph. Also recorded:
   `density` (share of audio words covered by anchors), `start_time`/`end_time`, and
   the unanchored audio words before the first and after the last anchor.
   - With several spans, `start`/`end` are the largest span's, and `ranges` lists the
     block ranges `[b0, b1]` covering all spans in text order. Spans fewer than 8 blocks
     apart merge into one range, including the blocks between them. Stage 4 builds the
     reference from `ranges` (stage3-1.0). Before this, only the largest span was used,
     and 1.8 h of book text in 20 sections fell into Stage 7's unreferenced X segments.
6. **Edge snapping between consecutive sections** (single-span, same text, in section
   order). Readable blocks (para/verse/lines) between section A's end and B's start are
   assigned:
   - to A, as far as A's unanchored tail can hold them (a block of W words needs about
     0.6·W unanchored words);
   - else to B, if B's unanchored head can hold them, or if the gap text fuzzily
     matches B's Stage 2 header words (a chapter title in a `<p>` that was spoken in
     the header);
   - else it is a **gap**: text the reader skipped, or audio that isn't there. It's
     flagged with its word count.

   Short paragraphs ("Stas' head swam.") can't contain a 4-gram anchor; snapping is
   what puts them back in a section.

## Flags (`/out/review/stage3/flagged.tsv`)

| Flag | Meaning |
|---|---|
| `no_source_text` | the reading has no Gutenberg text (319 sections; see STATUS) |
| `no_anchors` | nothing matched; e.g. a cast-list section with no book text |
| `low_density` | under half of the audio words are anchored |
| `suspect_source` | the reading's median density is under 0.2, so the listed source is probably the wrong book |
| `out_of_order_N_spans` | the section reads several pieces of the text out of order |
| `gap_N_blocks_N_words` | text between consecutive sections that neither side accounts for |
| `overlap_or_reversed` | the section starts before the previous one ends. Not flagged: alternate versions (Monte Cristo), a shared block, or the same text read by another reader (dialect collection) |
| `unanchored_start/end` | more than 60 unanchored audio words at the edge: ad-libs, a preface not in the text, or the text missing |
