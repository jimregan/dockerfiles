# Stage 5: word alignment

`uv run python -m librivox_mult.stage5_align [--work W] [--reading ID] [--force] [--jobs N]`
(CPU, ~3 min for everything with 24 jobs). Writes `/out/stage5/<work>/<reading_id>.json.gz`.
Skips a reading whose Stage 2/3/4 inputs and `VERSION` are unchanged.

## Sequences

- **Reference** per section: the Stage 2 header elements matched in the audio (the
  matched variant's words, in audio order), the Stage 4 body words (`[word, token, nsw]`),
  then the footer elements. Each row is `[word, stage1_token, nsw, element]`
  (`element` = -1 for body, else an index into `elements`, ids `H01`…/`F01`…).
  Unmatched templates are not in the reference: they are expectations, not text.
- **Hypothesis:** every wav2vec2 word in the file (lower case, `'` kept, edge `'` stripped).

## DP

Kaldi `align_ctm_ref.smith_waterman_alignment` scoring (match +1, sub/ins/del −1) and its
tie-breaking (diagonal first, then del, then ins, each only if strictly better), made
**global**: traced back from the end of both sequences, so unsaid reference words are
`del` and every hyp word appears. A repeated word's *first* copy is the insertion, so
inserted material attaches to what follows. A test checks it gives the vendored DP's
alignment whenever that one covers the whole reference.

Speed: anchors are the interiors (edge word dropped) of exact runs of ≥4 words in a
rapidfuzz Levenshtein alignment, and the DP runs between them. A chunk over 4 M cells falls
back to Levenshtein opcodes (counted as `big_chunks`; 1 in the corpus).

- **Out-of-order reading:** if under 90% of ref words are `cor` (or Stage 3 found
  out-of-order spans), the body blocks are re-sorted by the median hyp position of their
  rare 4-grams and realigned. The better alignment is kept (`reordered: true`).
- **Stage 3 suspect sections** (density < 0.2): the body is dropped and only headers and
  footers are aligned (`body_dropped`).

## Labels (one per maximal non-`cor` run)

| Region | Label | Rule |
|---|---|---|
| header/footer | `template` / `template_sub` / `extra` | template words unsaid / misheard / ad-lib words |
| body | `orth` | same letters ignoring spaces and `'` (`bed post` / `bedpost`) |
| | `contraction` | one side contracted (`it's` / `it is`, `won't` / `will not`) |
| | `unread_heading`, `unread_marker` | deleted heading/contents words; deleted `[1]` footnote markers |
| | `skip` | other deletions |
| | `repetition` | only insertions, equal to the ref words just before or just after |
| | `partial` | insertions = the next ref words up to a proper prefix of the following one (`the fo the fogg`) |
| | `revision` | insertions that start with the next ref words, then diverge (`he walked he went`), or a partial followed by other material |
| | `reorder` | an inserted word deleted within 6 positions (`he seemed always` for `he always seemed`); both runs |
| | `ins` / `sub` | the rest |

Nothing is removed or rewritten: `words` keeps every hyp word with its time and score.

## WhisperX

- **Vote** per body run (`whisperx`): WhisperX words with midpoints in the run's time span
  compared by letters with the run's ref words (`ref`: WhisperX has the book, the usual
  sign that wav2vec2 heard a real dysfluency) and its hyp words (`w2v`: both heard it),
  else `other`. `None` when there's no WhisperX file.
- **Per-word agreement:** body ref words that are `equal` in a Levenshtein alignment of
  the body to the WhisperX words. Not used for headers.

## Output per section

- `ref`, `elements`.
- `words`: `[ref_i, hyp_i, kind, start, end, score]` in order (`-1` / `null` where absent).
- `spans`: `type, region, pairs [p0, p1), ref, hyp, w2v, start, end, whisperx`.
- `units`: per Stage 1 sentence index (body) or element id:
  `n, cor, whisperx_eq, mean_score, confidence`.
  - `confidence` = (cor + whisperx_eq) / 2n × mean wav2vec2 score of the `cor` words.
  - Headers use cor / n × mean score.
- `counts`, `big_chunks`.
