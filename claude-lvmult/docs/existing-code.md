# Existing code in `/src-projects` (reviewed 2026-11-05)

Four projects. `/src-projects` is read-only, and anything kept is copied verbatim
into `vendor/` (provenance in `vendor/README.md`) and adapted in `src/` when the
stage that needs it is built.

## 1. `sync-asr/` contains the aligner (git, HEAD `e774b51`, 2026-09-27)

A general toolkit (2022–2026) for aligning ASR output to text. Most of it is Swedish
Riksdag / Kashubian / Hungarian work. The aligner is in two layers:

**`sync_asr/kaldi/align_ctm_ref.py`**: Kaldi's `steps/cleanup/internal/align_ctm_ref.py`
(Manohar/Gao, Apache-2.0), diffed against `/opt/kaldi` on the worker. It is near-verbatim:
inlined arg helpers, CTM channel 0 or 1, and an unused `hfjson` choice.
- Pure Python. No Kaldi binaries.
- `smith_waterman_alignment(ref, hyp, sim, del, ins, eps, align_full_hyp)`: match +1,
  sub/ins/del −1 by default.
- `get_ctm_edits` writes `file chan start dur hyp conf ref edit`. Edit is
  cor/sub/ins/del/sil, deletions get dur 0, and there is **no reference index**.
- CLI: `python align_ctm_ref.py --ref text --hyp ctm --output ctm_edits [--*-penalty]`.
- **Local on the reference** (verified): the traceback starts at the best ref position at
  the end of the hyp, so ref words before or after the matched span are *silently
  dropped*, not emitted as `del`. With ref "this is a librivox recording chapter one phileas
  fogg … end of chapter one" and hyp "phileas fo fogg …", the header and footer vanish. That
  is right for Kaldi cleanup (a hyp segment against a long ref) and wrong for
  skips and headers/footers here.
- **Slow**: list-of-lists DP plus eagerly formatted `logger.debug` in the inner loop.
  800×800 takes 10 s (verified); 2000×2000 takes 72 s and 0.6 GB (subagent run). A chapter
  of 5k words is minutes and GBs, so it needs a numpy/banded rewrite or Stage 3 anchors.
- Exact string equality. **Casing and punctuation are not ignored here.**
- Inherited bugs: `raise Warning` then an unreachable `continue` (one bad recording aborts
  the run), `logger.warn`, module-level globals.

**Passes of the aligner across branches.** Jim confirmed the Kaldi Smith–Waterman in
sync-asr is the aligner, with several passes at it. The working tree is on branch
`extractions`, not `main`. `align_ctm_ref.py` exists in four versions (by blob hash). The
DP core is the same in all of them; only the word comparison differs:

| Version | Branches | Comparison |
|---|---|---|
| `f61fa139` Kaldi | 22 older branches (sentence-extraction, whisperx-reverse, hungarian…) | exact match; CTM channel must be 1 |
| `2b5340e8` | `ignore-punctuation-in-aligner` (2024-03-20, "does not work currently") | hyp edge-punctuation stripped and lowercased; ref compared raw |
| `073ec86f` | `main` (PR #60, 2026-09-10, same "does not work currently" note) | `--ignore-punctuation` (default true): `clean_text` strips **edge** punctuation and lowercases **both** sides. Inconsistent defaults: `get_edit_type` False, `get_ctm_edits` True |
| `fae56d77` | `extractions` (reviewed above; newest) | exact match in the Kaldi file; case/punct handling moved into `alignment.py` (strips **all** `P*`) |

So the DP is settled, and the open part is the comparison. `main`'s edge-only stripping is
closer to what we need than `alignment.py`'s all-`P*` stripping, since it keeps `don't`.
But it would also turn a partial `fo-` into `fo`, and wav2vec emits partials without a
hyphen anyway. Plan: take the DP from Kaldi, take the index-preserving wrapper from
`extractions`' `alignment.py`, and do the comparison by mapping both sides into the wav2vec
alphabet in our normaliser (A–Z, `'`, word breaks) before the DP, not inside it. The three
other versions are kept for reference in `vendor/sync_asr/variants/`.

**`sync_asr/alignment.py`** (2026-07 to 09, Jim's own) is the actual "adaptation".
- `align_smith_waterman(ref, hyp, ignore_case=, ignore_punctuation=, heuristics=)` wraps
  the Kaldi DP and returns `AlignedSequence` of `AlignedGroup(a_indices, b_indices, kind)`.
  **Indices into the original ref/hyp lists** survive, which is what we need to carry
  `char_start/char_end` and word times through.
- Case/punctuation-insensitive comparison is here (`_comparison_text`: casefold, then drop
  every Unicode `P*` char). Punctuation-only tokens are skipped in the DP.
  - Caveat: dropping all `P*` also removes intra-word `'` and `-`, so `fo-` becomes `fo` and
    `don't` becomes `dont`. Our normaliser must do this itself, keeping apostrophes and the
    partial-word hyphen.
- Heuristics layer (`RepeatedTokenHeuristic`, `MergedTokenHeuristic`, `WordPairDictionary…`)
  adds alternative groups without deleting the originals. Useful precedent for
  repetition/partial labelling.
- Also contains a difflib `align_text`. It imports `AlignedGroup` from
  `riksdag/time_aligner.py`, which drags in bs4 via `riksdag/__init__`.

Other relevant pieces:
- `elements.py`: `TimedWord`/`TimedWordSentence`, with times as **int ms**.
- `ctm.py` truncates `int(float*1000)` (590/100k values off by 1 ms).
- `riksdag/time_aligner.py`: two-pass time-overlap alignment of two timed word
  sequences. A good fit for the wav2vec ↔ WhisperX cross-check.
- `whisperx_json_input.py`, `hf_json_input.py` (HF pipeline `chunks`), `vtt_input.py`,
  `gentle_json_input.py`: small readers. The last three have constructor bugs.
- `ctm_edit.py` fixers (`mark_correct_ignore_punct`, `possible_false_start`,
  `merge_consecutive`…). Several, with `make_equal=True`, **overwrite hyp text with ref
  text**, which is forbidden here.
- `kaldi/segment_ctm_edits{,_mild}.py`: Kaldi's cleanup segmenters. They are byte-identical
  or nearly so to upstream, and untested.

Quality:
- 103 tests pass (py3.11, deps pytest/bs4/requests). The DP is covered by 1 Kaldi vector
  test plus 18 tests in `test_alignment.py`. `get_ctm_edits`, the CLI and the segmenters
  are untested.
- Apache-2.0, headers kept.
- Hard-coded `/Users/joregan/...` paths in riksdag scripts; many untracked stray files
  at the root.

No hint anywhere of the LibriVox wav2vec model: model references are Swedish or phoneme
models, plus `jonatasgrosman/wav2vec2-large-xlsr-53-english` in a test comment.

## 2. `librivox-matching/` (no VCS, no tests, no LICENSE)

The previous attempt at this task, written from its own CLAUDE.md spec:
1. VibeVoice chunks → n-gram match against the whole etext.
2. CTC segmentation of the etext words per chunk with `jonatasgrosman/wav2vec2-large-xlsr-53-english`.
3. A Flask review app for mismatches.

- `ngram_match.py`: a numpy port of the kblabb n-gram convolution matcher (1..6-grams,
  weighted convolution, threshold, join gaps). Clean and self-contained.
- `chunk_matcher.py`: matches each chunk against the whole book independently, with no
  monotonicity or continuity between chunks, and keeps the longest region.
  - **Index bug:** word indices from the *normalised* text (hyphen→space, numbers→words
    change the token count) are used to slice the *raw* `etext.split()`, so passages drift.
  - Fuzzy fallback is `rapidfuzz.partial_ratio_alignment` over the whole book string.
- `ctc_align.py`: **cannot have run.** It calls `ctc_segmentation(lpz, words, config)`, but
  the 1.7.5 signature is `ctc_segmentation(config, lpz, ground_truth)` with a label matrix
  from `prepare_text`. It then indexes `timings[i]` per word, when timings are per frame
  and words need `determine_utterance_segments`. It also re-decodes the whole mp3 with pydub
  for every chunk.
- `vibevoice.py`: `merge_chunks` merges time-contiguous chunks **across speaker changes**,
  which is wrong for Task 2. It also doesn't handle the speaker-less `[Silence]`/`[Music]`
  segments (`item["Speaker"]` raises KeyError on real files).
- `normalize.py`: lowercases, strips all punctuation (apostrophes too) and uses num2words
  cardinals only.
- **Policy conflict:** where etext and VibeVoice diverge, its spec falls back to the
  VibeVoice text, which the "reference is right" rule forbids. It also treats VibeVoice as
  the primary hypothesis, but Task 1 must not depend on VibeVoice.
- `review_app/`: Flask + JS diff viewer with a global SequenceMatcher of VibeVoice vs etext.
  Its idea (per-chunk diff ops, save annotations) is sound, but it is bound to the old
  output format.

## 3. `librivox-multispeaker/` (code absent)

Only remnants of `spoken-sentence-transformers` (multi-axis speech-to-speech
similarity: WavLM encoders, a contrastive sampler):
- egg-info listing sources that **aren't present**
- `sanity_check.py` (nearest-neighbour retrieval per axis over a HF dataset)
- `tests/__pycache__` only
- `docs/questions.md` (empty)
- `process-tts-data.ipynb`, which builds `utterance-map.json`: same-sentence groups for
  VCTK (38k utts → 8.1k groups), ARCTIC awb and GBI, from `fdupes` on transcripts
- a saved LibriVox wiki **Accents Table** (self-reported reader → location/accent,
  about 116 rows)

Jim (2026-11-05): exploratory work, mostly looking for accents. That explains the saved
Accents Table. Nothing in it does alignment or attribution.

## 4. `librivox-mult-work-files/` (Jim's working repo; data, plus one notebook)

Mostly data (see `docs/inputs.md`). The only code is
`text/the-crocodile/parse_text.ipynb` (2026-02-26 to 03-03, branch `text`), a Stage 1
prototype for *The Crocodile* (pg40745):
- **Manual cut.** The Gutenberg HTML was split by hand into `text_01..04`, one per LibriVox
  section. Numbering then restarts per file, i.e. per LibriVox **section**. CLAUDE.md wants
  chapter/paragraph from the **text**.
- **Paragraph = every top-level element with text.** `h2` title, `h3` subtitle, the
  epigraph `div.blockquot`, chapter numerals ("I", "III") and `<p>` all count as
  paragraphs. `span.pagenum` is removed, and `get_text(separator='')` avoids spaces around
  inline tags. This is Jim's working answer to the open "paragraph-counting conventions"
  question: everything counts, as its own paragraph.
- **Sentences:** `mosestokenizer.MosesSentenceSplitter('en')`, then `rejoin_semicolons`
  (rejoin a split ending in `;`). Output TSV `paragraph, sentence, text`.
- **No character offsets, no quote spans, whitespace collapsed.** The output can't point
  back into the source, so it doesn't meet the "never lose the source text" rule as it stands.

## Decisions: integrate / reuse / discard

| Item | Decision | Reason |
|---|---|---|
| `sync-asr` `smith_waterman_alignment` (Kaldi) | **Reuse, adapt** → `vendor/sync_asr/align_ctm_ref.py` | Proven DP and Jim's chosen aligner. Make it global over the reference (or emit flanking ref as `del`), vectorise or band it, and keep times as float seconds |
| `sync-asr` `alignment.py` (`align_smith_waterman`, `AlignedGroup`, heuristics) | **Reuse, adapt** → `vendor/sync_asr/alignment.py` | Index-preserving output carries ref char spans and hyp times. Replace its blanket `P*` stripping with our normaliser |
| `sync-asr` `elements.py` | **Reuse (types only)** → vendor | Needed by `alignment.py`. Move to float seconds when adapted |
| `sync-asr` `riksdag/time_aligner.py` | **Reuse** → vendor | Two-pass time-overlap aligner fits the wav2vec ↔ WhisperX agreement vote |
| `sync-asr` `whisperx_json_input.py` | **Reuse as reference** → vendor | Matches the `/data/whisperx` format. A 30-line reader we'll fold into our ASR loader |
| `sync-asr` `get_ctm_edits` / ctm_edits format | **Discard** | Loses the reference index, so it can't be the record. Our word alignment JSON replaces it |
| `sync-asr` `ctm_edit.py` fixers | **Discard** | `make_equal` relabelling rewrites spoken tokens to match the text, against the dysfluency rule |
| `sync-asr` `segment_ctm_edits{,_mild}.py` | **Discard** (reference only) | Kaldi utterance splitting. Our segmentation is Stage 7's |
| `sync-asr` `ctm.py`, `hf_json_input.py`, `vtt_input.py`, `gentle_json_input.py` | **Discard** | ms truncation or constructor bugs, and formats we don't have. Rewrite readers once the wav2vec format is known |
| `sync-asr` riksdag/, specific/, lexicons, notebooks, root clutter | **Discard** | Swedish/Kashubian/Hungarian specifics, irrelevant to English LibriVox |
| `librivox-matching` `ngram_match.py` | **Reuse** → `vendor/librivox_matching/ngram_match.py` | Clean numpy n-gram anchor scorer for Stage 3 section → text mapping. Needs a monotonic wrapper |
| `librivox-matching` `chunk_matcher.py`, `fuzzy_match.py` | **Discard** | Per-chunk independent matching with a normalised/raw index bug. Stage 3 will match monotonically with char offsets |
| `librivox-matching` `ctc_align.py` | **Discard** | Broken API use, never ran. Kaldi-style alignment on the wav2vec hyp is the specified method |
| `librivox-matching` `vibevoice.py` | **Discard** (rewrite trivially) | Merges across speakers and crashes on speaker-less non-speech segments |
| `librivox-matching` `normalize.py` | **Discard** | Strips apostrophes and partial-word hyphens, cardinal-only numbers. Superseded by NeMo TN in Stage 4 |
| `librivox-matching` `review_app/` | **Discard code, keep idea** | Bound to the old format. Our review tables are TSV/HTML under `/out/review/`; revisit if Jim wants an interactive tool |
| `librivox-multispeaker` code remnants, `utterance-map.json` | **Discard** | No source present; VCTK/ARCTIC data, not LibriVox |
| `librivox-multispeaker` Accents Table | **Keep as reference data** → `data/ref/librivox-wiki-accents-table.html` | Optional self-reported reader accent/location metadata for readers in the corpus |
| work-files `respiro/` | **Integrate (as input data)** | Full-coverage breath intervals, read in place by Stage 0 (`asr.respiro`). Jim calls it a dead end, so it is used only to confirm sentence-end boundaries it agrees with |
| work-files `whisperx/`, `vibevoice/`, `index*/` | **Discard (duplicates)** | Same as `/data`, except 2 VibeVoice files only on `main`. Ask Jim to add those to `/data` rather than read git blobs |
| work-files `parse_text.ipynb` | **Reuse the conventions, rewrite the code** | Its paragraph rule (every top-level text element counts) and Moses + semicolon-rejoin splitter are the Stage 1 baseline. Needs char offsets, quote spans, text-based (not section-based) numbering, and no manual cutting |
| work-files `pg40745-images.html`, `output_0N.tsv` | **Keep as Stage 1 test fixture/oracle** | The only Gutenberg HTML on hand. The TSVs give expected sentence splits to regress against |
