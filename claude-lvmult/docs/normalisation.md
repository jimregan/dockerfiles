# Stage 4: normalisation to the wav2vec2 alphabet

`src/librivox_mult/stage4_norm.py`, in two steps.

## 4a `text` (worker; needs NeMo)

`ssh worker 'cd /work && PYTHONPATH=src /opt/venv/bin/python -m librivox_mult.stage4_norm text'`

Per Stage 1 text, writes `/out/stage4/text/<text_id>.json.gz`. A text is recomputed only
if its Stage 1 sha256 or `TEXT_VERSION` changes. NeMo answers are memoised in
`/out/stage4/nsw_cache.json`.

- **Plain tokens** get their spoken words: lower case, `’`→`'`, diacritics and ligatures
  folded, hyphens split (`half-past` → `half past`), edge apostrophes stripped
  (`’tis` → `tis`, `don’t` kept). Punctuation tokens have no words.
- **NSW spans** cover numbers merged across joiners (`20,000`, `£20,000`, `8.45`,
  `12:30`, `1848-53`, `50%`), ordinals (`2nd`, `23d`), roman numerals of 2+ letters,
  `&`/`£`/`$`/`%`/`½`…, and abbreviations followed by `.` (Mr., Mrs., Dr., St., etc.,
  &c., …).
  - `No.`/`vol.`/`ch.`/`p.` and units count only before a number; "I said No." is a word.
  - `No. 7` is one span.
- **NeMo** (`/out/cache/nemo_tn`): `NormalizerWithAudio` (cased, n_tagged 30) gives
  candidates and `Normalizer` the default. Roman numerals are queried with the previous
  word as context ("Henry VIII" vs "Chapter XII"), and the context is stripped from the
  answers.
- **`words`:** `[word, stage1_token, nsw_id or -1]`. NSW words use the default form.
  Every word keeps its token, and so its char span: one source span can map to many
  words ("1872" → "eighteen seventy two").

## 4b `select` (CPU)

`uv run python -m librivox_mult.stage4_norm select`

Per reading, writes `/out/stage4/<work>/<reading_id>.json`. For each mapped section
(Stage 3 span, whole blocks):

- **Candidates:** NeMo's, plus rule-based forms it misses ("twenty thousand pounds",
  "eight forty five", "eighteen seventy two", "et cetera"/"and so forth", "the
  twelfth"…). Dropped: "candidates" that are just the unconverted input (`mr`,
  `xii`). An unconverted NeMo default is replaced by a real form.
- **Audio window:** the reference with default forms is aligned to the wav2vec2 book
  words (rapidfuzz word opcodes). The NSW's window is the audio words between its
  nearest exactly matched neighbours.
- **Score:** CER of the candidate against its best-matching contiguous run of window
  words, so misheard neighbours ("doctor libesey") don't count against it. The lowest
  wins; ties go to the default. With no window the default stands.
- **Output:** `ref` is the section's reference words with the chosen forms
  `[word, token, nsw]`. `nsw` records surface, kind, default, candidates, choice, score
  and window per NSW.

## Results (2026-11-06)

48,857 NSW occurrences in 5,112 sections. 94% have a choice with CER ≤0.2 against the
audio (abbreviations 99%). The low-confidence rest (`/out/review/stage4/nsw_lowconf.tsv`)
is mostly text that isn't read aloud: footnote markers `[1]` (323), footnote and list
numbers at the start of a block (208), and contents entries (108). Stage 5 will mark
those as unspoken.
