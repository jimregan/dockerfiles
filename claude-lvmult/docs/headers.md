# Stage 2: LibriVox metadata and header/footer templates

`uv run lvm-headers [--work W] [--reading SLUG] [--force] [--jobs N]`
(`src/librivox_mult/stage2_headers.py`, normaliser in `src/librivox_mult/spoken.py`)

- **Inputs:** `/out/inventory.jsonl` (Stage 0, which parses index.html: readers,
  sections, cast, sources) and the wav2vec2 transcripts in `/out/w2v/`.
- **Output:** `/out/stage2/<work>/<reading_id>.json` per reading, plus review tables in
  `/out/review/stage2/`.
- **Runtime:** about 20 s for the corpus.

## Templates (expectations, not truth)

For each section file, template **elements** are generated, each with several spoken
variants. The variants are normalised to the wav2vec2 alphabet: lower case, numbers as
words ("ninety three", "one o two", "number sixteen"), "Mr." → "mister", years as
"eighteen forty eight", ligatures folded.

| Element | Role | Variants |
|---|---|---|
| `intro` | header | "{section\|chapter\|part\|stave…} {n} of {title}[ by {author}][ translated by {translator}]". Also the numbers from the section title ("Part I, Chapter 05" gives "chapter five") |
| `disclaimer_long` | header | "This is a LibriVox recording. All LibriVox recordings are in the public domain. For more information or to volunteer, please visit librivox dot org." and the "…find out how to volunteer, please contact…" form |
| `disclaimer_short` | header | "This LibriVox recording is in the public domain." / "This is a LibriVox public domain recording." / "A public domain recording for librivox dot org" / "This recording (poem) is in the public domain." |
| `title_author` | header | "{title} by {author}"; title variants: full, before `,`/`:`/`(`, without "Volume N"; also the bare title |
| `credit` | header and footer | "recording/recorded/read/reading/this reading/narrated by {reader}", "read for librivox dot org by {reader}", "read by {reader} for librivox dot org" |
| `section_title` | header | the LibriVox section title, cleaned (`may_be_book_text`: chapter headings are often in the Gutenberg text too) |
| `end` | footer | "end of {section word} {n}[ of {title}]", "end of {section word}", "end of {section title}", "end of poem/story/fable/text", plus numbers from the section title |
| `footer_pd` | footer | "This recording is in the public domain." / short disclaimer |
| `end_of_book` | footer | last section only: "end of {title}[ by {author}]" |
| `cast` | body (whole file) | cast-list sections (title has cast/dramatis/credits/characters): "{role} read by/played by {reader}", and "{role} read by" alone because readers say their real name rather than the page's username |

- **Titles:** "(version 2)", "(Dramatic Reading)" etc. are removed.
- **Translator:** taken from "Translated by" on the page.
- **Section readers:** from the section table, falling back to "Read by".
- **Reader pools:** cast entries such as "Minor characters read by: a, b, c" are reader
  pools, flagged `pool: true` by the index parser, and get no elements.

## Matching

- **Regions:**
  - header: words starting before min(180 s, max(60 s, 35% of the file))
  - footer: words ending within min(120 s, max(45 s, 35%)) of the end
  - If the two overlap (short files), the words are split at the midpoint.
  - Cast sections search the whole file.
- **Fuzzy matching:** each variant is located with rapidfuzz `partial_ratio_alignment` on
  **space-free letter strings**, so ASR word splits ("libri vox", "libre ox") don't count
  against it. A word belongs to the match if most of its letters are inside the aligned
  span. Matches are accepted at score ≥80, or ≥90 for needles under 15 letters. The best
  variant per element wins.
- **Overlaps:** matches are ranked by score band (5-point bins), then length. A small
  overlap (≤2 words or ≤20%) trims the weaker match; a larger one drops it.
- **Recorded per match:** `kind`, `role`, the variant, `score`, the word span
  `w0`/`w1`, `start`/`end` times on the recording timeline, and the matched words (`hyp`).
  Cast matches also get `after_verb`: the words after "read by", up to the next
  "read/played by". That window contains the spoken name, plus the next character's name,
  which can't be split off reliably.
- **Recorded per section:**
  - `header_gaps` / `footer_gaps`: unmatched word runs between matches, i.e. ad-libs
    such as places, dates and dedications.
  - `header_after`: the 12 words after the last header match. These are ad-libs or
    already book text; Stage 3 finds where the book starts.

## Attribution classes (`disclaimer` in the review)

| Class | Meaning |
|---|---|
| `long` / `short` | disclaimer in the header |
| `credit+footer` | older format: "read for librivox dot org by X" … "this recording is in the public domain" at the end |
| `credit` | LibriVox named in the credit only |
| `footer` | public-domain sentence in the footer only |
| none | flagged `no_disclaimer` |

Header/footer **segment IDs** (`H{nn}`/`F{nn}`) are assigned in Stage 7 from the
segments actually committed, in spoken order. Stage 2 only says what was found where.
