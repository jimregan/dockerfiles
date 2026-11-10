# Stage 1: source text structure

`uv run lvm-text [--id N ...] [--work W] [--reading SLUG] [--force]`
(`src/librivox_mult/stage1_text.py`)

- **Input:** `/out/gutenberg/pg<N>-images.html` (sha256 in `manifest.jsonl`).
- **Output:** `/out/stage1/pg<N>.json.gz`, plus review tables in `/out/review/stage1/`.
- **Rerunning:** a text is skipped when its output has the same source sha256, `VERSION`
  and overrides hash.

## What counts as text

- **Licence boilerplate:** everything outside `#pg-start-separator` …
  `#pg-end-separator` (or `#pg-footer`) is dropped. All 59 files have these ids.
- **Gutenberg editions note:** the "There are several editions of this ebook…" note
  (an `h4` plus a table, at the top of the body in 7 files) is stripped up to the next
  heading. The number of blocks stripped is in `stats.stripped_pg_note_blocks`.
- **Not text:** page numbers (`span.pagenum`/`pageno`) and images, including alt text.
  Everything else visible in the body is text.
- **Coverage check:** every visible text node in the body is either inside a block or
  attached to one by the safety net, and `stats.chars_total == chars_covered` is
  checked per text. All 59 texts are at 100%.

## Canonical text and offsets

- **`text`:** block texts joined by `\n\n`, verse/br lines by `\n`. Whitespace is collapsed
  within a line. Characters are otherwise exactly as in the HTML: curly quotes, dashes,
  `_italic_` underscores, `[1]` footnote markers.
- **Offsets:** every `char_start`/`char_end` (blocks, lines, sentences, tokens, quotes)
  indexes into `text`.
- **Back to the HTML:** each block also records `html` (an element path such as
  `/body[1]/div[3]/p[12]`) and `html_id`, so the HTML stays the ground truth.

## Blocks = paragraphs

- **Blocks:** the innermost text-bearing block elements: `p`, `h1`–`h6`, `pre`, `li`,
  `dd`/`dt`, `td`/`th`, captions, stanza-level verse divs, and divs whose text is
  entirely inline (title-page lines, letters, TEI output).
- **Paragraph numbering:** every block is a paragraph, numbered 1-based within its
  division. This is Jim's prototype convention, confirmed 2026-11-06: headings, epigraphs, captions, contents
  entries and verse stanzas all count. The `kind` field says what each block is, so
  later stages can filter without renumbering.

| kind | meaning |
|---|---|
| `heading` | `h1`–`h6`; TEI `div.tei-head` etc.; `p.ph1/ph2`; a short leaf `div.chapter`; or a short `p`/`div` that is entirely "CHAPTER IV", "Stave One", "ACT II", "SCENE ONE", "III." (never in tables, nor in contents runs) |
| `para` | ordinary prose paragraph (`flags: ["blockquote"]` when inside a blockquote: epigraphs, letters) |
| `verse` | one stanza (see below) |
| `lines` | line-broken (`<br>`) block that is not verse: letters, signatures, addresses, title pages, lists |
| `toc` | in a division headed Contents/Illustrations, or under a toc/contents class |
| `caption`, `footnote`, `separator` | figure captions; footnotes; asterisms and rules ("* * *") |

## Divisions (chapters)

- **What opens a division:** a run of consecutive headings, e.g. "CHAPTER I." plus "The
  Beginning", or "PART ONE" plus "Chapter 1". Text before the first heading is division
  0, which exists only if there is such text. Headed divisions are numbered from 1 in
  document order. `divisions[].heading` keeps the heading texts.
- **Division numbers are ordinals, not the book's chapter numbers.** Front matter
  (title, contents, preface) takes the first divisions, so "CHAPTER I" is often
  division 2. That is deliberate. The labels are inconsistent across Gutenberg files,
  ordinals are deterministic from the HTML, and the heading text is kept for display.
  Confirmed by Jim (2026-11-06) and recorded in `docs/ids.md`.
- **Known texts without chapter headings in the body:**
  - *Father Goriot* (pg1237) and *Robinson Crusoe in Words of One Syllable* (pg6936):
    3 divisions each.
  - *Prince and the Pauper* (pg1837): chapters are bare `<a id="cN">` anchors, but there
    are `<p>` "Chapter N." lines that the text rule catches (36 divisions).

## Verse

- **What counts as verse:**
  - **Markup** (confidence `markup`): `div.poem`, `div.stanza`, `.verse`, `.poetry`, TEI
    `div.tei-lg`, `<pre>`. Inside a stanza-level container, child elements are lines, not
    blocks.
  - **Heuristic** (confidence `heuristic`): a `<br>`-broken `p`/`td`/`li`/`div` with ≥2
    lines, a median line ≤60 chars and a max ≤90. It is rejected when the block is
    mostly capitals, has page references, is digit-heavy, opens like a letter ("Yours…",
    "Dear…") or contains chapter-heading lines; those become `lines`.
  - **Override** (confidence `override`): from `data/stage1_overrides.json`.
- **Stanzas:** a blank line (`<br><br>`, an empty line in `<pre>`) splits a stanza. Each
  stanza is a block with `lines[]` (`char_start`, `char_end`, `indent` = `iN` class,
  margin style or `pre:<spaces>`).
- **Poems:** consecutive verse blocks in one division. `poems[]` holds `poem` (ordinal in
  the text), `division`, `title` (the immediately preceding heading, if any), `blocks`,
  `confidence` (weakest of its stanzas) and `source`. Each verse block also gets `poem`
  and `stanza`, and lines are 1-based within the stanza.
- **The two hierarchies:** for verse, poem → stanza → line is in `blocks[].lines`, and
  poem → sentence is in `sentences[]` with `unit: "poem"`. They cover the same tokens.

## Sentences

- **Splitter:** the Moses sentence splitter (`mosestokenizer`, English), then rejoining
  any split that ends in `;`. This is Jim's prototype, and it reproduces his Crocodile
  splits exactly (897/897, `tests/test_stage1.py`).
- **Units:** prose is split per block; verse per poem, so a verse sentence can cross
  lines and stanzas.
- **Back to offsets:** splitter output is mapped back with a whitespace-insensitive
  match that must consume the whole unit. Any failure falls back to one sentence per
  unit, counted in `stats.split_failures` (0 for all 59).
- **Fields:** `sentences[]` holds `char_start`, `char_end`, `unit`, `block` (first block),
  `division`, `para`, and `n`, which is 1-based within the paragraph for prose or within
  the poem for verse.
- **Known behaviour:** Moses splits inside quoted dialogue at `!`/`?`
  ("How horrid that crocodile is! | I am really frightened," she said). Jim accepted
  these as sentence boundaries (2026-11-06).

## Tokens

- **Storage:** columnar arrays (`surface`, `start`, `end`, `punct`, `sent`, `block`,
  `line`). A token is either a word `[^\W_]+(?:['’-][^\W_]+)*` (so `don’t`, `half-past`)
  or a single punctuation character.
- **Coverage:** every non-space character is in exactly one token and exactly one
  sentence, which is tested.
- **Normalisation is Stage 4's job;** tokens are the unmodified source.

## Quotes

- **Detection:** `quotes[]` gives `start`, `end`, `mark` (`“ ‘ " ”`), `level` (nesting),
  `closed` and `block`, detected per block with a stack.
  - `’` counts as a closing quote only if a `‘` is open and the `’` is not word-internal
    (`don’t`). A leading elision like `’tis` can still be misread.
  - A quote left open at block end is `closed: false`. Usually the speech continues in
    the next paragraph: 83% of the 5,378 unclosed quotes are followed by a paragraph
    that opens a quote.
- **Weak on straight `"` quotes** (pg13028, pg20644, pg27200, pg37106, pg40745), where
  open and close can only be toggled. Task 2 will need better.

## Review (`/out/review/stage1/`)

- `poems.html`: every poem as parsed, heuristic ones first (orange), with the HTML path
  for overrides.
- `poems.tsv`, `lines_blocks.tsv` (line-broken non-verse blocks), `divisions.tsv`
  (division → heading → first text) and `summary.tsv` (per-text counts, coverage, kinds).
- **Overrides:** `data/stage1_overrides.json`, `{"pgN": {"<html path>": "verse" | "lines" |
  "para" | "heading" | "toc" | …}}`. The file is checked in and hashed into each output's
  meta, so a change reruns the affected texts.
