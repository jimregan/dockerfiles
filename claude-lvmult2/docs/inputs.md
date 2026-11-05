# Inputs (`/data`), surveyed 2026-11-05

Produced by Stage 0 (`uv run lvm-survey [--work W] [--reading SLUG] [--force]`),
which writes `/out/inventory.jsonl` (one row per reading), `/out/stage0/summary.json`
and `/out/stage0/ffprobe.jsonl` (probe cache). Numbers below are from that run.

## Totals

| | |
|---|---|
| work directories | 44 |
| readings (inventory rows) | 251 |
| audio files | 5,442 mp3 (88 GB) |
| audio | 1,581.3 h |
| WhisperX outputs | 5,326 / 5,442 files |
| VibeVoice outputs | 4,349 / 5,442 files |
| Respiro breath outputs | 5,442 / 5,442 files (from `/src-projects/librivox-mult-work-files`) |
| **wav2vec2 outputs** | **0**: `/data/wav2vec2/` exists but is empty (also on the worker) |
| Gutenberg HTML | **none in `/data`**; only links in `index.html` |
| sampling rate | 44.1 kHz 5,410; 22.05 kHz 20; 32 kHz 6; 48 kHz 5; 16 kHz 1. Ten readings mix rates across sections |
| channels | mono 4,841; stereo 601 |
| form (inventory) | prose 173, verse 45, unknown 32 (collections), none 1 |
| dramatic readings | 20, all multi-reader (none single-reader; see below) |

## Layout

```
/data/
  getter.sh, run-vibevoice.sh      # download + VibeVoice scripts that built this tree
  <work>/                          # 44 hand-made groupings (slug), NOT LibriVox ids
    <librivox-slug>/               # one LibriVox project = usually one reading
      index.html                   # saved LibriVox project page (214 of 215 dirs)
      files                        # URL list used by getter.sh (184 dirs; absent for collections)
      <stem>.mp3                   # section audio (full-quality mp3, not the 64 kb copies)
      <stem>.json                  # VibeVoice output, same stem (missing for some)
    list, indir                    # (aesop/, if/) LibriVox URL lists used for downloading
  whisperx/<work>/<librivox-slug>/<stem>.{json,srt,tsv,txt,vtt}
  wav2vec2/                        # empty
```

- **Readings:** an inventory row is one LibriVox project directory, except for
  **multi-version** projects (genre "Multi-version (Weekly and Fortnightly poetry)", or
  every section has the same title and a different reader). Those are split into one
  reading per section, 38 rows in all (`if/if-by-rudyard-kipling*`, `weekly-poetry/halloween…`).
- **Collections** (32 rows; the section table has Author/Source columns): only the
  relevant section's mp3 was downloaded. The same project can appear under several works,
  e.g. First Chapter Collection 002 (`lv9296`) under oliver-twist, the-secret-garden
  and treasure-island, so collection `reading_id`s include the section (`lv9296-s05`).
  Collection genres describe the whole collection (a "Poetry" collection holds the prose
  *Little Match Girl*), so `form` is `unknown` for them and Stage 1 decides from markup.
- **Alternate versions inside one project:** Monte Cristo `lv47` has 8 sections titled
  "…, alternate version" (`*alt_dumas.mp3`), a second reader's take on a chapter.
  They are flagged `alternate: true` per section and are a within-project multi-reading.
- **Work grouping is by text, loosely:** several dirs are overlapping Andersen collections
  (`andersens-fairy-tales`, `fairy-tales-by-hans-christian-andersen`,
  `hans-christian-andersen-fairy-tale-collection`, `hans-christian-andersen-fairytales-and-short-stories`).
  `a-christmas-carol/christmas_poetry_and_hymn_collection` is the *poem* "A Christmas Carol"
  from *Pickwick Papers*, a different text from the novella.

## inventory.jsonl schema (per reading)

`work, reading_id, reading_dir, librivox_slug, librivox_id, archive_id, title, authors,
credits_line (has "Translated by …"), language, catalog_date, readers[{name,id}], n_readers,
coordinators, gutenberg_ids, online_text[{text,url}], section_source_urls, source_kind
(gutenberg|other|none), collection, multiversion, verse, form (prose|verse|unknown), dramatic,
play, solo, evidence{genres,cast_entries,slug_says_dramatic,section_reader_text}, cast[{roles,
readers,raw,group?}], n_sections_listed, n_sections_local, duration, asr_coverage, issues,
sections[]`.

Per section: `file, path, stem, section, title, alternate, readers, listed_time, num_samples,
duration (= num_samples / sampling_rate), duration_header, sampling_rate, channels, bit_rate,
section_{author,source,source_urls,gutenberg_ids}` (collections),
`asr.{wav2vec2,whisperx,vibevoice}`.

- `reading_id` is `lv<LibriVox project id>`, `-s<section>` for multi-version and collection
  rows, and `dir:<slug>` when there is no index.html. **This is a survey key, not the frozen
  `{reader}` ID of `docs/ids.md`**, which is still to be decided.
- `readers`: section reader column, plus every cast-list reader for dramatic readings (the
  section reader cell there is literally "Group"), falling back to "Read by". `librivox:<id>`
  where a reader page is linked. Some cast entries link a personal website or give plain
  text, so `id` is null; Stage 2 resolves those by name.
- `duration` is from **decoding** (ffmpeg → mono u8, count bytes = `num_samples`). mp3 header
  durations were off by more than 1 s for 110 files and more than 5 s for 11 (worst 14.9 s,
  `copperfield_51_dickens.mp3`). Manifests must use the decoded count, and the same decoder.

## ASR outputs

**WhisperX** (`/data/whisperx/…/<stem>.json`): `{segments:[{start,end,text,words:[{word,
start,end,score}]}], word_segments, language}`, all `language: en`, no speaker labels,
14.9 M words, all timed. Cased with punctuation, numbers as digits ("1872", "No. 7", "Chapter 1."), and names mangled ("Philius Fogg", "Puspart II"); it is a text-normalising system, as CLAUDE.md says.
Missing for 116 files: all of `celebration-of-dialects-and-accents` (55) and all
`shakespeares-sonnets` readings (61). Two files end more than 30 s before the audio does
(`theaesopforchildren_02`, `davidcopperfield_21`).

**VibeVoice** (`/data/<work>/<slug>/<stem>.json`): a JSON list of
`{Start, End, Speaker, Content}`, utterance-level (p10 3 s, median 16 s, p90 38 s, max 41 s), no word times.
Cased text with punctuation and digits. Non-speech segments have **no `Speaker` key** and
bracketed content: `[Noise]` 2,072, `[Music]` 1,102, `[Silence]` 817,
`[Environmental Sounds]` 524, `[Speech]` 484 (speech it did not transcribe),
`[Human Sounds]` 412, `[Unintelligible Speech]` 32.
- Coverage: 1,093 files missing. 19 readings have none, 87 are partial. The longest file
  with output is 46.0 min. Of the missing files, 147 are 45 min or longer, but 599 are
  30–45 min and 347 are under 30 min (shortest 50 s). So "too long for a 3090" is not
  the whole story; possibly the run did not finish. **Ask.**
- Two outputs are empty (end = 0): `olivertwist5_10_dickens_128kb`, `olivertwistver2_14_dickens`.
- Speakers: solo readings give 1 speaker in 1,832 of 1,840 files. Dramatic readings give a
  median of 3–7 speakers per file against 16–48 cast readers, consistent with
  "few speakers" in CLAUDE.md.
- Timing is coarse: the first WhisperX word starts a median 0.7 s after the first VibeVoice
  segment (p90 1.7 s), over 200 sampled files.

**wav2vec2:** none. The worker's HF cache has `facebook/wav2vec2-large-960h-lv60-self`
(the SETUP.md default, which uses an upper-case letter vocabulary with `'` and `|`), but
nothing records that this model was the one used, and there are no outputs. **Ask.**

## `/src-projects/librivox-mult-work-files` (Jim's working repo, git)

The working tree is on branch `respiro` (HEAD `a126390b`, 2026-04-15). Read-only, so the
other branches were read with `git show` / `ls-tree`.

| Path / branch | Contents | Relation to `/data` |
|---|---|---|
| `respiro/<work>/<slug>/<stem>.jsonl` | **Breath detection**: one line `{audio_file, breaths: [[start, end], …]}` (s) | **New.** Covers all 5,442 mp3s. Now in the inventory as `asr.respiro` |
| `whisperx/` (all branches) | 5,326 WhisperX JSON | Same content as `/data/whisperx`; the repo copy is pretty-printed, and 300/300 sampled files are equal when parsed |
| `vibevoice/` (tree: 4,336; `main` and `vv-missing`: 4,344) | VibeVoice JSON | 4,342 identical to `/data`. **2 only on `main`**: `the-crocodile/…/crocodile_01_dostoyevsky_128kb.json` and `wuthering-heights/…-2/wutheringheights_21_bronte.json`. A stash on `vv-missing` edits 6 Christmas Carol files |
| `index/` (tree) | 201 index.html | Whitespace-stripped copies of `/data` pages, parsing identically. The 13 missing are celebration-of-dialects, northanger-abbey and shakespeares-sonnets |
| `index-html/` (`index-pages` branch) | 214 index.html | Byte-identical to `/data`. **No page for `the-secret-garden-by-frances-hodgson-burnett-2`** here either |
| `text/the-crocodile/` (tree; `text` branch history) | `pg40745-images.html` (Gutenberg HTML, the *Short Stories* anthology that includes *The Crocodile*), hand-cut `text_01..04` (one HTML chunk per LibriVox section), `parse_text.ipynb`, `output_0N.tsv` | **Stage 1 prototype for one text**, see `docs/existing-code.md` §4. The only Gutenberg HTML anywhere |
| `index-html/` (tree) | empty dir | |

Respiro notes:
- Median 126 breaths per file.
- 1,335 files have a first interval starting at 0.0 s (median end 0.70 s). This may be
  leading silence or room tone detected as a breath, so check it before using it as a
  pause cue.
- 73 files have **no breaths**. They cluster by reading (`lv14329` has 18 files; others
  are `lv356`, `lv6263`, `lv8149`, `lv2994` and the aesop volumes).
- **Status (Jim, 2026-11-05): a dead end.** Respiro is used only as a confirming vote
  when a breath agrees with a sentence-end boundary (see `docs/decisions.md`). It is not
  used for pause measurement or segmentation, and no further investigation is planned.

Still absent: wav2vec2 output, Gutenberg HTML (except pg40745), and the index page for Secret Garden reading 2.

## Source texts

- Gutenberg IDs come from the "Online text" link (`gutenberg.org/etext/N` or `ebooks/N`),
  or for collections from the section's Source column. 228 readings have one; 13 link
  elsewhere (Google Books, ICDL, archive.org, hca.gilead.org.il for the six Andersen
  "Fairytales and Short Stories" volumes, adelaide.edu.au for `oliver-twist-by-charles-dickens-2`,
  kipling.org.uk, wikisource, bartleby…); 10 have none.
- Works whose readings use **different Gutenberg texts** (editions or translations,
  handled at linking): a-christmas-carol (46/19337/24022), aesop (7 different), alice
  (11/19033/28885/35688), from-the-earth-to-the-moon (83/44278), if (15529/556/32772),
  in-desert-and-wilderness (30365/60177), journey-to-the-centre (3748/18857),
  little-women (514/37106), northanger-abbey (121/13028 anthology), oliver-twist (730/5343),
  puck-of-pooks-hill (15976/26027), shakespeares-sonnets (1041/1600), the-secret-garden
  (113/17396), wuthering-heights (768/13623 anthology).
- **No Gutenberg HTML is present.** `www.gutenberg.org` is on the allowlist, but its
  comment says "metadata lookups only". Fetching the 59 distinct Gutenberg books (to `/out/gutenberg/`?)
  needs Jim's go-ahead.

## Dramatic readings

20 readings: cast list in the description (headers "Cast List:", "Cast listing:",
"Cast lists:", or none at all, as in Oliver Twist v5), "Dramatic Readings" genre, or a
`dramatic-reading` slug. All are multi-reader. One is a collection with 10 sub-casts
(`lv16498`), and one (`alice-in-wonderland-by-alice-gerstenberg`) is an actual **play**
(by genre and cast; Gerstenberg's stage adaptation), so its speakers are in the text.
`a-christmas-carol-condensed…` ("Condensed by the Author for his Dramatic Readings")
is Dickens's public-reading text read solo, so it is *not* dramatic.
**Single-reader dramatic readings are not identifiable from metadata**; LibriVox
doesn't tag "one reader doing voices". Finding them needs listening or an audio heuristic.

## Data issues

- `the-secret-garden/the-secret-garden-by-frances-hodgson-burnett-2`: **no index.html**.
  27 mp3s with VibeVoice and WhisperX output. Its audio differs from the other four Secret
  Garden readings, so it is a fifth reading with unknown LibriVox id, reader and source.
- Sections listed on LibriVox but not downloaded: Monte Cristo `lv6263` 14 of 120,
  aesop `lv9041` 3 of 20, and one each in `lv193`, `lv18216`, `lv4838`, `lv1135`, `lv9632`.
- `aesop/aesops-fables-new-translation-by-aesop/…_04_aesop_128kb.mp3.1`: a duplicate wget
  download, ignored.
- 16 mp3s decode with "Header missing / Invalid data" (corrupt frames; sample counts may
  differ between decoders): 8 aesop vol 3/4 files, Monte Cristo `032alt`/`043alt`/`045alt`,
  `copperfield_51`, `fairytales_02_andersen`, `andersen_21_thebeetle…`, `huckleberry_finn_04`/`_15`
  (see `/out/logs/stage0.log`). 157 more only have bad ID3 comment frames, which is harmless.
- 40 files differ from the LibriVox-listed time by more than 5 s. Most come from the
  header-estimate problem above, and the listed times are unreliable too.

## Per work

| work | readings | hours | Gutenberg IDs (readings) | form | dramatic (multi-reader) | collection / multi-version | files: wx / vv / total |
|---|---:|---:|---|---|---|---|---|
| a-christmas-carol | 18 | 45.8 | 46 (12), 19337 (3), — (2), 24022 (1) | prose 15, unknown 3 | 1 (1) | 3 / 0 | 79 / 39 / 79 |
| aesop | 19 | 32.6 | 11339 (13), 60004 (1), 25433 (1), — (1), 21 (1), 19994 (1), 18732 (1) | prose 17, verse 2 | 0 (0) | 0 / 0 | 569 / 568 / 569 |
| alices-adventures-in-wonderland | 12 | 28.1 | 11 (8), 19033 (2), 28885 (1), 35688 (1) | prose 12 | 2 (2) | 0 / 0 | 131 / 129 / 131 |
| andersens-fairy-tales | 2 | 11.7 | 1597 (2) | prose 2 | 0 (0) | 0 / 0 | 37 / 31 / 37 |
| anna-karenina-garnett | 8 | 38.5 | 1399 (8) | prose 8 | 0 (0) | 0 / 0 | 239 / 239 / 239 |
| anne-of-avonlea | 3 | 24.6 | 47 (3) | prose 3 | 1 (1) | 0 / 0 | 90 / 90 / 90 |
| anne-of-green-gables | 8 | 80.3 | 45 (8) | prose 8 | 1 (1) | 0 / 0 | 305 / 296 / 305 |
| anne-of-the-island | 4 | 29.6 | 51 (4) | prose 4 | 1 (1) | 0 / 0 | 164 / 164 / 164 |
| around-the-world-in-80-days | 5 | 36.4 | 103 (5) | prose 5 | 1 (1) | 0 / 0 | 174 / 174 / 174 |
| celebration-of-dialects-and-accents | 2 | 1.0 | — (1), 21 (1) | prose 2 | 0 (0) | 0 / 0 | 0 / 0 / 55 |
| crime-and-punishment | 4 | 86.7 | 2554 (4) | prose 4 | 1 (1) | 0 / 0 | 174 / 87 / 174 |
| david-copperfield | 5 | 137.9 | 766 (4), — (1) | prose 5 | 0 (0) | 0 / 0 | 266 / 122 / 266 |
| dracula | 5 | 82.5 | 345 (5) | prose 5 | 1 (1) | 0 / 0 | 135 / 11 / 135 |
| fairy-tales-by-hans-christian-andersen | 1 | 5.2 | — (1) | prose 1 | 0 (0) | 0 / 0 | 49 / 49 / 49 |
| father-goriot | 2 | 23.2 | 1237 (2) | prose 2 | 0 (0) | 0 / 0 | 31 / 0 / 31 |
| from-the-earth-to-the-moon | 2 | 9.7 | 44278 (1), 83 (1) | prose 2 | 0 (0) | 0 / 0 | 55 / 55 / 55 |
| hans-christian-andersen-fairy-tale-collection | 1 | 4.7 | 27200 (1) | unknown 1 | 0 (0) | 1 / 0 | 25 / 24 / 25 |
| hans-christian-andersen-fairytales-and-short-stories | 6 | 44.2 | — (6) | prose 6 | 0 (0) | 0 / 0 | 156 / 128 / 156 |
| if | 31 | 2.2 | 15529 (20), — (5), 556 (5), 32772 (1) | verse 18, unknown 13 | 0 (0) | 13 / 17 | 58 / 58 / 58 |
| in-desert-and-wilderness | 2 | 29.0 | 60177 (1), 30365 (1) | prose 2 | 0 (0) | 0 / 0 | 90 / 81 / 90 |
| journey-to-the-centre-of-the-earth | 2 | 18.6 | 3748 (1), 18857 (1) | prose 2 | 0 (0) | 0 / 0 | 88 / 88 / 88 |
| just-so-stories | 6 | 20.6 | 2781 (6) | prose 6 | 1 (1) | 0 / 0 | 73 / 70 / 73 |
| little-women | 5 | 74.3 | 514 (4), 37106 (1) | prose 4, unknown 1 | 2 (2) | 1 / 0 | 189 / 155 / 189 |
| no-second-troy | 2 | 0.0 | — (2) | unknown 2 | 0 (0) | 2 / 0 | 2 / 2 / 2 |
| northanger-abbey | 6 | 41.1 | 121 (5), 13028 (1) | prose 5, unknown 1 | 1 (1) | 1 / 0 | 158 / 0 / 158 |
| notes-from-the-underground | 2 | 10.2 | 600 (2) | prose 2 | 0 (0) | 0 / 0 | 26 / 18 / 26 |
| oliver-twist | 9 | 114.8 | 730 (7), 5343 (1), — (1) | prose 7, unknown 2 | 1 (1) | 2 / 0 | 374 / 360 / 374 |
| puck-of-pooks-hill | 2 | 13.1 | 26027 (1), 15976 (1) | prose 2 | 0 (0) | 0 / 0 | 20 / 0 / 20 |
| robinson-crusoe-in-words-of-one-syllable | 2 | 5.5 | 6936 (2) | prose 2 | 0 (0) | 0 / 0 | 24 / 24 / 24 |
| shakespeares-sonnets | 5 | 12.4 | 1041 (5) | verse 4, unknown 1 | 0 (0) | 1 / 0 | 0 / 0 / 61 |
| the-adventures-of-huckleberry-finn | 9 | 78.4 | 76 (9) | prose 7, unknown 2 | 1 (1) | 2 / 0 | 302 / 297 / 302 |
| the-brothers-karamazov | 4 | 124.2 | 28054 (4) | prose 3, unknown 1 | 0 (0) | 1 / 0 | 290 / 205 / 290 |
| the-call-of-cthulhu | 3 | 3.2 | 68283 (3) | prose 2, unknown 1 | 0 (0) | 1 / 0 | 7 / 3 / 7 |
| the-count-of-monte-cristo | 2 | 98.0 | 1184 (2) | prose 2 | 0 (0) | 0 / 0 | 234 / 165 / 234 |
| the-crocodile | 2 | 3.6 | 40745 (2) | prose 2 | 0 (0) | 0 / 0 | 8 / 5 / 8 |
| the-jungle-book | 3 | 16.2 | 236 (3) | prose 3 | 0 (0) | 0 / 0 | 43 / 32 / 43 |
| the-last-of-the-mohicans | 2 | 32.8 | 940 (2) | prose 2 | 0 (0) | 0 / 0 | 68 / 32 / 68 |
| the-little-match-girl | 2 | 1.5 | — (1), 1597 (1) | unknown 1, prose 1 | 0 (0) | 1 / 0 | 8 / 8 / 8 |
| the-prince-and-the-pauper | 4 | 30.2 | 1837 (4) | prose 4 | 1 (1) | 0 / 0 | 113 / 108 / 113 |
| the-secret-garden | 5 | 31.2 | 113 (3), — (1), 17396 (1) | prose 3, ? 1, unknown 1 | 1 (1) | 1 / 0 | 110 / 104 / 110 |
| tom-sawyer | 3 | 22.3 | 74 (3) | prose 3 | 1 (1) | 0 / 0 | 91 / 86 / 91 |
| treasure-island | 6 | 35.7 | 120 (6) | prose 5, unknown 1 | 1 (1) | 1 / 0 | 147 / 140 / 147 |
| weekly-poetry | 21 | 0.8 | 20644 (21) | verse 21 | 0 (0) | 0 / 21 | 21 / 21 / 21 |
| wuthering-heights | 4 | 38.7 | 768 (3), — (1) | prose 3, unknown 1 | 1 (1) | 1 / 0 | 103 / 81 / 103 |
