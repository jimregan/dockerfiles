# LibriVox multi-reading corpus

Build a corpus of **same and similar spoken sentences** from LibriVox: works with
several readings, each aligned to its Project Gutenberg source text at the sentence
(or verse-line) level, with stable IDs that link the same text unit across readings.

Two tasks:

1. **Alignment.** Time-align every reading to its source text. Commit to a named
   segmentation, keep word timings alongside it, and include LibriVox headers and footers.
2. **Speaker attribution (dramatic readings).** Label which character, and which
   reader, speaks each part of each sentence, including switches within a sentence.
   This is only partly specced. See Task 2.

Jim (the user) is a speech-technology PhD candidate and runs the host cluster.
He prefers direct technical discussion and does not need things simplified.

---

## Environment

You are in the `claude` container. There is no GPU and no Docker socket, and outbound
network is limited to an allowlist (`/etc/claude-firewall/allowed-domains.txt`).
Do not edit `docker/claude/allowed-domains.txt` without asking Jim. If you need
a domain added, ask.

| Path | Mode | Contents |
|---|---|---|
| `/data` | ro | Audio, LibriVox `index.html` per reading, Gutenberg HTML, ASR outputs |
| `/work` | rw | This repo |
| `/out` | rw | Generated manifests, logs, review tables. Never commit these |
| `/src-projects` | ro | Earlier related code (see below) |
| `/cache/hf` | worker only | Shared Hugging Face cache |

**GPU and heavy jobs run on the worker** (CUDA, Kaldi in `/opt/kaldi`, Python env in
`/opt/venv`). The same paths are mounted there:

```bash
ssh worker 'cd /work && python -m pkg.stage --args'
# long jobs: detach and log
ssh worker 'cd /work && tmux new -d -s s3 "python -m pkg.align ... 2>&1 | tee /out/logs/s3.log"'
```

- The worker has **no internet**. New Python deps go in `docker/worker/requirements.txt`.
  Tell Jim to rebuild; don't work around it. Models must already be in `/cache/hf`;
  Jim downloads them with `docker compose run --rm fetch download <repo>`.
- A **local LLM** (Qwen/Gemma via vLLM) may be running at `$LLM_BASE_URL`
  (OpenAI-compatible, model name `local`). Check `curl $LLM_BASE_URL/models`.
  It is started on demand; if it's down, ask.
- Light CPU work (text parsing, alignment post-processing, tests) can run here with `uv`.
- **Do not call the Anthropic API from pipeline code** without Jim's go-ahead.
  If a Claude-labelled reference subset is wanted (Task 2), propose the size and
  cost first, using Batch API with prompt caching.

## Existing code in /src-projects

- `librivox-matching/`: two small tools written earlier for this task
- `librivox-multispeaker/`: related work (speaker side)
- Jim's adaptation of the **LibriSpeech Kaldi aligner** (location to be confirmed;
  look for it, or ask)

First job: read these, write `docs/existing-code.md` (what each does, quality,
what to keep). Then integrate, reuse or discard, each with a one-line reason.
Copy anything kept into this repo; never modify `/src-projects`.

---

## Hard rules

- **Never sanitise dysfluencies.** Fumbles, repetitions, partial words, false starts
  and self-corrections are data, not noise. The dysfluent native reading of
  *Around the World in 80 Days* is particularly valuable. Nothing in the pipeline
  may drop or "repair" spoken material to make it match the text.
- **Never lose the source text.** Every normalised token keeps a pointer to its
  character span in the original Gutenberg text, so punctuation, casing, quotes
  and paragraph structure can always be recovered.
- **The reference text is right for its reading.** Each reading's Gutenberg text is
  the one LibriVox lists as its source. Mismatches are what the reader did,
  not an edition problem. Different readings of a work may use *different translations*.
  That is handled at the linking step, not in alignment.
- **Stages are deterministic, resumable and idempotent.** Each stage reads files and
  writes files under `/out/<stage>/`, can be rerun for one book or reading, and skips
  completed work unless forced.
- **Manual-check checkpoints are real stops.** Where a stage says *review*, produce a
  review table (TSV or HTML under `/out/review/`), summarise it, and wait for Jim.
- Commit to git at the end of each working stage, with tests.

---

## Inputs

The exact layout of `/data` is not documented yet. **Stage 0 surveys it** and writes
`docs/inputs.md` plus `/out/inventory.jsonl` (one row per reading):

- work, reader(s), LibriVox ID
- sections, with audio files and durations
- the Gutenberg source and its ID
- which ASR outputs exist for each section
- verse or prose; dramatic or solo

Known so far:

- **ASR outputs:** wav2vec, WhisperX, VibeVoice. Some VibeVoice transcripts are missing
  (files too long for a 3090).
- The **wav2vec model** and its vocabulary are not yet recorded. Find out from the
  outputs: letters or phones, apostrophes, case. Ask if it's unclear.

### Role of each ASR system

| System | Use | Caveats |
|---|---|---|
| **wav2vec** | Hypothesis for the Kaldi aligner, and the **record of what was said** | CTC output keeps fumbles as letter strings. This is the most reliable system |
| **WhisperX** | Word-timing cross-check; second opinion that tends to clean up the reading | Whisper "repairs" fumbles. Its word times come from forcing *Whisper's* text onto the audio, so they are least trustworthy exactly where it smoothed over a fumble. Whisper matching the book where wav2vec doesn't is a **dysfluency signal** |
| **VibeVoice** | Chunk timings (coarse anchors, section to text mapping); speaker turns (Task 2) | Missing for some files. Task 1 must not depend on it |

---

## Task 1: alignment pipeline

### Stage 1: source text

Parse the Gutenberg HTML into a structured text JSON with character offsets:

- Strip the Gutenberg licence boilerplate. This is separate from LibriVox headers.
- **Prose:** chapter → paragraph (`<p>` order, 1-based within the chapter) → sentence.
  Document what counts as a paragraph (e.g. are captions, epigraphs, letters inside
  the text, and chapter titles counted, or treated as their own units?).
- **Verse:** poem → stanza → line, **and** poem → sentence. These two hierarchies cross,
  so both are stored over the same tokens. Gutenberg verse markup is inconsistent
  (`<br/>` inside `<p>`, `div.poem` with indent spans, `<pre>`). Write rules for each
  pattern and **review** poem parses.
- Sentence splitting must handle abbreviations, quotes and dialogue punctuation.
  Keep quote-mark spans; Task 2 needs them.
- Tokens: `{surface, char_start, char_end, sent, para|line, ...}`.

### Stage 2: LibriVox metadata and header/footer templates

Parse `index.html` to get readers, sections, the source-text link, the **cast list**
(character → reader) for dramatic readings, and dedications where present.

Generate the expected header and footer text for each section. Typical forms:

- the LibriVox disclaimer ("This is a LibriVox recording. All LibriVox recordings are
  in the public domain…")
- "Chapter N of *Title* by *Author*", "Read by *Name*"
- "End of chapter N", end of book
- the cast list in chapter 0 of dramatic readings

Readers ad-lib (dates, locations, dedications), so templates are expectations, not truth.
Header and footer segments are aligned like text and get their own IDs.
**Review** anything that matches its template poorly.

### Stage 3: section → text span mapping

LibriVox sections don't map one to one onto Gutenberg chapters (splits, merges,
prefaces). Coarsely align each section's wav2vec hypothesis (or VibeVoice, where it
exists) against the whole text to find the start and end paragraph or line.
Output a mapping table and **review** it.

### Stage 4: normalisation, with NeMo non-deterministic TN

- Normalise the reference to the wav2vec vocabulary: case, punctuation, apostrophes.
  Jim's aligner adaptation already ignores punctuation and casing.
- Non-standard words (numbers, dates, roman numerals, abbreviations like "Mr.", "&c.",
  currency): use `nemo_text_processing`'s **`NormalizerWithAudio`**, which produces
  several candidate spoken forms. Two passes:
  1. Align with a default spoken form for each non-standard word, or a wildcard,
     to find the local wav2vec hypothesis window.
  2. Score each candidate against that window (NeMo's CER-based selection),
     choose one, and realign.
- A chosen spoken form maps **many tokens to one source span** ("80" → "eighty").
  Record the candidates, the choice and its score.

### Stage 5: alignment (Kaldi LibriSpeech aligner)

- Reference = normalised text for the section's span plus header and footer templates.
  Hypothesis = word-timed wav2vec output.
- **Keep the word-level alignment** (match, substitution, insertion, deletion, with times).
  The aligner's own utterance splitting is not the segmentation; see Stage 7.
- Inserted and substituted runs are **kept**. Label them as dysfluency spans,
  by rule where possible:
  - **repetition:** an inserted token matching a neighbouring ref token
  - **partial:** an inserted token that is a prefix of the next ref word
  - **revision/false start:** inserted material followed by a restart
  - **skip:** a deleted ref token
  - otherwise **sub** or **ins**

  WhisperX agreeing with the reference where wav2vec doesn't raises confidence
  that a span is a real dysfluency.
- Per-word scores; per-sentence confidence from alignment score plus cross-system agreement.
- Missing VibeVoice transcripts = fewer votes, nothing more.
- Optional, later: rerun VibeVoice on long files cut at silences found by
  wav2vec/WhisperX, then stitch.

### Stage 6: pauses

For every candidate boundary (sentence end, line end, both), measure the actual pause
from wav2vec gaps and/or Silero VAD. This gives per-reading boundary behaviour
(pauses at line ends only, sentence ends only, both, or neither), which is data in itself.

### Stage 7: segmentation (committed)

The output **commits to a segmentation**. Each segmentation is a named, complete
division of a reading's speech (headers and footers are their own segments):

- `S`: sentences (prose; and verse, as the second pass)
- `L`: verse lines (**first pass for poetry**)

Boundary rules (set per segmentation, documented in `docs/segmentation.md`):

- With a pause: cut inside the silence, with padding limited by the gap
  (record `boundary: pause`).
- Without a pause: cut at the word boundary, record `boundary: forced`.
- Material inserted between units (false starts, repeats) attaches to the
  **following** segment.
- A later `S` pass over poetry may reuse the `L` pass's word alignment, but it
  records its own boundary decisions.

### Stage 8: manifests and validation

Write JSONL (gzip is fine) in **lhotse's schema, without using the lhotse library
in the pipeline**. lhotse is a test-only dependency: a test checks that the
manifests load.

`recordings.jsonl`: one line per LibriVox section audio file.

```json
{"id": "around80days_r042_ch07",
 "sources": [{"type": "file", "channels": [0], "source": "/data/.../ch07.mp3"}],
 "sampling_rate": 44100, "num_samples": 79380000, "duration": 1800.0, "channel_ids": [0]}
```

`supervisions.jsonl`: one line per committed segment.

```json
{"id": "around80days/r042/S/07/0012/03",
 "recording_id": "around80days_r042_ch07",
 "start": 412.38, "duration": 6.91, "channel": 0,
 "text": "Phileas Fogg had won his wager.",
 "language": "en", "speaker": "librivox:4211",
 "alignment": {"word": [
   {"symbol": "phileas", "start": 412.52, "duration": 0.41, "score": 0.93},
   {"symbol": "fo-",     "start": 412.95, "duration": 0.18, "score": 0.61},
   {"symbol": "fogg",    "start": 413.20, "duration": 0.33, "score": 0.95}]},
 "custom": {
   "segmentation": "S",
   "text_id": "pg103/07/0012/03",
   "verbatim": "phileas fo- fogg had won his wager",
   "src_span": [123456, 123487],
   "boundary": {"start": "pause", "end": "pause"},
   "pause_before": 0.42, "pause_after": 0.65,
   "dysfluencies": [{"type": "partial", "tokens": [1]}],
   "nsw": [],
   "confidence": 0.91,
   "header_footer": null}}
```

Field notes:

- `text` is the **reference** text with its original punctuation. `alignment.word`
  follows **what was spoken**. `dysfluencies[].tokens` index into `alignment.word`.
- All times are on the **recording's timeline**, including alignment items
  (lhotse convention).
- `speaker` is the **reader**. Characters (Task 2) go in `custom`.

Validation:

- schema check
- no overlaps within a segmentation for a recording
- segments fall within the recording's duration
- every ref token is covered exactly once per segmentation
- lhotse load test

### IDs

| Unit | `text_id` (shared across readings of the same text) | supervision `id` |
|---|---|---|
| Prose sentence | `{text}/{chapter}/{para:04d}/{sent:02d}` | `{work}/{reader}/S/{chapter}/{para:04d}/{sent:02d}` |
| Verse line | `{text}/{poem}/{stanza:02d}/{line:02d}` | `{work}/{reader}/L/{poem}/{stanza:02d}/{line:02d}` |
| Verse sentence | `{text}/{poem}/s{n:02d}` | `{work}/{reader}/S/{poem}/s{n:02d}` |
| Header/footer | none (`template` field instead) | `{work}/{reader}/{section}/H{nn}` / `F{nn}` |

- `{text}` is the Gutenberg text (e.g. `pg103`). `{work}` is a slug for the work.
- Chapter and paragraph come from the **text**, not from the LibriVox section;
  `recording_id` gives the section.
- Freeze the ID rules in `docs/ids.md` before any manifest is released.

### Linking readings (same / similar)

- **Same text:** join on `text_id`.
- **Different translations of one work:** align sentences once per pair of texts
  (monolingual, embedding-based, Bertalign/vecalign style, allowing 1–2 and 2–1 merges)
  and write `/out/links/text_links.jsonl` (`text_id_a`, `text_id_b`, `score`, `type`).
  Same-text pairs are implicit with score 1.
- This step never touches the audio.

---

## Task 2: speaker attribution (draft; spec with Jim before building)

Background ideas, not decisions:

- **Label set:** the `index.html` cast list (character → reader), plus narrator.
- **Plays vs. prose:** plays label speakers in the text, so no attribution is needed.
  Prose dramatic readings need quote attribution.
- **Text side:** quote spans from Stage 1, then LLM attribution of each span to a
  cast member, constrained to the cast list. Use the local LLM; a Claude-labelled
  subset may be used to measure it (ask first).
- **Audio side:** parts are usually recorded by separate volunteers and spliced
  together, so channel and room changes are strong cues. Check switch points with
  speaker embeddings (ECAPA) on either side of the word-aligned candidate boundary.
  VibeVoice diarisation is a weak vote only. It handles few speakers and does poorly
  on women's voices.
- **Enrollment:** chapter 0 cast introductions, if each reader introduces themselves.
  Check this; often the coordinator reads the whole list. Fallback: long,
  confidently attributed quotes, clustered across the book.
- **Intersection:** keep segments where the text label and the audio label agree.
- **Representation:** sentences with several characters need `custom.turns`
  (`character`, `reader`, `tok_start`, `tok_end`, `start`, `end`) or sub-segments.
  Not yet decided.
- **Single-reader dramatic readings** (one person voicing every part) are a distinct,
  valuable condition. The channel cue doesn't apply to them.

---

## Open questions (ask; don't guess)

- The wav2vec model and its vocabulary
- Where Jim's adapted LibriSpeech aligner lives and how it's invoked
- Paragraph-counting conventions for non-body text
- Task 2: `turns` vs. sub-segments; how the chapter 0 enrollment is actually read

## Conventions

- Python 3.11, package under `src/`, `uv`; `pytest` tests under `tests/`,
  with small fixtures copied from `/data` (one short section per case:
  prose, verse, dramatic, dysfluent).
- One CLI entry point per stage; `--work`/`--reading` filters; `--force`.
- Logs to `/out/logs/`. Review tables to `/out/review/`.
- Keep `docs/` current: `inputs.md`, `existing-code.md`, `ids.md`, `segmentation.md`,
  and `decisions.md` (dated, one line each).
