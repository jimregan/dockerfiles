# Task 2: speaker attribution (draft spec, 2026-11-09; labelling plan decided 2026-11-10)

## What the data looks like (survey 2026-11-09)

- **20 dramatic readings, 176 h**: 19 prose and one play (lv7904, Alice). The Stage 0
  `dramatic` flag requires a cast list, so **no single-reader dramatic readings** are
  detected; any that exist need another way to find them.
- **Cast lists:** 578 of 591 cast readers resolve to a LibriVox reader id by name. Roles can
  be combined ("Anne/Narrator", "Narrator/Jo"), there can be two narrators (lv8889, lv10177),
  and first-person books (Huck Finn, Treasure Island) have no separate narrator.
- **Sections:** 527 of 595 are "Group" on the LibriVox page (no reader); a few list 2–11
  readers. These are the 95 k S supervisions with `speaker: null`.
- **Text side:** Stage 1 `quotes[]` (start, end, mark, nesting level, closed, block):
  700–4,300 per text.
- **Chapter 0 enrollment:** the cast list is read as "X read by Y" (lv5009, lv10387,
  lv7545, lv6943). Whether each reader reads their own line is not yet known: VibeVoice
  finds 41 speakers in Oliver Twist ch. 0 but 4–5 in the others, and it caps the number of
  speakers.
- **Tools:** ECAPA (`speechbrain/spkrec-ecapa-voxceleb`) and speechbrain are on the worker.
  The local LLM (`$LLM_BASE_URL`) is down right now.

## Proposed pipeline

1. **T2.1 cast:** a label set per reading: roles, plus Narrator → reader ids. Combined roles
   share one reader.
2. **T2.2 text turns:** split each sentence's tokens into narration and quote spans (Stage 1
   quotes, with nesting). Each quote gets a character:
   - by rule where the text says so ("said Marilla" next to the quote, or a play's speaker
     prefix);
   - otherwise by the local LLM, constrained to the cast list, with a paragraph of context
     on each side.
   - Output: character, method, confidence.
3. **T2.3 audio turns:** turn edges come from Stage 5 word times. Compute an ECAPA
   embedding for each turn longer than about 1 s, and compare the embeddings across each
   switch point: a cosine change from splicing should be large.
4. **T2.4 enrollment:** reader voiceprints from ch. 0 cast lines if each reader reads their
   own line (check by clustering). Otherwise from narration of known readers and long
   quotes whose rule-based attribution is confident, clustered across the book.
5. **T2.5 intersection:** keep turns where the text label's reader matches the
   audio's nearest voiceprint. Others are marked `disagree`, not dropped.

## Output (proposed)

`custom.turns` on S/L supervisions: `[{character, reader, tok_start, tok_end, start, end,
text_method, audio_score, agree}]`. `tok_*` index `alignment.word`. `speaker` is the reader
when one reader speaks the whole segment; otherwise null, with the turns. No new
segmentation is needed, so the committed S/L stay as they are.

## Pilot before a full run

Anne of Green Gables (lv5009) and Oliver Twist (lv10387), three chapters each:
- Do the ch. 0 readers read their own lines?
- Do the audio switch points separate cleanly?
- How accurate are rule and LLM attribution, against a small hand-checked set or a
  Claude-labelled one (to be costed and approved first)?

## Claude-labelled reference set (proposal, awaiting approval)

- **Model and mode:** `claude-opus-5-5` through the Batch API (50% off: $2 in / $10 out per
  MTok), with the instructions and cast list cached.
- **One request per chapter:** chapter text with the quotes numbered, plus the cast list.
  The answer is JSON giving each quote its character, with evidence.
- **Pilot size:** 6 chapters, 332 quotes. About 30 k input and 40 k output tokens including
  thinking, so **about $0.50**. Three repeat runs to measure label stability: about $1.50.
- **Optional larger reference:** 2 chapters from each of the other 18 dramatic readings
  (about 2,500 quotes): about $5.
- **Full run (all dramatic texts, 2026-11-10):** 19 Gutenberg texts (pg514 has two
  readings), 9.4 M chars, 34,212 top-level quotes, about 650 chapters. About 2.6 M input
  tokens ($5) and 4 M output at about 120 tokens per quote including thinking ($41): **about
  $45** (range $25–75 depending on thinking length). Three repeats: about $135.
- **Decided 2026-11-10:** one Claude pass plus two local-model passes, instead of three Claude repeats.
- **Reuse:** labels belong to the text, so they carry over to every reading of the same
  Gutenberg text (pg45 has several Anne of Green Gables readings).
