# Stage 8: manifests

`uv run python -m librivox_mult.stage8_manifests [--work W] [--reading ID] [--force]`
(`src/librivox_mult/stage8_manifests.py`). lhotse's JSONL schema, written without lhotse;
`tests/test_stage8.py` loads and validates the output with lhotse (a dev dependency only).

## Files

Per reading, `/out/stage8/<work>/<reading_id>/`; all readings concatenated (inventory
order) in `/out/manifests/`:

| File | Contents |
|---|---|
| `recordings.jsonl.gz` | one Recording per section audio file with segments |
| `supervisions_S.jsonl.gz` | sentence segmentation: one SupervisionSegment per spoken segment |
| `supervisions_L.jsonl.gz` | line segmentation (readings with verse only) |
| `unspoken_S.jsonl.gz`, `unspoken_L.jsonl.gz` | reference units with no speech, same fields, `duration` 0 |
| `validation.json` (per reading) | check results |

Unspoken units (unread contents, skipped sentences) are not supervisions: they have no
audio extent, and lhotse rejects zero-length supervisions. They are kept so that the
reference is covered exactly once by `supervisions + unspoken`.

## Recording

`id` = `{work}_{reading_id}_{section}` (`dracula_lv12966_04`). `sources[0].source` is the
`/data` path; `sampling_rate`, `num_samples` and `channels` come from the Stage 0 probe;
`duration` = `num_samples / sampling_rate`.

## Supervision

IDs follow `ids.md`. `start`, `duration` and every `alignment.word` item are on the
recording's timeline. `channel` is 0, or `[0, 1]` for stereo files.

- `text`: the reference with its original punctuation (Stage 1 char span, widened over
  adjacent punctuation in the same sentence/line). Header/footer: the template text.
  X segments: `""`.
- `speaker`: `librivox:<reader id>` of the section's reader; the reading's sole reader if the
  section lists none; null if neither. Null mostly means a "Group" section: the
  LibriVox page names no reader because several people read parts of the chapter
  (Dracula's diaries, group Crime and Punishment); attributing those is Task 2 work. Sections with two readers use the first and list both
  in `custom.readers`.
- `alignment.word`: what was spoken (wav2vec2 words, partials marked `-`).
- `custom`: `segmentation`, `unit` (S/L/H/F/X), `text_id`, `gutenberg`, `section`,
  `verbatim`, `src_span`, `boundary {start, end}`, `pause_before`, `pause_after`,
  `dysfluencies [{type, tokens, ref_tokens, whisperx}]` (`tokens` index `alignment.word`,
  `ref_tokens` are Stage 1 token ids), `nsw`, `confidence`, `header_footer`, `unreferenced`
  (X only), `readers` (several readers only), and
  `edits [[kind, ref_word, stage1_token, alignment_index]]`: the word alignment, one entry per
  pair (cor/sub/del/ins and the dysfluency kinds' base pairs). NSW expansions share a
  `stage1_token`; header words have no token; `del` has no alignment index; `ins` no ref word.

## Validation

Per reading and segmentation: required fields and types; unique ids; supervisions with
`duration > 0` inside the recording; alignment items inside their supervision; no overlaps
within a recording; every Stage 5 reference row covered exactly once.
