# Decisions (dated, one line each)

- 2026-11-05: A *reading* is one LibriVox project dir; multi-version projects are split one reading per section; collections key on project+section.
- 2026-11-05: Inventory `reading_id` (`lv<id>[-s<sec>]`) is a survey key only; supervision `{reader}` IDs are deferred to `docs/ids.md`.
- 2026-11-05: Audio durations come from a full decode (num_samples), never the mp3 header; header estimates were off by up to 14.9 s.
- 2026-11-05: Collection readings get `form: unknown`; Stage 1 decides verse/prose from the text markup, not the collection's genre.
- 2026-11-05: Dramatic = cast list, "Dramatic Readings" genre, or `dramatic-reading` slug; not free-text "dramatic" (Dickens's "Condensed … for his Dramatic Readings" is solo).
- 2026-11-05: Kept code is vendored verbatim under `vendor/` with provenance; adaptation happens in `src/` at the consuming stage (see `docs/existing-code.md`).
- 2026-11-05: Kaldi SW aligner is reused but must be made global over the reference (flanking ref words currently vanish instead of becoming `del`).
- 2026-11-05: Respiro breath outputs are read in place from `/src-projects/librivox-mult-work-files/respiro` (`--respiro` to override) and recorded as `asr.respiro`.
- 2026-11-05: Respiro is a dead end as a pause/segmentation source (Jim). It is used only as confirmation: a breath interval that agrees with a sentence-end candidate boundary raises confidence in that boundary; it never creates, moves, or vetoes a boundary.
- 2026-11-05: Duplicated WhisperX/VibeVoice/index data in the work-files repo is ignored; `/data` stays the source of truth.
