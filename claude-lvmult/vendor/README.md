# Vendored code (verbatim, not yet wired into the package)

Copied unmodified on 2026-11-05 from read-only `/src-projects`. Adapt copies in
`src/librivox_mult/` when the consuming stage is built; leave these files as the
reference for what was taken. See `docs/existing-code.md` for the reasons.

| File | Source | Licence | Intended use |
|---|---|---|---|
| `sync_asr/align_ctm_ref.py` | `sync-asr@e774b51:sync_asr/kaldi/align_ctm_ref.py` (Kaldi `steps/cleanup/internal`, V. Manohar 2016 / D. Gao 2020) | Apache-2.0 (header kept) | Stage 5 DP core (`smith_waterman_alignment`), to be made global/banded |
| `sync_asr/alignment.py` | `sync-asr@e774b51:sync_asr/alignment.py` | Apache-2.0 | Stage 5 index-preserving wrapper (`AlignedGroup`), heuristics |
| `sync_asr/elements.py` | `sync-asr@e774b51:sync_asr/elements.py` | Apache-2.0 | timed-word types used by the above (ms ints; we will use float seconds) |
| `sync_asr/time_aligner.py` | `sync-asr@e774b51:sync_asr/riksdag/time_aligner.py` | Apache-2.0 | Stage 5 wav2vec ↔ WhisperX time-overlap cross-check |
| `sync_asr/whisperx_json_input.py` | `sync-asr@e774b51:sync_asr/whisperx_json_input.py` | Apache-2.0 | reference for WhisperX reader |
| `sync_asr/LICENSE` | `sync-asr` | Apache-2.0 | |
| `librivox_matching/ngram_match.py` | `librivox-matching:librivox_matching/ngram_match.py` (no VCS, no LICENSE file) | **unstated, Jim's code** | Stage 3 coarse section → text anchoring |

`data/ref/librivox-wiki-accents-table.html` is a saved LibriVox wiki page
(reader → location/accent, self-reported, ~116 rows) from `librivox-multispeaker`.

`sync_asr/variants/`: the other versions of `align_ctm_ref.py` in sync-asr, for
reference only: `main` (073ec86f, edge-punctuation `--ignore-punctuation`),
`ignore-punctuation-in-aligner` (2b5340e8), and upstream-Kaldi-like (f61fa139).
See `docs/existing-code.md`.
