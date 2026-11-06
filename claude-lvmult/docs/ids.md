# IDs

**Status: draft.** CLAUDE.md requires these rules to be frozen before any manifest is
released. Rules marked **confirmed** have Jim's sign-off; the rest are proposals.

## Text side (`text_id`, shared across readings of the same text)

| Component | Rule | Status |
|---|---|---|
| `{text}` | `pg<N>`, the Gutenberg ebook number of the HTML in `/out/gutenberg/` (sha256 in its manifest) | confirmed (CLAUDE.md) |
| `{chapter:03d}` | **division ordinal** from Stage 1: a run of consecutive headings opens a division; divisions are numbered from 1 in document order, front matter included; 0 = text before the first heading (only if any). Not the book's own chapter number; the heading text is kept in `divisions[].heading` for display | **confirmed 2026-11-06**; zero-padded to 3 digits (pg60004 has 382 divisions) |
| `{para:04d}` | block number, 1-based within the division. Every block counts (headings, epigraphs, captions, contents entries, stanzas), with `kind` for filtering | **confirmed 2026-11-06** |
| `{sent:02d}` | sentence number, 1-based within the paragraph (prose) | proposed |
| `{poem}` | Stage 1 poem ordinal within the text (`poems[].poem`) | proposed; zero-padding width to decide |
| `{stanza:02d}`, `{line:02d}` | 1-based within poem / within stanza | proposed |
| verse sentence `s{n:02d}` | 1-based within the poem (`sentences[].n` with `unit: "poem"`) | proposed |

Formats (from CLAUDE.md, with `{chapter:03d}`): prose sentence `{text}/{chapter:03d}/{para:04d}/{sent:02d}`;
verse line `{text}/{poem}/{stanza:02d}/{line:02d}`; verse sentence `{text}/{poem}/s{n:02d}`.

Stability: IDs are a deterministic function of the source HTML (sha256), the Stage 1
`VERSION` and `data/stage1_overrides.json`. Changing any of these can renumber, so they
are pinned in each Stage 1 output's `meta` and must be frozen for a release.

Open:
- Prose and verse in one division (poems inside a prose chapter): the stanzas have both a
  `{para}` (prose hierarchy) and `{poem}/{stanza}` IDs. Which does the `S` segmentation use?

## Reading side (supervision `id`)

`{work}/{reader}/…` per CLAUDE.md. `{reader}` is still to be decided; the Stage 0
`reading_id` (`lv<LibriVox id>[-s<section>]`) is a survey key, not yet this.
