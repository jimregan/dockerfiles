# IDs

**Status: frozen 2026-11-09** (Jim). Changing any rule here renumbers released manifests.

## Text side (`text_id`, shared across readings of the same text)

| Component | Rule | Status |
|---|---|---|
| `{text}` | `pg<N>`, the Gutenberg ebook number of the HTML in `/out/gutenberg/` (sha256 in its manifest) | confirmed (CLAUDE.md) |
| `{chapter:03d}` | **division ordinal** from Stage 1: a run of consecutive headings opens a division; divisions are numbered from 1 in document order, front matter included; 0 = text before the first heading (only if any). Not the book's own chapter number; the heading text is kept in `divisions[].heading` for display | **confirmed 2026-11-06**; zero-padded to 3 digits (pg60004 has 382 divisions) |
| `{para:04d}` | block number, 1-based within the division. Every block counts (headings, epigraphs, captions, contents entries, stanzas), with `kind` for filtering | **confirmed 2026-11-06** |
| `{sent:02d}` | sentence number, 1-based within the paragraph (prose) | confirmed |
| `{poem:03d}` | Stage 1 poem ordinal within the text (`poems[].poem`) | **confirmed 2026-11-09**, 3 digits |
| `{stanza:02d}`, `{line:02d}` | 1-based within poem / within stanza | confirmed |
| verse sentence `s{n:02d}` | 1-based within the poem (`sentences[].n` with `unit: "poem"`) | confirmed |

Formats (from CLAUDE.md, with `{chapter:03d}`): prose sentence `{text}/{chapter:03d}/{para:04d}/{sent:02d}`;
verse line `{text}/{poem:03d}/{stanza:02d}/{line:02d}`; verse sentence `{text}/{poem:03d}/s{n:02d}`.
Padding is a minimum width: a number that needs more digits is written in full.

Stability: IDs are a deterministic function of the source HTML (sha256), the Stage 1
`VERSION` and `data/stage1_overrides.json`. Changing any of these can renumber, so they
are pinned in each Stage 1 output's `meta` and must be frozen for a release.

Verse inside prose (poems in a prose chapter): the stanzas have both a `{para}` and
`{poem}/{stanza}` position. Sentences of verse use the **poem** hierarchy
(`{text}/{poem}/s{n:02d}`), lines `{text}/{poem}/{stanza:02d}/{line:02d}` (Jim, 2026-11-06).

## Reading side (supervision `id`)

`{work}/{reader}/{tail}`, confirmed 2026-11-09.

- `{work}`: the Stage 0 work slug (`dracula`, `around-the-world-in-80-days`).
- `{reader}`: the Stage 0 `reading_id`, `lv<LibriVox project id>`, with `-s<NN>` (the LibriVox
  section) when one project holds several readings of the work (collections, multi-version
  projects). Unique per reading. It names the recording project, not a person: group readings
  have one `reading_id` and many readers. The person is the supervision's `speaker`,
  `librivox:<LibriVox reader id>` (e.g. Corrinne LePage = `librivox:11959`), from that section's
  reader in the LibriVox page.
- `{tail}`:

| Unit | tail |
|---|---|
| prose sentence | `S/{chapter:03d}/{para:04d}/{sent:02d}` |
| verse sentence | `S/{poem:03d}/s{n:02d}` |
| verse line | `L/{poem:03d}/{stanza:02d}/{line:02d}` |
| header / footer | `{section}/H{nn}`, `{section}/F{nn}` (Stage 2 element id; `{section}` = LibriVox section number) |
| unreferenced speech | `{section}/X{nn}`, numbered in time order within the section |

- **Multi-text readings** (a reading whose sections use more than one Gutenberg text): body
  tails carry the text, `S/{text}/…`, `L/{text}/…` (confirmed 2026-11-09).
- **Repeats:** a unit read again gets `.2`, `.3`… on its id, unique across the reading in
  section order: within a section after another unit has started (reordered reading), or in
  a later section (alternate versions, overlapping sections, re-read fables).
- **X segments** are supervisions with `text: ""`, the spoken words in `custom.verbatim`
  and `alignment`, and `custom.unreferenced: true` (confirmed 2026-11-09).
- `text_id` is the `{text}/…` body tail without the `S/`/`L/` prefix; null for H/F/X.
- `recording_id`: `{work}_{reader}_{section}` (one per section audio file); see `docs/manifests.md`.
