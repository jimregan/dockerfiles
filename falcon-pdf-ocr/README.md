# Falcon PDF OCR

Batch OCR for PDFs using Falcon OCR layout mode.

The container expects:

- `/data` mounted read-only with PDFs anywhere under it
- `/output` mounted writable for OCR results

Discovery is recursive, so both of these layouts are handled without special cases:

- `[Top level]/[Year]/[Quarter]/*.pdf`
- `1960/[extra-level]/[Quarter]/*.pdf`

## Output layout

For an input like:

```text
/data/1974/Q2/report.pdf
```

the output will be written to:

```text
/output/1974/Q2/report/
  manifest.json
  page-0001.json
  page-0001.md
  page-0002.json
  page-0002.md
  ...
```

Each page JSON contains the page size plus layout blocks with category, bounding box, score, and text. Markdown is assembled from those layout blocks in reading order, with light formatting for headings and list items.

## Build and run

```bash
cd falcon-pdf-ocr
docker build -t falcon-pdf-ocr .
docker run --rm \
  --gpus all \
  -v /absolute/path/to/pdfs:/data:ro \
  -v /absolute/path/to/output:/output \
  falcon-pdf-ocr
```

Or use the helper script:

```bash
./run.sh /absolute/path/to/pdfs /absolute/path/to/output
```

## Useful environment variables

```bash
docker run --rm \
  --gpus all \
  -e RENDER_DPI=240 \
  -e PAGE_BATCH_SIZE=4 \
  -e OCR_BATCH_SIZE=32 \
  -e SKIP_EXISTING=true \
  -e MAX_DOCS=10 \
  -e MAX_PAGES_PER_DOC=25 \
  -v /absolute/path/to/pdfs:/data:ro \
  -v /absolute/path/to/output:/output \
  falcon-pdf-ocr
```

## Notes

- This is designed for NVIDIA Docker hosts with GPU access.
- The first run will download the Falcon OCR weights from Hugging Face into the container cache.
- The container now installs `torch==2.9.0` and `torchvision==0.24.0` explicitly because Falcon OCR's current remote model code imports `AuxRequest` from `torch.nn.attention.flex_attention`, which is present in newer PyTorch releases but not in older `2.5.x` builds.
- Old scanned PDFs are precisely the cases where layout mode is most useful, but Falcon OCR's own model card still notes that degraded scans and tiny text remain challenging.
