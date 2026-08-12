# qpsr-rag

RAG over the OCR'd STL-QPSR (Dept. for Speech, Music and Hearing) quarterly
reports produced by `gemma4_ocr.py`. Runs against an Ollama instance already
serving in Docker — set `OLLAMA_HOST` if it isn't on `localhost:11434`.

## Setup (on the server, where Ollama's Docker container is)

Start the Ollama container first if it isn't already running — `entrypoint.sh`
just polls `OLLAMA_HOST` until it answers, so it'll hang forever against a
stopped container rather than erroring out.

```bash
# once Ollama is up, against its published port:
ollama pull nomic-embed-text   # embedding model
ollama pull gemma4:26b         # generation model (already pulled for OCR)

docker build -t qpsr-rag .
```

Since Ollama publishes 11434 to the host, run qpsr-rag with `--network host`
so `localhost:11434` (the Dockerfile's default `OLLAMA_HOST`) reaches it
directly — no need to look up container names or IPs.

## Ingest

```bash
docker run --rm \
  --network host \
  -v /path/to/output:/markdown:ro \
  -v qpsr-chroma-data:/data \
  qpsr-rag
```

Or without Docker, directly:

```bash
pip install -r requirements.txt
python ingest.py /path/to/output --persist-dir ./chroma_db
```

- Groups OCR pages back into articles using each article's `manifest.json`
  (only `status: completed` articles are indexed; `error.json` is ignored —
  it's stale from before OCR was re-run).
- Extracts year/volume/number/pages from the directory path, and
  title/author from the page-1 front matter.
- Chunks each article's body text (~800 tokens, ~150 token overlap) and
  embeds each chunk with `--embed-model` (default `nomic-embed-text`).
- Safe to re-run: existing chunk IDs are skipped, so it can be run
  incrementally as more of the corpus is synced.

Add `--dry-run` to validate parsing/chunking without a live Ollama
connection or writing to Chroma — useful for a first pass over a partial
sync.

## Query

```bash
docker run --rm -it \
  --network host \
  -v qpsr-chroma-data:/data \
  qpsr-rag python3 query.py --persist-dir /data/chroma_db
```

Or without Docker: `python query.py "What speech synthesis work came out of
the department in the 1970s?"`

Omit the question to start an interactive session. Each answer is grounded
in the top-k retrieved excerpts (`--top-k`, default 6) and followed by a
`Sources:` list with the article citation for each.

## Config

All of `--persist-dir`, `--collection`, `--ollama-host`, `--embed-model`,
`--gen-model` can also be set via the env vars `CHROMA_DIR`,
`OLLAMA_HOST`, `EMBED_MODEL`, `GEN_MODEL`.
