from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from tqdm import tqdm

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
DEFAULT_CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
DEFAULT_COLLECTION = "stl-qpsr"
DEFAULT_CHUNK_TOKENS = 800
DEFAULT_CHUNK_OVERLAP = 150

_encoding = tiktoken.get_encoding("cl100k_base")
_METADATA_KEY_RE = re.compile(r"^(journal|volume|number|year|pages):\s*", re.IGNORECASE)


@dataclass
class Article:
    rel_dir: str
    year: str
    volume: str
    number: str
    pages: str
    title: str | None
    author: str | None
    text: str


def find_articles(output_root: Path):
    for manifest_path in sorted(output_root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            continue
        yield manifest_path.parent, manifest


def parse_path_metadata(rel_dir: str) -> tuple[str, str, str, str]:
    parts = rel_dir.split("/")
    year, volume, number = parts[0], parts[1], parts[2]
    pages = parts[-1].rsplit("_", 1)[-1]
    return year, volume, number, pages


def parse_title_author(page1_text: str) -> tuple[str | None, str | None]:
    blocks = [b.strip() for b in page1_text.split("\n\n") if b.strip()]
    header_blocks = []
    for block in blocks:
        if _METADATA_KEY_RE.match(block):
            break
        header_blocks.append(block)

    title_idx = None
    for i, block in enumerate(header_blocks):
        if block.startswith("# "):
            title_idx = i
    if title_idx is None:
        return None, None

    title = header_blocks[title_idx][2:].replace("\n", " ").strip() or None

    author = None
    if title_idx + 1 < len(header_blocks):
        candidate = header_blocks[title_idx + 1]
        if not candidate.startswith("#"):
            author = candidate.replace("\n", " ").strip() or None

    return title, author


def build_article(article_dir: Path, output_root: Path) -> Article | None:
    page_files = sorted(article_dir.glob("page-*.md"))
    if not page_files:
        return None

    page1_text = page_files[0].read_text(encoding="utf-8").strip()
    body_text = "\n\n".join(
        text
        for f in page_files[1:]
        if (text := f.read_text(encoding="utf-8").strip())
    )
    if not body_text:
        return None

    rel_dir = str(article_dir.relative_to(output_root))
    year, volume, number, pages = parse_path_metadata(rel_dir)
    title, author = parse_title_author(page1_text)

    return Article(
        rel_dir=rel_dir,
        year=year,
        volume=volume,
        number=number,
        pages=pages,
        title=title,
        author=author,
        text=body_text,
    )


def _tok_len(text: str) -> int:
    return len(_encoding.encode(text))


def chunk_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush():
        if current:
            chunks.append("\n\n".join(current))

    for para in paragraphs:
        para_len = _tok_len(para)

        if para_len > max_tokens:
            flush()
            current.clear()
            current_len = 0
            tokens = _encoding.encode(para)
            step = max_tokens - overlap_tokens
            for start in range(0, len(tokens), step):
                chunks.append(_encoding.decode(tokens[start:start + max_tokens]))
            continue

        if current_len + para_len > max_tokens and current:
            flush()
            overlap: list[str] = []
            overlap_len = 0
            for p in reversed(current):
                p_len = _tok_len(p)
                if overlap_len + p_len > overlap_tokens:
                    break
                overlap.insert(0, p)
                overlap_len += p_len
            current = overlap
            current_len = overlap_len

        current.append(para)
        current_len += para_len

    flush()
    return chunks


def ingest(
    output_root: Path,
    persist_dir: str,
    collection_name: str,
    embed_model: str,
    ollama_host: str,
    chunk_tokens: int,
    chunk_overlap: int,
    dry_run: bool,
) -> None:
    collection = None
    ollama_client = None
    existing_ids: set[str] = set()

    if not dry_run:
        import chromadb
        import ollama

        chroma_client = chromadb.PersistentClient(path=persist_dir)
        collection = chroma_client.get_or_create_collection(collection_name)
        existing_ids = set(collection.get(include=[])["ids"])
        ollama_client = ollama.Client(host=ollama_host)

    articles = list(find_articles(output_root))
    n_articles = n_embedded = n_skipped = 0

    for article_dir, _manifest in tqdm(articles, desc="Articles"):
        article = build_article(article_dir, output_root)
        if article is None:
            continue
        n_articles += 1

        chunks = chunk_text(article.text, chunk_tokens, chunk_overlap)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{article.rel_dir}::chunk{i:03d}"
            if chunk_id in existing_ids:
                n_skipped += 1
                continue

            if dry_run:
                n_embedded += 1
                continue

            embedding = ollama_client.embeddings(model=embed_model, prompt=chunk)["embedding"]
            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "year": article.year,
                    "volume": article.volume,
                    "number": article.number,
                    "pages": article.pages,
                    "title": article.title or "",
                    "author": article.author or "",
                    "article_dir": article.rel_dir,
                    "chunk_index": i,
                }],
            )
            n_embedded += 1

    print(f"Articles with content: {n_articles}")
    print(f"Chunks {'that would be ' if dry_run else ''}embedded: {n_embedded}")
    print(f"Chunks already indexed (skipped): {n_skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk and embed STL-QPSR OCR markdown output into a persistent Chroma collection."
    )
    parser.add_argument("output_root", type=Path, help="Root of the gemma4_ocr.py output tree.")
    parser.add_argument("--persist-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk without calling Ollama or writing to Chroma.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest(
        output_root=args.output_root.resolve(),
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embed_model=args.embed_model,
        ollama_host=args.ollama_host,
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
