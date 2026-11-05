from __future__ import annotations

import argparse
import os

import ollama

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
DEFAULT_GEN_MODEL = os.environ.get("GEN_MODEL", "gemma4:26b")
DEFAULT_CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
DEFAULT_COLLECTION = "stl-qpsr"
DEFAULT_TOP_K = 6

SYSTEM_PROMPT = (
    "You are an expert on the history and research of the Dept. for Speech, Music "
    "and Hearing, drawing on 40 years of its STL-QPSR quarterly progress reports. "
    "Answer only using the numbered excerpts provided in the context below. "
    "Cite claims with their excerpt number in brackets, e.g. [2]. "
    "If the excerpts don't contain enough information to answer, say so plainly "
    "instead of guessing."
)


def format_citation(metadata: dict) -> str:
    title = metadata.get("title") or "(untitled)"
    author = metadata.get("author") or "unknown author"
    year = metadata.get("year", "?")
    volume = metadata.get("volume", "?")
    number = metadata.get("number", "?")
    pages = metadata.get("pages", "?")
    return f"{author} — \"{title}\", STL-QPSR {volume}({number}), {year}, pp. {pages}"


def retrieve(collection, ollama_client, embed_model: str, question: str, top_k: int):
    query_embedding = ollama_client.embeddings(model=embed_model, prompt=question)["embedding"]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    return list(zip(documents, metadatas))


def build_context(chunks: list[tuple[str, dict]]) -> str:
    parts = []
    for i, (document, metadata) in enumerate(chunks, start=1):
        parts.append(f"[{i}] {format_citation(metadata)}\n{document}")
    return "\n\n---\n\n".join(parts)


def ask(collection, ollama_client, embed_model: str, gen_model: str, question: str, top_k: int) -> None:
    chunks = retrieve(collection, ollama_client, embed_model, question, top_k)
    if not chunks:
        print("No indexed content found.")
        return

    context = build_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}"},
    ]

    for part in ollama_client.chat(model=gen_model, messages=messages, stream=True):
        print(part["message"]["content"], end="", flush=True)
    print()

    print("\nSources:")
    for i, (_document, metadata) in enumerate(chunks, start=1):
        print(f"  [{i}] {format_citation(metadata)} ({metadata.get('article_dir')})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions over the indexed STL-QPSR corpus.")
    parser.add_argument("question", nargs="?", help="Question to ask. Omit to start an interactive session.")
    parser.add_argument("--persist-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--gen-model", default=DEFAULT_GEN_MODEL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import chromadb

    chroma_client = chromadb.PersistentClient(path=args.persist_dir)
    collection = chroma_client.get_collection(args.collection)
    ollama_client = ollama.Client(host=args.ollama_host)

    if args.question:
        ask(collection, ollama_client, args.embed_model, args.gen_model, args.question, args.top_k)
        return

    print("Interactive mode. Ctrl-D to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except EOFError:
            break
        if not question:
            continue
        ask(collection, ollama_client, args.embed_model, args.gen_model, question, args.top_k)


if __name__ == "__main__":
    main()
