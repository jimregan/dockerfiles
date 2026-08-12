"""
title: STL-QPSR Expert
author: qpsr-rag
version: 0.1
description: RAG over the Dept. for Speech, Music and Hearing's STL-QPSR quarterly reports.
requirements: chromadb, ollama
"""

from __future__ import annotations

import os
from typing import Generator, Iterator, List, Union

from pydantic import BaseModel

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


class Pipeline:
    class Valves(BaseModel):
        OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        EMBED_MODEL: str = os.environ.get("EMBED_MODEL", "nomic-embed-text")
        GEN_MODEL: str = os.environ.get("GEN_MODEL", "gemma4:26b")
        CHROMA_DIR: str = os.environ.get("CHROMA_DIR", "/data/chroma_db")
        COLLECTION: str = "stl-qpsr"
        TOP_K: int = 6

    def __init__(self):
        self.id = "qpsr-rag"
        self.name = "STL-QPSR Expert"
        self.valves = self.Valves()
        self.collection = None
        self.ollama_client = None

    async def on_startup(self):
        import chromadb
        import ollama

        chroma_client = chromadb.PersistentClient(path=self.valves.CHROMA_DIR)
        self.collection = chroma_client.get_collection(self.valves.COLLECTION)
        self.ollama_client = ollama.Client(host=self.valves.OLLAMA_HOST)

    async def on_shutdown(self):
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        query_embedding = self.ollama_client.embeddings(
            model=self.valves.EMBED_MODEL, prompt=user_message
        )["embedding"]
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=self.valves.TOP_K
        )
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        if not documents:
            return "No indexed content found for this question."

        context = "\n\n---\n\n".join(
            f"[{i}] {format_citation(metadata)}\n{document}"
            for i, (document, metadata) in enumerate(zip(documents, metadatas), start=1)
        )
        chat_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {user_message}"},
        ]
        stream = self.ollama_client.chat(
            model=self.valves.GEN_MODEL, messages=chat_messages, stream=True
        )

        def generate():
            for part in stream:
                yield part["message"]["content"]
            sources = "\n".join(
                f"  [{i}] {format_citation(m)} ({m.get('article_dir')})"
                for i, m in enumerate(metadatas, start=1)
            )
            yield f"\n\nSources:\n{sources}"

        return generate()
