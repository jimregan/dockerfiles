FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ingest.py query.py /app/

ENV OLLAMA_HOST=http://localhost:11434
ENV EMBED_MODEL=nomic-embed-text
ENV GEN_MODEL=gemma4:26b
ENV CHROMA_DIR=/data/chroma_db

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "ingest.py", "/markdown", "--persist-dir", "/data/chroma_db"]
