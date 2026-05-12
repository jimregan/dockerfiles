FROM ollama/ollama:latest AS builder

RUN nohup ollama serve > /tmp/ollama.log 2>&1 & \
    sleep 10 && \
    ollama pull gemma4:27b

FROM ollama/ollama:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.ollama /root/.ollama

COPY requirements.txt .
RUN pip3 install --break-system-packages ollama pymupdf Pillow tqdm

COPY gemma4_ocr.py /app/gemma4_ocr.py

ENV DATA_DIR=/data
ENV OUTPUT_DIR=/output
ENV BACKEND=ollama
ENV GEMMA_MODEL_ID=gemma4:27b

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 11434
ENTRYPOINT ["/entrypoint.sh"]
