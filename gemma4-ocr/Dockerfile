FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1
ENV DATA_DIR=/data
ENV OUTPUT_DIR=/output
ENV BACKEND=ollama
ENV OLLAMA_URL=http://localhost:11434
ENV GEMMA_MODEL_ID=gemma4:27b

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY gemma4_ocr.py .

CMD ["python", "/app/gemma4_ocr.py", "--input-root", "/data", "--output-root", "/output"]
