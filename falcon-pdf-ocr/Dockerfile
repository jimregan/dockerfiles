FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV DATA_DIR=/data
ENV OUTPUT_DIR=/output
ENV HF_HOME=/models/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-venv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --break-system-packages --upgrade pip && \
    python3 -m pip install --break-system-packages \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.9.0 torchvision==0.24.0 && \
    python3 -m pip install --break-system-packages -r requirements.txt

COPY batch_ocr.py .

CMD ["python3", "/app/batch_ocr.py", "--input-root", "/data", "--output-root", "/output"]
