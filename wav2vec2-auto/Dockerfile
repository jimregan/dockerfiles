FROM pytorch/pytorch:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torchaudio transformers soundfile librosa

WORKDIR /app

RUN mkdir -p /input /output

VOLUME ["/input", "/output"]

COPY transcribe.py /app/transcribe.py

ENTRYPOINT ["python", "/app/transcribe.py"]
