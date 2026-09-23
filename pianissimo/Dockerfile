FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /app

ENV HF_HOME=/cache/huggingface \
    NEMO_CACHE_DIR=/cache/nemo \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg libsndfile1 \
    && pip install --no-cache-dir "nemo_toolkit[asr]==2.5.0" \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY transcribe.py /app/transcribe.py

RUN mkdir -p /cache/huggingface /cache/nemo /audio && chmod -R 777 /cache /audio

ENTRYPOINT ["python", "/app/transcribe.py"]
CMD ["--help"]
