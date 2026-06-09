ARG PYTORCH_IMAGE=pytorch/pytorch:2.12.0-cuda12.6-cudnn9-devel
FROM ${PYTORCH_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir transformers soundfile librosa

WORKDIR /app
RUN mkdir -p /input /output
VOLUME ["/input", "/output"]

COPY transcribe.py /app/transcribe.py

ENTRYPOINT ["python", "/app/transcribe.py"]
