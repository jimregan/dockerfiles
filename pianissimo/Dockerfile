FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ARG MODEL_ID=KlangAI/pianissimo-sv
ARG MODEL_REVISION=8f1f6d8f8bd7482a5ea1d2bfaf6ef5be61597138

WORKDIR /app

ENV HF_HOME=/cache/huggingface \
    NEMO_CACHE_DIR=/cache/nemo \
    MODEL_PATH=/models/pianissimo-sv.nemo \
    MODEL_ID=${MODEL_ID} \
    MODEL_REVISION=${MODEL_REVISION} \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg libsndfile1 \
    && pip install --no-cache-dir "nemo_toolkit[asr]==2.5.0" \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /models \
    && python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='${MODEL_ID}', filename='pianissimo-sv.nemo', revision='${MODEL_REVISION}', local_dir='/models')"

COPY transcribe.py batch_transcribe.py entrypoint.sh /app/

RUN mkdir -p /cache/huggingface /cache/nemo /audio /output \
    && chmod 755 /app/entrypoint.sh \
    && chmod -R 777 /cache /audio /output

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
