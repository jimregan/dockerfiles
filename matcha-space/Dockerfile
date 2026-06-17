FROM python:3.10-slim

# get_user_data_dir() returns Path(MATCHA_HOME)/"matcha_tts", so MATCHA_HOME is
# the parent: the app reads checkpoints from /data/matcha_tts.
ENV DEBIAN_FRONTEND=noninteractive \
    MATCHA_HOME=/data \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        libsndfile1 \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pull app.py (and the rest of the Space) so we don't have to vendor it here.
RUN git clone --depth 1 https://huggingface.co/spaces/kth-tmh/Matcha-TTS /app

# Same package set the Space's builder installs, but on Python 3.10 where
# numpy==1.24.3 / cython==0.29.35 (matcha-tts build deps) and matplotlib==3.7.4
# all have prebuilt wheels, so nothing compiles from source.
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -r requirements.txt \
        "gradio[oauth]==3.44.3" \
        "uvicorn>=0.14.0" \
        "websockets>=10.4" \
        spaces \
        "huggingface_hub<0.26"

EXPOSE 7860
CMD ["python", "app.py"]

