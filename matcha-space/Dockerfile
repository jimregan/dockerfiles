FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    MATCHA_HOME=/data/matcha_tts \
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

# Bake the checkpoints into the image. get_user_data_dir() honors MATCHA_HOME,
# so the app finds these and skips all runtime downloads.
RUN mkdir -p "$MATCHA_HOME" \
 && BASE=https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0 \
 && curl -fL -o "$MATCHA_HOME/matcha_ljspeech.ckpt" "$BASE/matcha_ljspeech.ckpt" \
 && curl -fL -o "$MATCHA_HOME/matcha_vctk.ckpt"     "$BASE/matcha_vctk.ckpt" \
 && curl -fL -o "$MATCHA_HOME/hifigan_T2_v1"        "$BASE/generator_v1" \
 && curl -fL -o "$MATCHA_HOME/hifigan_univ_v1"      "$BASE/g_02500000"

EXPOSE 7860
CMD ["python", "app.py"]

