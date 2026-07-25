FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    MATCHA_HOME=/data \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        build-essential \
        libsndfile1 \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone --depth 1 https://huggingface.co/spaces/kth-tmh/Matcha-TTS /app

RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -r requirements.txt \
        "gradio[oauth]==3.43.2" \
        "uvicorn>=0.14.0" \
        "websockets>=10.4" \
        spaces \
        "huggingface_hub<0.26"

RUN python - <<'PY'
from pathlib import Path

from matcha.cli import MATCHA_URLS, VOCODER_URLS, assert_model_downloaded
from matcha.utils.utils import get_user_data_dir

LOCATION = Path(get_user_data_dir())
assert_model_downloaded(LOCATION / "matcha_ljspeech.ckpt", MATCHA_URLS["matcha_ljspeech"])
assert_model_downloaded(LOCATION / "hifigan_T2_v1", VOCODER_URLS["hifigan_T2_v1"])
assert_model_downloaded(LOCATION / "matcha_vctk.ckpt", MATCHA_URLS["matcha_vctk"])
assert_model_downloaded(LOCATION / "hifigan_univ_v1", VOCODER_URLS["hifigan_univ_v1"])
PY

EXPOSE 7860
CMD ["python", "app.py"]
