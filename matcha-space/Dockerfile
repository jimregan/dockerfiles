FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

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
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone --depth 1 https://huggingface.co/spaces/kth-tmh/Matcha-TTS /app

# torch is already provided by the base image. huggingface_hub<0.26 keeps
# cached_download available for the old diffusers that matcha-tts pulls in.
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -r requirements.txt \
        "gradio[oauth]==3.44.3" \
        "uvicorn>=0.14.0" \
        "websockets>=10.4" \
        spaces \
        "huggingface_hub<0.26"

# Download the checkpoints into the image so there is no runtime download.
RUN python - <<'PY'
import os, urllib.request

dest = os.path.join(os.environ["MATCHA_HOME"], "matcha_tts")
os.makedirs(dest, exist_ok=True)

base = "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0"
files = {
    "matcha_ljspeech.ckpt": "matcha_ljspeech.ckpt",
    "matcha_vctk.ckpt": "matcha_vctk.ckpt",
    "hifigan_T2_v1": "generator_v1",
    "hifigan_univ_v1": "g_02500000",
}
for out_name, asset in files.items():
    out = os.path.join(dest, out_name)
    print(f"downloading {asset} -> {out}", flush=True)
    urllib.request.urlretrieve(f"{base}/{asset}", out)
PY

EXPOSE 7860
CMD ["python", "app.py"]

