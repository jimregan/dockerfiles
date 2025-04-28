FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

USER root

ARG DEBIAN_FRONTEND=noninteractive

LABEL github_repo="https://github.com/SWivid/F5-TTS"

RUN set -x \
    && apt-get update \
    && apt-get -y install wget curl man git less openssl libssl-dev unzip unar build-essential aria2 tmux vim \
    && apt-get install -y openssh-server sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev ffmpeg \
    && apt-get install -y librdmacm1 libibumad3 librdmacm-dev libibverbs1 libibverbs-dev ibverbs-utils ibverbs-providers \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
WORKDIR /workspace

RUN git clone https://github.com/SWivid/F5-TTS.git \
    && cd F5-TTS \
    && git submodule update --init --recursive \
    && pip install -e . --no-cache-dir

# WhisperX parts
RUN pip install git+https://github.com/m-bain/whisperX
RUN pip install -U torchvision
RUN python -c 'from faster_whisper.utils import download_model; download_model("large-v3")'

ARG LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib/:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV SHELL=/bin/bash

WORKDIR /workspace/F5-TTS
