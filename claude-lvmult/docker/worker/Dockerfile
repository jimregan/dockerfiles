# GPU worker: CUDA + Kaldi (CPU build, for the LibriSpeech-style aligner) + Python stack.
# Reached from the claude container over SSH on an internal-only network.

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ---------- Kaldi (CPU, static libs, OpenBLAS) ----------
FROM ubuntu:22.04 AS kaldi
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ gfortran make automake autoconf libtool cmake \
        git wget curl ca-certificates zlib1g-dev unzip bzip2 xz-utils sox \
        python3 python-is-python3 \
    && rm -rf /var/lib/apt/lists/*
ARG KALDI_COMMIT=master
RUN git clone https://github.com/kaldi-asr/kaldi.git /opt/kaldi \
    && cd /opt/kaldi && git checkout "${KALDI_COMMIT}" && rm -rf .git
WORKDIR /opt/kaldi/tools
RUN extras/install_openblas.sh && make -j"$(nproc)" openfst sph2pipe
WORKDIR /opt/kaldi/src
RUN ./configure --use-cuda=no --mathlib=OPENBLAS \
        --openblas-root=../tools/OpenBLAS/install \
    && make -j"$(nproc)" depend && make -j"$(nproc)" \
    && find /opt/kaldi/src -name '*.o' -delete \
    && rm -rf /opt/kaldi/tools/OpenBLAS/OpenBLAS* /opt/kaldi/tools/openfst-*/src/*/.libs/*.o

# ---------- Runtime ----------
FROM ${CUDA_IMAGE}
ARG DEBIAN_FRONTEND=noninteractive
ARG UID=1000
ARG GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server tmux git curl ca-certificates jq less procps \
        ffmpeg sox libsndfile1 libgfortran5 python3 perl gawk \
        build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY --from=kaldi /opt/kaldi /opt/kaldi
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN groupadd -o -g "${GID}" agent \
    && useradd -m -o -u "${UID}" -g "${GID}" -s /bin/bash agent \
    && usermod -p '*' agent \
    && mkdir -p /cache/hf /keys && chown agent:agent /cache/hf /keys

# Python env (heavy, slow-changing deps only; project code lives in /work)
ENV UV_PYTHON_INSTALL_DIR=/opt/python
COPY requirements.txt /tmp/requirements.txt
COPY constraints.txt /tmp/constraints.txt
RUN uv venv /opt/venv --python 3.11 \
    && uv pip install --python /opt/venv/bin/python \
        torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 \
    && uv pip install --python /opt/venv/bin/python -r /tmp/requirements.txt -c /tmp/constraints.txt \
    && chmod -R a+rX /opt/python /opt/venv

# Environment for SSH sessions (sshd doesn't inherit Docker ENV; PAM reads this)
RUN printf '%s\n' \
    'PATH="/opt/venv/bin:/opt/kaldi/src/bin:/opt/kaldi/tools/openfst/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"' \
    'VIRTUAL_ENV="/opt/venv"' \
    'KALDI_ROOT="/opt/kaldi"' \
    'HF_HOME="/cache/hf"' \
    'HF_HUB_OFFLINE="1"' \
    'LLM_BASE_URL="http://llm:8000/v1"' \
    'LLM_MODEL_NAME="local"' \
    > /etc/environment

ENV PATH=/opt/venv/bin:/opt/kaldi/src/bin:${PATH} KALDI_ROOT=/opt/kaldi

RUN mkdir -p /run/sshd && printf '%s\n' \
    'PermitRootLogin no' \
    'PasswordAuthentication no' \
    'KbdInteractiveAuthentication no' \
    'AuthorizedKeysFile /keys/authorized_keys' \
    'StrictModes no' \
    'AllowUsers agent' \
    'UsePAM yes' \
    'X11Forwarding no' \
    > /etc/ssh/sshd_config.d/agent.conf

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 755 /usr/local/bin/entrypoint.sh
EXPOSE 22
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
