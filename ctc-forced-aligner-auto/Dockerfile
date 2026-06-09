FROM pytorch/pytorch:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git

WORKDIR /output
RUN mkdir -p /input /output
VOLUME ["/input", "/output"]

COPY docker/ctc-forced-aligner/run-align.sh /usr/local/bin/run-align
RUN chmod +x /usr/local/bin/run-align

ENTRYPOINT ["run-align"]

