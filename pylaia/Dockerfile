FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyLaia's non-torch dependencies first, then PyLaia itself without deps
# to avoid conflicts with the old torchaudio/torchvision pins.
# pytorch-lightning must be pinned <1.9 as PyLaia uses convert_inf which was
# removed in later versions.
RUN pip install --no-cache-dir \
    "pytorch-lightning>=1.4,<1.9" \
    jsonargparse[signatures] \
    editdistance \
    textdistance \
    imagesize \
    prettytable \
    pillow \
    scipy \
    python-bidi \
    mdutils

RUN pip install --no-cache-dir --no-deps \
    "git+https://gitlab.teklia.com/atr/pylaia.git"

WORKDIR /data

CMD ["/bin/bash"]
