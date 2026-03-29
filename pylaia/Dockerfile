FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyLaia's non-torch dependencies first, then PyLaia itself without deps
# to avoid conflicts with the old torchaudio/torchvision pins
RUN pip install --no-cache-dir \
    pytorch-lightning \
    jsonargparse[signatures] \
    editdistance \
    pillow \
    scipy

RUN pip install --no-cache-dir --no-deps \
    "git+https://gitlab.teklia.com/atr/pylaia.git"

WORKDIR /data

CMD ["/bin/bash"]
