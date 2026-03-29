FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "imagesize==1.4.1" \
    "jsonargparse[signatures]==4.7" \
    "matplotlib==3.9.2" \
    "natsort==8.4.0" \
    "pytorch-lightning==1.4.2" \
    "scipy==1.11.3" \
    "textdistance==4.6.3" \
    "torch>=1.13,<1.14" \
    "torchaudio>=0.13,<0.14" \
    "torchmetrics<0.8.0" \
    "torchvision>=0.14,<0.15" \
    "mdutils==1.6.0" \
    "prettytable==3.11.0" \
    "python-bidi==0.6.0"

RUN pip install --no-cache-dir --no-deps \
    "git+https://gitlab.teklia.com/atr/pylaia.git"

WORKDIR /data

CMD ["/bin/bash"]
