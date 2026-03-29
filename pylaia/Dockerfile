FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyLaia from source to avoid old torch version pins on PyPI
RUN pip install --no-cache-dir \
    "git+https://gitlab.teklia.com/atr/pylaia.git"

WORKDIR /data

CMD ["/bin/bash"]
