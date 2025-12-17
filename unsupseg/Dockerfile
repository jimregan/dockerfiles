FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
RUN git clone https://github.com/felixkreuk/UnsupSeg.git
WORKDIR UnsupSeg
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates gnupg && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm -f cuda-keyring_1.0-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends libsndfile1
RUN pip install -r requirements.txt
