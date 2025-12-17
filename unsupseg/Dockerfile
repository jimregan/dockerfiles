FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
RUN git clone https://github.com/felixkreuk/UnsupSeg.git
WORKDIR UnsupSeg
RUN set -eux; \
  apt-get update \
    -o Acquire::AllowInsecureRepositories=true \
    -o Acquire::AllowDowngradeToInsecureRepositories=true \
    -o Acquire::AllowUnauthenticated=true; \
  apt-get install -y --no-install-recommends \
    -o APT::Get::AllowUnauthenticated=true \
    libsndfile1;
RUN pip install -r requirements.txt
