FROM pytorch/pytorch
RUN apt update
RUN apt install -y git g++ ffmpeg unzip
RUN git clone https://github.com/pytorch/fairseq
WORKDIR fairseq
RUN pip install -e .
