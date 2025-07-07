FROM pytorch/pytorch
RUN apt update
RUN apt install -y git g++ ffmpeg unzip libsndfile1
RUN pip install soundfile tensorboardX editdistance
RUN git clone https://github.com/pytorch/fairseq
WORKDIR fairseq
RUN pip install -e .
