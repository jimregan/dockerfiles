FROM pytorch/pytorch
RUN apt update && apt install -y git g++ ffmpeg unzip libsndfile1 espeak-ng
RUN pip install git+https://github.com/tabahi/bournemouth-forced-aligner.git
