FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
RUN git clone https://github.com/felixkreuk/UnsupSeg.git
WORKDIR UnsupSeg
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1
RUN pip install -r requirements.txt
