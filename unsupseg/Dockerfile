FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
RUN git clone https://github.com/felixkreuk/UnsupSeg.git
WORKDIR UnsupSeg
RUN conda run -n base  conda install -y --file requirements.txt
