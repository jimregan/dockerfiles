FROM kaldiasr/kaldi:gpu-ubuntu18.04-cuda10.0-2023-05-08
RUN git clone https://github.com/hitachi-speech/EEND
WORKDIR EEND

RUN apt install -y libatlas-base-dev python3.7-dev
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
RUN bash /tmp/miniconda.sh -b -p $CONDA_DIR
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda update -n base -c defaults conda && \
    conda env create -f tools/environment.yml
RUN wget https://github.com/cupy/cupy/releases/download/v6.2.0/cupy_cuda100-6.2.0-cp37-cp37m-manylinux1_x86_64.whl -P /tmp && \
    conda run -n eend pip install /tmp/cupy_cuda100-6.2.0-cp37-cp37m-manylinux1_x86_64.whl chainer==6.2.0

ENV PATH=$CONDA_DIR/envs/eend/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV KALDI=/opt/kaldi

# Default shell
CMD ["bash"]

