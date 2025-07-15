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
ENV CUDA_PATH=/usr/local/cuda-10.0

RUN cp tools/env.sh.in tools/env.sh && \
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> tools/env.sh
RUN ln -sd /opt/kaldi tools/kaldi
ENV DEBIAN_FRONTEND=noninteractive
RUN apt install -y flac

RUN echo 'export train_cmd="run.pl"' >> egs/mini_librispeech/v1/cmd.sh
RUN echo 'export infer_cmd="run.pl"' >> egs/mini_librispeech/v1/cmd.sh
RUN echo 'export simu_cmd="run.pl"' >> egs/mini_librispeech/v1/cmd.sh
RUN echo 'export train_cmd="run.pl"' >> egs/callhome/v1/cmd.sh
RUN echo 'export infer_cmd="run.pl"' >> egs/callhome/v1/cmd.sh
RUN echo 'export simu_cmd="run.pl"' >> egs/callhome/v1/cmd.sh
RUN pip install librosa==0.8.1
RUN pip install 'numpy<=1.19.5'

# Default shell
CMD ["bash"]

