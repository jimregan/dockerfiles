FROM pytorch/pytorch
RUN apt update
RUN apt install -y git git-lfs g++
RUN git lfs install
RUN pip install librosa
RUN pip install git+https://github.com/pyf98/espnet@owsm-ctc
RUN pip install espnet_model_zoo
RUN git clone https://huggingface.co/pyf98/owsm_ctc_v3.2_ft_1B

COPY aligner.py /workspace
