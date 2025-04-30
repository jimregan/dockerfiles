FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git ffmpeg
RUN git clone https://github.com/neu-vi/OmniControl/
WORKDIR /workspace/OmniControl
RUN conda env update --file environment.yml

RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install gdown spacy
RUN python -m spacy download en_core_web_sm

RUN bash prepare/download_smpl_files.sh
RUN bash prepare/download_glove.sh
RUN bash prepare/download_t2m_evaluators.sh


