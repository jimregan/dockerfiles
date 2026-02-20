FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git ffmpeg
RUN git clone https://github.com/neu-vi/OmniControl/
WORKDIR /workspace/OmniControl
RUN conda env update --file environment.yml

RUN pip install git+https://github.com/openai/CLIP.git

RUN apt-get install -y unzip

# Set env vars to use it as default
ENV CONDA_ENV_NAME=omnicontrol
ENV PATH /opt/conda/envs/${CONDA_ENV_NAME}/bin:$PATH
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc

RUN conda run -n omnicontrol python -m spacy download en_core_web_sm

