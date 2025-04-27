#FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git libx11-dev
RUN conda create -n mocap2smplx python=3.8
RUN conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
RUN git clone https://github.com/LvXinTao/Mocap-to-SMPLX
WORKDIR /workspace/Mocap-to-SMPLX
RUN pip install -r requirements.txt

COPY run_par.sh .
