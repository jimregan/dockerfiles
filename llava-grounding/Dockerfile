FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git curl libglib2.0-0 libsm6 libxext6 libxrender-dev python3-opencv
RUN pip install --upgrade pip setuptools wheel
RUN git clone https://github.com/UX-Decoder/LLaVA-Grounding.git
WORKDIR LLaVA-Grounding
RUN pip install -e .
RUN pip install opencv-python
RUN conda install -y opencv

RUN python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Haozhangcx/llava_grounding_gd_vp", cache_dir="checkpoints")'

RUN apt install g++
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

WORKDIR /workspace
RUN git clone https://github.com/fundamentalvision/Deformable-DETR
WORKDIR Deformable-DETR
RUN pip install -r requirements.txt
WORKDIR models/ops
RUN python setup.py build install
WORKDIR /workspace/LLaVA-Grounding

ENV TRANSFORMERS_DYNAMIC_MODULE_NAME="trusted"

