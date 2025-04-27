FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git curl libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN pip install --upgrade pip setuptools wheel
RUN pip install \
    numpy==1.26.4 \
    torch==2.1.0 \
    torchvision==0.16.0 \
    transformers==4.42.0 \
    accelerate==0.25.0 \
    einops \
    sentencepiece \
    safetensors \
    tensorflow-cpu==2.15.0
ENV TRANSFORMERS_DYNAMIC_MODULE_NAME="trusted"

# Download model and processor at build time to avoid runtime fetch
RUN python -c "from transformers import AutoModelForCausalLM, AutoProcessor; \
AutoModelForCausalLM.from_pretrained('allenai/Molmo-7B-D-0924', trust_remote_code=True, torch_dtype='auto'); \
AutoProcessor.from_pretrained('allenai/Molmo-7B-D-0924', trust_remote_code=True)"

