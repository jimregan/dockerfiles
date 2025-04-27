#FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
RUN apt update
RUN apt install -y git git-lfs g++
RUN git lfs install
RUN git clone https://github.com/chanhee-luke/RoboSpatial-Eval
RUN pip install numpy tqdm pyyaml datasets
RUN python RoboSpatial-Eval/download_benchmark.py robospatial
RUN pip install einops
RUN pip install accelerate
RUN pip install -U huggingface-hub
RUN pip install flash-attn==2.4.1
RUN pip install -U transformers

COPY config.yaml /workspace/RoboSpatial-Eval
COPY models--meta-llama--Meta-Llama-3-8B-Instruct /root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct

WORKDIR /workspace/RoboSpatial-Eval
