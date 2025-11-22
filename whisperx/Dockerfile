FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
RUN apt update
RUN apt install -y git ffmpeg
RUN pip install git+https://github.com/m-bain/whisperX
RUN pip uninstall -y torchvision
RUN python -c 'from faster_whisper.utils import download_model; download_model("large-v3")'
ENV TRANSFORMERS_NO_TORCHVISION=1
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64::/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/"
