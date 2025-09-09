FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel
RUN apt update
RUN apt install -y git ffmpeg
RUN pip install git+https://github.com/m-bain/whisperX
ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
RUN pip install ctranslate2==4.5.0
RUN python -c 'from faster_whisper.utils import download_model; download_model("large-v3")'
