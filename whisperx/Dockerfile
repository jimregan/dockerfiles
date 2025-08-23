FROM pytorch/pytorch
RUN apt update
RUN apt install -y git
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install git+https://github.com/m-bain/whisperX
RUN python -c 'from faster_whisper.utils import download_model; download_model("large-v3")'
