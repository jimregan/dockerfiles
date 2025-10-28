FROM huggingface/transformers-pytorch-gpu:4.35.2
RUN apt update
RUN apt install -y git
RUN git clone https://github.com/lingjzhu/CharsiuG2P.git
WORKDIR CharsiuG2P
COPY prompt-change.patch /tmp
RUN git apply /tmp/prompt-change.patch
RUN pip install labml-nn jiwer==3.1.0
