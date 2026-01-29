FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

WORKDIR /app

# Clone the repo
RUN git clone https://github.com/microsoft/VibeVoice.git .

# Install PyTorch with CUDA support first
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install flash-attn (requires torch and packaging to be installed first)
RUN pip install packaging ninja
RUN pip install flash-attn --no-build-isolation

# Install the package and dependencies
RUN pip install -e .

# Install optional liger-kernel for better performance
RUN pip install liger-kernel || true

# Install huggingface_hub for downloading
RUN pip install huggingface_hub

# Download the model during build
ENV HF_HOME=/app/models
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/VibeVoice-ASR', local_dir='/app/models/VibeVoice-ASR')"

# Expose Gradio port
EXPOSE 7860

# Run the demo with the local model path
CMD ["python3", "demo/vibevoice_asr_gradio_demo.py", "--model_path", "/app/models/VibeVoice-ASR", "--host", "0.0.0.0", "--port", "7860"]
