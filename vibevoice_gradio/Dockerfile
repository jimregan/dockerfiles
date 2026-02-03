FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the repo
RUN git clone https://github.com/microsoft/VibeVoice.git .

# Install the package and dependencies
RUN pip install -e .

# Ensure gradio dependencies are complete
RUN pip install pytz

# Skipping liger-kernel - can cause compatibility issues
# RUN pip install liger-kernel || true

# Install huggingface_hub for downloading
RUN pip install huggingface_hub

# Download the model during build
ENV HF_HOME=/app/models
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/VibeVoice-ASR', local_dir='/app/models/VibeVoice-ASR')"

# Expose Gradio port
EXPOSE 7860

# Run the demo with the local model path (using sdpa attention for compatibility)
CMD ["python3", "demo/vibevoice_asr_gradio_demo.py", "--model_path", "/app/models/VibeVoice-ASR", "--attn_implementation", "sdpa", "--host", "0.0.0.0", "--port", "7860"]
