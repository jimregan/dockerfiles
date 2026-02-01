FROM jimregan/vibevoice-gradio

# Install system dependencies
RUN apt-get update && apt-get install -y bash \
    && rm -rf /var/lib/apt/lists/*

COPY transcribe.py /app/vibevoice/scripts
