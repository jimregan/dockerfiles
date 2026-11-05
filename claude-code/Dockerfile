FROM node:20-slim

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    less espeak-ng vim \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code

WORKDIR /workspace

ENV ANTHROPIC_BASE_URL="http://130.237.3.105:11434"
ENV ANTHROPIC_AUTH_TOKEN=ollama
ENV ANTHROPIC_API_KEY=""

ENTRYPOINT ["claude", "--model", "gemma4:e4b"]
