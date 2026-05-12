FROM ollama/ollama:latest AS builder

RUN nohup ollama serve > /tmp/ollama.log 2>&1 & \
    sleep 10 && \
    ollama pull gemma4:27b

FROM ollama/ollama:latest

COPY --from=builder /root/.ollama /root/.ollama

EXPOSE 11434
ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]
