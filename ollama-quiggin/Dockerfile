FROM ollama/ollama:latest as builder

COPY qwen-3.5-9b/Modelfile .

#RUN ollama create mymodel -f Modelfile

RUN nohup ollama serve > /tmp/ollama.log 2>&1 & \
    sleep 10 && \
    ollama pull qwen3.5:9b && \
    ollama create quiggin -f Modelfile

FROM ollama/ollama:latest

COPY --from=builder /root/.ollama /root/.ollama

EXPOSE 11434
ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]
