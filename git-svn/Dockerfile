FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        git-svn \
        subversion && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
