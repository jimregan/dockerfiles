FROM node:22-bookworm

ARG UID=1000
ARG GID=1000
ARG CLAUDE_CODE_VERSION=latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        git openssh-client tmux less jq ripgrep fd-find procps vim-tiny \
        sudo iptables ipset iproute2 dnsutils curl ca-certificates \
        python3 python3-venv ffmpeg sox \
    && rm -rf /var/lib/apt/lists/*

# Replace the image's `node` user with `agent`, matching the host UID/GID.
RUN userdel -r node \
    && groupadd -o -g "${GID}" agent \
    && useradd -m -o -u "${UID}" -g "${GID}" -s /bin/bash agent

RUN npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"

# uv for CPU-side Python work in this container (pin a tag once it works)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY init-firewall.sh entrypoint.sh /usr/local/bin/
RUN chmod 755 /usr/local/bin/init-firewall.sh /usr/local/bin/entrypoint.sh \
    && echo "agent ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" > /etc/sudoers.d/agent-firewall \
    && chmod 0440 /etc/sudoers.d/agent-firewall \
    && mkdir -p /home/agent/.claude /keys && chown -R agent:agent /home/agent /keys

USER agent
ENV SHELL=/bin/bash
WORKDIR /work

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]
