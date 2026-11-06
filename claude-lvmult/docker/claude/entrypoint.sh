#!/bin/bash
set -euo pipefail

# 1. Egress firewall (allowlist). Fail closed: if it doesn't come up, don't start.
sudo /usr/local/bin/init-firewall.sh

# 2. SSH key for `ssh worker`. Regenerated per start; worker reads /keys/authorized_keys.
mkdir -p ~/.ssh && chmod 700 ~/.ssh
if [ ! -f ~/.ssh/id_ed25519 ]; then
    ssh-keygen -q -t ed25519 -N '' -f ~/.ssh/id_ed25519
fi
cp ~/.ssh/id_ed25519.pub /keys/authorized_keys
cat > ~/.ssh/config <<'EOF'
Host worker
    HostName worker
    User agent
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
    ServerAliveInterval 60
EOF
chmod 600 ~/.ssh/config

exec "$@"
