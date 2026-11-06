#!/bin/bash
set -euo pipefail
ssh-keygen -A >/dev/null
exec /usr/sbin/sshd -D -e
