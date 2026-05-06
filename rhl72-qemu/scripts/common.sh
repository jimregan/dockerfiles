#!/bin/bash

set -euo pipefail

ROOT_PASSWORD=${ROOT_PASSWORD:-rootpassword}
SSH_PORT=${SSH_PORT:-2222}
SSH_HOST=${SSH_HOST:-localhost}

qemu_kvm_args() {
    if [ -e /dev/kvm ]; then
        printf '%s\n' "-enable-kvm -cpu host"
    fi
}

ssh_cmd() {
    sshpass -p "$ROOT_PASSWORD" ssh \
        -p "$SSH_PORT" \
        -o BatchMode=no \
        -o PreferredAuthentications=password \
        -o PubkeyAuthentication=no \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=5 \
        -o HostKeyAlgorithms=+ssh-rsa,ssh-dss \
        -o PubkeyAcceptedAlgorithms=+ssh-rsa,ssh-dss \
        -o KexAlgorithms=+diffie-hellman-group1-sha1 \
        -o Ciphers=+aes128-cbc,3des-cbc \
        "root@$SSH_HOST" "$@"
}

scp_to_guest() {
    sshpass -p "$ROOT_PASSWORD" scp \
        -P "$SSH_PORT" \
        -o BatchMode=no \
        -o PreferredAuthentications=password \
        -o PubkeyAuthentication=no \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=5 \
        -o HostKeyAlgorithms=+ssh-rsa,ssh-dss \
        -o PubkeyAcceptedAlgorithms=+ssh-rsa,ssh-dss \
        -o KexAlgorithms=+diffie-hellman-group1-sha1 \
        -o Ciphers=+aes128-cbc,3des-cbc \
        "$@"
}

wait_for_ssh() {
    local attempts=${1:-90}
    local sleep_seconds=${2:-5}

    echo "Waiting for guest SSH on ${SSH_HOST}:${SSH_PORT}..."
    for i in $(seq 1 "$attempts"); do
        if ssh_cmd true 2>/dev/null; then
            return 0
        fi
        echo "  attempt ${i}/${attempts}"
        sleep "$sleep_seconds"
    done

    echo "Guest SSH did not become available."
    return 1
}

shutdown_guest() {
    ssh_cmd "shutdown -h now" || true
}
