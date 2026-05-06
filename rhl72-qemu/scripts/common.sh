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

require_bootable_disk() {
    local disk=${1:?disk path required}
    local mbr
    local signature

    if [ ! -f "$disk" ]; then
        echo "No disk image found at $disk. Run install-vm.sh first."
        return 1
    fi

    mbr=$(mktemp)
    if ! qemu-img dd -f qcow2 if="$disk" of="$mbr" bs=512 count=1 >/dev/null 2>&1; then
        rm -f "$mbr"
        echo "Cannot read the first sector of $disk."
        return 1
    fi

    signature=$(tail -c 2 "$mbr" | od -An -tx1 | tr -d ' \n')
    rm -f "$mbr"

    if [ "$signature" != "55aa" ]; then
        echo "$disk exists, but it does not contain a bootable MBR signature."
        echo "The RHL 7.2 install likely did not complete. Re-run: ISO_DIR=/path/to/isos ./build.sh"
        return 1
    fi
}
