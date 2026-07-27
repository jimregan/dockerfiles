#!/bin/bash

set -euo pipefail

ROOT_PASSWORD=${ROOT_PASSWORD:-rootpassword}
SSH_PORT=${SSH_PORT:-2222}
SSH_HOST=${SSH_HOST:-localhost}

qemu_kvm_args() {
    local cpu=${RUN_CPU:-pentium2}
    if [ "${USE_KVM:-1}" != "0" ] && [ -e /dev/kvm ]; then
        printf '%s\n' "-enable-kvm -cpu $cpu"
    else
        printf '%s\n' "-cpu $cpu"
    fi
}

qemu_install_cpu_args() {
    local cpu=${INSTALL_CPU:-pentium2}
    if [ "${INSTALL_USE_KVM:-0}" != "0" ] && [ -e /dev/kvm ]; then
        printf '%s\n' "-enable-kvm -cpu $cpu"
    else
        printf '%s\n' "-cpu $cpu"
    fi
}

qemu_net_device_args() {
    local model=${RUN_NET_MODEL:-rtl8139}
    printf '%s\n' "-device $model,netdev=net0"
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
        -o KexAlgorithms=+diffie-hellman-group-exchange-sha1,diffie-hellman-group1-sha1 \
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
        -o KexAlgorithms=+diffie-hellman-group-exchange-sha1,diffie-hellman-group1-sha1 \
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

configure_guest_network() {
    ssh_cmd "cat > /etc/resolv.conf <<'EOF'
nameserver 10.0.2.3
EOF

cat > /etc/sysconfig/network <<'EOF'
NETWORKING=yes
HOSTNAME=localhost.localdomain
GATEWAY=10.0.2.2
GATEWAYDEV=eth0
EOF"

    ssh_cmd "cat > /etc/sysconfig/network-scripts/ifcfg-eth0 <<'EOF'
DEVICE=eth0
BOOTPROTO=dhcp
ONBOOT=yes
USERCTL=no
PEERDNS=no
TYPE=Ethernet
EOF"

    ssh_cmd "/sbin/route -n | awk '\$1 == \"0.0.0.0\" { found = 1 } END { exit found ? 0 : 1 }' || /sbin/route add default gw 10.0.2.2 eth0"
}

shutdown_guest() {
    ssh_cmd "shutdown -h now" || true
}

require_bootable_disk() {
    local disk=${1:?disk path required}
    local mbr
    local signature
    local partitions

    if [ ! -f "$disk" ]; then
        echo "No disk image found at $disk. Run install-vm.sh first."
        return 1
    fi

    mbr=/tmp/fc3-mbr
    rm -f "$mbr"
    if ! qemu-img dd -f qcow2 if="$disk" of="$mbr" bs=512 count=1 >/dev/null 2>&1; then
        rm -f "$mbr"
        echo "Cannot read the first sector of $disk."
        return 1
    fi

    signature=$(tail -c 2 "$mbr" | od -An -tx1 | tr -d ' \n')
    partitions=$(dd if="$mbr" bs=1 skip=446 count=64 2>/dev/null | od -An -tx1 | tr -d ' \n')

    if [ "$signature" != "55aa" ]; then
        echo "$disk exists, but it does not contain a bootable MBR signature."
        echo "The FC3 install likely did not complete. Re-run: ISO_DIR=/path/to/isos ./build.sh"
        echo "First sector:"
        od -Ax -tx1 "$mbr"
        rm -f "$mbr"
        return 1
    fi

    if [ "$partitions" = "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000" ]; then
        echo "$disk has an MBR signature, but the partition table is empty."
        echo "The FC3 installer did not partition the disk successfully."
        echo "First sector:"
        od -Ax -tx1 "$mbr"
        rm -f "$mbr"
        return 1
    fi

    rm -f "$mbr"
}
