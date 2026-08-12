#!/bin/bash
# Run inside docker build: boot the VM, install the RPM, shut down cleanly.
# The committed container layer will contain the updated disk image.
set -euo pipefail

. /fc3/scripts/common.sh

DISK=/disk/fc3.qcow2
require_bootable_disk "$DISK"

if ! find /rpms -maxdepth 1 -name '*.rpm' -print -quit 2>/dev/null | grep -q .; then
    echo "No RPMs found in /rpms. Provide a Tcl plugin RPM via Dockerfile.final or ./output."
    exit 1
fi

KVM_FLAG=$(qemu_kvm_args)
NET_DEVICE=$(qemu_net_device_args)

qemu-system-i386 \
    -m 512 \
    -hda "$DISK" \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    $NET_DEVICE \
    $KVM_FLAG \
    -no-acpi \
    -boot order=c \
    -nographic \
    -serial null \
    -monitor none &

QEMU_PID=$!
trap "kill $QEMU_PID 2>/dev/null || true" EXIT

wait_for_ssh
configure_guest_network

scp_to_guest /rpms/*.rpm "root@${SSH_HOST}:/tmp/"
ssh_cmd "rpm -Uvh /tmp/*.rpm"
shutdown_guest
wait $QEMU_PID || true

echo "RPM installed into disk image."
