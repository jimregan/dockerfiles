#!/bin/bash
# Boot the RHL 7.2 installer with a kickstart file.
# Expects the install tree to already be served on port 8080.
# Usage: install-vm.sh [disk-image-path]
set -euo pipefail

DISK=${1:-/disk/rhl72.qcow2}
KS_URL="http://10.0.2.2:8080/ks.cfg"

if [ -f "$DISK" ]; then
    echo "Disk image already exists at $DISK, skipping install."
    exit 0
fi

qemu-img create -f qcow2 "$DISK" 8G

# 10.0.2.2 is the QEMU user-mode NAT gateway (host), so the kickstart
# and install tree are reachable from inside the guest as 10.0.2.2:8080.
qemu-system-i386 \
    -m 256 \
    -hda "$DISK" \
    -cdrom /rhl72/isos/disc1.iso \
    -boot d \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -device ne2k_pci,netdev=net0 \
    ${KVM_FLAG:-} \
    -append "ks=$KS_URL" \
    -kernel /rhl72/isos/disc1-vmlinuz \
    -initrd /rhl72/isos/disc1-initrd.img \
    -nographic \
    -serial mon:stdio

# The kernel/initrd above need to be extracted from disc1 first.
# See scripts/extract-boot.sh.
