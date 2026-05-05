#!/bin/bash
# Boot the final RHL 7.2 image with Mozilla + Tcl Plugin installed.
# VNC on :0 (port 5900) — connect with any VNC client.
# The VM will start an X session with Mozilla on boot.
set -euo pipefail

DISK=/disk/rhl72.qcow2

KVM_FLAG=""
[ -e /dev/kvm ] && KVM_FLAG="-enable-kvm -cpu host"

echo "VNC on port 5900. Connect with: vncviewer localhost:5900"

exec qemu-system-i386 \
    -m 256 \
    -hda "$DISK" \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -device ne2k_pci,netdev=net0 \
    $KVM_FLAG \
    -vnc :0
