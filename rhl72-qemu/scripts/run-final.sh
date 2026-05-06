#!/bin/bash
# Boot the final RHL 7.2 image with Mozilla + Tcl Plugin installed.
# VNC on :0 (port 5900) — connect with any VNC client.
# The VM will start an X session with Mozilla on boot.
set -euo pipefail

. /rhl72/scripts/common.sh

DISK=/disk/rhl72.qcow2
require_bootable_disk "$DISK"

KVM_FLAG=$(qemu_kvm_args)

echo "noVNC available at: http://localhost:6080/vnc.html"
echo "SSH forwarded to port 2222."

qemu-system-i386 \
    -m 256 \
    -hda "$DISK" \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -device ne2k_pci,netdev=net0 \
    $KVM_FLAG \
    -boot order=c \
    -vnc 127.0.0.1:0 &

QEMU_PID=$!

trap "kill $QEMU_PID 2>/dev/null || true" EXIT

websockify --web /usr/share/novnc/ 6080 127.0.0.1:5900

wait $QEMU_PID
