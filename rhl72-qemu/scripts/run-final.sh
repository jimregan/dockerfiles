#!/bin/bash
# Boot the final RHL 7.2 image with Mozilla + Tcl Plugin installed.
# VNC on :0 (port 5900) — connect with any VNC client.
# The VM will start an X session with Mozilla on boot.
set -euo pipefail

DISK=/disk/rhl72.qcow2

KVM_FLAG=""
[ -e /dev/kvm ] && KVM_FLAG="-enable-kvm -cpu host"

echo "noVNC available at: http://localhost:6080/vnc.html"

qemu-system-i386 \
    -m 256 \
    -hda "$DISK" \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -device ne2k_pci,netdev=net0 \
    $KVM_FLAG \
    -vnc 127.0.0.1:0 &

QEMU_PID=$!

websockify --web /usr/share/novnc/ 6080 127.0.0.1:5900

wait $QEMU_PID
