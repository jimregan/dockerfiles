#!/bin/bash
# Start the RHL 7.2 VM with VNC on :0 (port 5900) and SSH forwarded to 2222.
set -euo pipefail

. /rhl72/scripts/common.sh

DISK=${1:-/disk/rhl72.qcow2}
require_bootable_disk "$DISK"

if [ -e /dev/kvm ]; then
    echo "KVM available, using hardware acceleration."
else
    echo "No KVM, running in software emulation (slow)."
fi
KVM_FLAG=$(qemu_kvm_args)

echo "SSH forwarded to port 2222 — connect with: ssh -p 2222 root@localhost"
echo "noVNC available at: http://localhost:6080/vnc.html"

# VNC only on loopback — noVNC/websockify is the external interface
qemu-system-i386 \
    -m 256 \
    -hda "$DISK" \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -device ne2k_isa,netdev=net0,irq=10,iobase=0x300 \
    $KVM_FLAG \
    -no-acpi \
    -vga cirrus \
    -boot order=c \
    -vnc 127.0.0.1:0 \
    -monitor unix:/tmp/qemu-monitor.sock,server,nowait &

QEMU_PID=$!

trap "kill $QEMU_PID 2>/dev/null || true" EXIT

websockify --web /usr/share/novnc/ 6080 127.0.0.1:5900

wait $QEMU_PID
