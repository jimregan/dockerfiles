#!/bin/bash
# Start the RHL 7.2 VM with VNC on :0 (port 5900) and SSH forwarded to 2222.
set -euo pipefail

DISK=${1:-/disk/rhl72.qcow2}

if [ ! -f "$DISK" ]; then
    echo "No disk image found at $DISK. Run install-vm.sh first."
    exit 1
fi

KVM_FLAG=""
if [ -e /dev/kvm ]; then
    KVM_FLAG="-enable-kvm -cpu host"
    echo "KVM available, using hardware acceleration."
else
    echo "No KVM, running in software emulation (slow)."
fi

echo "VNC available on port 5900 (display :0)"
echo "SSH forwarded to port 2222 — connect with: ssh -p 2222 root@localhost"

exec qemu-system-i386 \
    -m 256 \
    -hda "$DISK" \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -device ne2k_pci,netdev=net0 \
    $KVM_FLAG \
    -vnc :0 \
    -monitor unix:/tmp/qemu-monitor.sock,server,nowait
