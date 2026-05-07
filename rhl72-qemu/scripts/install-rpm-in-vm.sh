#!/bin/bash
# Run inside docker build: boot the VM, install the RPM, shut down cleanly.
# The committed container layer will contain the updated disk image.
set -euo pipefail

. /rhl72/scripts/common.sh

DISK=/disk/rhl72.qcow2
require_bootable_disk "$DISK"

if ! find /rpms -maxdepth 1 -name '*.rpm' -print -quit 2>/dev/null | grep -q .; then
    echo "No RPMs found in /rpms. Run the builder first so ./output contains RPMs."
    exit 1
fi

KVM_FLAG=$(qemu_kvm_args)

qemu-system-i386 \
    -m 512 \
    -hda "$DISK" \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -device ne2k_isa,netdev=net0,irq=10,iobase=0x300 \
    $KVM_FLAG \
    -no-acpi \
    -boot order=c \
    -nographic \
    -serial null \
    -monitor none &

QEMU_PID=$!
trap "kill $QEMU_PID 2>/dev/null || true" EXIT

wait_for_ssh

scp_to_guest /rpms/*.rpm "root@${SSH_HOST}:/tmp/"
ssh_cmd "rpm -Uvh /tmp/*.rpm"
ssh_cmd "cat > /root/.xinitrc <<'EOF'
mozilla
EOF
chmod +x /root/.xinitrc
grep -q 'startx -- :0' /etc/rc.d/rc.local || cat >> /etc/rc.d/rc.local <<'EOF'

if [ -x /usr/X11R6/bin/startx ]; then
    su - root -c 'startx -- :0' >/var/log/startx.log 2>&1 &
fi
EOF"
shutdown_guest
wait $QEMU_PID || true

echo "RPM installed into disk image."
