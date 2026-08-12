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
scp_to_guest /fc3/scripts/configure-xorg.sh "root@${SSH_HOST}:/tmp/configure-xorg.sh"
ssh_cmd "rpm -Uvh /tmp/*.rpm"
ssh_cmd "bash /tmp/configure-xorg.sh
grep -q 'www.speech.kth.se' /etc/hosts || echo '10.0.2.2 www.speech.kth.se' >> /etc/hosts
cat > /root/.xinitrc <<'EOF'
xset -dpms 2>/dev/null || true
xset s off 2>/dev/null || true
xset s noblank 2>/dev/null || true
exec mozilla -geometry 1024x768+0+0 http://www.speech.kth.se/labs/analysis/
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
