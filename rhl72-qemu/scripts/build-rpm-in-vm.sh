#!/bin/bash
# Run inside docker build: boot the VM, build the RPM, copy it out, shut down.
set -euo pipefail

DISK=/disk/rhl72.qcow2
SSH="ssh -p 2222 -o StrictHostKeyChecking=no -o ConnectTimeout=3 root@localhost"
SCP="scp -P 2222 -o StrictHostKeyChecking=no"

mkdir -p /output

qemu-system-i386 \
    -m 512 \
    -hda "$DISK" \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -device ne2k_pci,netdev=net0 \
    -nographic \
    -serial null \
    -monitor none &

QEMU_PID=$!
trap "kill $QEMU_PID 2>/dev/null || true" EXIT

echo "Waiting for VM..."
for i in $(seq 1 60); do
    $SSH true 2>/dev/null && break
    sleep 5
done
$SSH true || { echo "VM did not come up"; exit 1; }

$SSH "mkdir -p /root/rpmbuild/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}"
$SCP -r /rpmbuild/. root@localhost:/root/rpmbuild/

$SSH "rpmbuild -ba /root/rpmbuild/SPECS/*.spec"
$SCP "root@localhost:/root/rpmbuild/RPMS/i386/*.rpm" /output/

$SSH "shutdown -h now" || true
wait $QEMU_PID || true

echo "RPM built:"
ls -lh /output/*.rpm
