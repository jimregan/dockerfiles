#!/bin/bash
# Boot the VM headlessly, copy in the spec file, run rpmbuild, copy out the RPM.
set -euo pipefail

. /rhl72/scripts/common.sh

DISK=/disk/rhl72.qcow2
SPEC_SRC=/rpmbuild
RPM_OUT=/output
require_bootable_disk "$DISK"

mkdir -p "$RPM_OUT"

if ! find "$SPEC_SRC/SPECS" -maxdepth 1 -name '*.spec' -print -quit 2>/dev/null | grep -q .; then
    echo "No spec file found under $SPEC_SRC/SPECS."
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
    -monitor unix:/tmp/qemu-monitor.sock,server,nowait &

QEMU_PID=$!
trap "kill $QEMU_PID 2>/dev/null" EXIT

wait_for_ssh

echo "Copying build files..."
ssh_cmd "rm -rf /root/rpmbuild && mkdir -p /root/rpmbuild"
scp_to_guest -r "$SPEC_SRC"/. "root@${SSH_HOST}:/root/rpmbuild/"

echo "Running rpmbuild inside VM..."
ssh_cmd "cd /root && rpmbuild -ba rpmbuild/SPECS/*.spec" 2>&1

echo "Retrieving built RPMs..."
scp_to_guest "root@${SSH_HOST}:/root/rpmbuild/RPMS/i386/*.rpm" "$RPM_OUT/"

echo "Done. RPMs are in $RPM_OUT:"
ls -lh "$RPM_OUT"/*.rpm

shutdown_guest
wait $QEMU_PID || true
