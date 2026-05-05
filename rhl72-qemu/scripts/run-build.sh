#!/bin/bash
# Boot the VM headlessly, copy in the spec file, run rpmbuild, copy out the RPM.
set -euo pipefail

DISK=/disk/rhl72.qcow2
SPEC_SRC=/rpmbuild
RPM_OUT=/output
SSH="ssh -p 2222 -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@localhost"
SCP="scp -P 2222 -o StrictHostKeyChecking=no"

mkdir -p "$RPM_OUT"

KVM_FLAG=""
[ -e /dev/kvm ] && KVM_FLAG="-enable-kvm -cpu host"

qemu-system-i386 \
    -m 256 \
    -hda "$DISK" \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -device ne2k_pci,netdev=net0 \
    $KVM_FLAG \
    -nographic \
    -serial null \
    -monitor unix:/tmp/qemu-monitor.sock,server,nowait &

QEMU_PID=$!
trap "kill $QEMU_PID 2>/dev/null" EXIT

echo "Waiting for VM to boot and SSH to become available..."
for i in $(seq 1 60); do
    $SSH "echo ready" 2>/dev/null && break
    sleep 5
    echo "  attempt $i/60..."
done

$SSH "echo ready" || { echo "VM did not come up in time"; exit 1; }

echo "Copying build files..."
$SCP -r "$SPEC_SRC"/* root@localhost:/root/rpmbuild/

echo "Running rpmbuild inside VM..."
$SSH "cd /root && rpmbuild -ba rpmbuild/SPECS/*.spec" 2>&1

echo "Retrieving built RPMs..."
$SCP "root@localhost:/root/rpmbuild/RPMS/i386/*.rpm" "$RPM_OUT/"

echo "Done. RPMs are in $RPM_OUT:"
ls -lh "$RPM_OUT"/*.rpm

kill $QEMU_PID
