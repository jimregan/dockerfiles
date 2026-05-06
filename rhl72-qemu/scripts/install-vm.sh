#!/bin/bash
# Install RHL 7.2 into a qcow2 disk image.
# All prep (combining ISOs, serving HTTP, extracting boot files) happens here.
#
# Usage:
#   DISC1=/path/to/disc1.iso DISC2=... DISC3=... ./install-vm.sh
#
# Or with defaults (ISOs in /rhl72/isos/):
#   ./install-vm.sh
set -euxo pipefail

DISC1=${DISC1:-/rhl72/isos/disc1.iso}
DISC2=${DISC2:-/rhl72/isos/disc2.iso}
DISK=${DISK:-/disk/rhl72.qcow2}

if [ -f "$DISK" ]; then
    echo "Disk image already exists at $DISK, skipping install."
    exit 0
fi

for ISO in "$DISC1" "$DISC2"; do
    [ -f "$ISO" ] || { echo "Missing ISO: $ISO"; exit 1; }
done

# --- Combine all three ISOs into a single package tree ---
TREE=$(mktemp -d)
MNT=$(mktemp -d)
HTTP_PID=""
cleanup() {
    umount "$MNT" 2>/dev/null || true
    rm -rf "$MNT" "$TREE"
    [ -n "$HTTP_PID" ] && kill "$HTTP_PID" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "$TREE/RedHat/RPMS"

for ISO in "$DISC1" "$DISC2"; do
    echo "Mounting $ISO ..."
    mount -o loop,ro "$ISO" "$MNT"
    rsync -a "$MNT/RedHat/RPMS/" "$TREE/RedHat/RPMS/"
    [ -d "$MNT/RedHat/base" ] && rsync -a --ignore-existing "$MNT/RedHat/base/" "$TREE/RedHat/base/"
    umount "$MNT"
done

# Copy kickstart into the tree so it's reachable at http://10.0.2.2:8080/ks.cfg
cp /rhl72/kickstart.cfg "$TREE/ks.cfg"

# --- Extract installer kernel + initrd from disc1 ---
mount -o loop,ro "$DISC1" "$MNT"
VMLINUZ="$TREE/vmlinuz"
INITRD="$TREE/initrd.img"

echo "Looking for installer kernel on disc1..."
find "$MNT" -maxdepth 3 \( -name "vmlinuz" -o -name "initrd.img" -o -name "ramdisk.img" \) | sort

if [ -f "$MNT/isolinux/vmlinuz" ]; then
    echo "Using isolinux/"
    cp "$MNT/isolinux/vmlinuz"    "$VMLINUZ"
    cp "$MNT/isolinux/initrd.img" "$INITRD"
elif [ -f "$MNT/images/pxeboot/vmlinuz" ]; then
    echo "Using images/pxeboot/"
    cp "$MNT/images/pxeboot/vmlinuz"    "$VMLINUZ"
    cp "$MNT/images/pxeboot/initrd.img" "$INITRD"
elif [ -f "$MNT/images/vmlinuz" ]; then
    echo "Using images/vmlinuz"
    cp "$MNT/images/vmlinuz" "$VMLINUZ"
    for name in initrd.img ramdisk.img; do
        [ -f "$MNT/images/$name" ] && cp "$MNT/images/$name" "$INITRD" && break
    done
else
    echo "Could not find installer kernel. Full disc1 listing:"
    find "$MNT" -maxdepth 3 -type f | sort
    umount "$MNT"
    exit 1
fi

[ -f "$INITRD" ] || { echo "Could not find initrd"; find "$MNT" -maxdepth 3 -type f | sort; umount "$MNT"; exit 1; }

echo "Kernel: $VMLINUZ ($(du -h "$VMLINUZ" | cut -f1))"
echo "Initrd: $INITRD ($(du -h "$INITRD" | cut -f1))"

umount "$MNT"

# --- Serve the tree over HTTP (reachable from QEMU guest as 10.0.2.2:8080) ---
python3 -m http.server 8080 --directory "$TREE" &
HTTP_PID=$!
echo "HTTP server PID $HTTP_PID serving install tree on :8080"

# --- Create disk and run installer ---
qemu-img create -f qcow2 "$DISK" 8G

KVM_FLAG=""
[ -e /dev/kvm ] && KVM_FLAG="-enable-kvm -cpu host"

qemu-system-i386 \
    -m 256 \
    -hda "$DISK" \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -device ne2k_pci,netdev=net0 \
    $KVM_FLAG \
    -kernel "$VMLINUZ" \
    -initrd "$INITRD" \
    -append "ks=http://10.0.2.2:8080/ks.cfg method=http://10.0.2.2:8080 console=ttyS0" \
    -nographic \
    -serial mon:stdio

echo "Install complete. Disk image at $DISK"
