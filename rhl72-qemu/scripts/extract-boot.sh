#!/bin/bash
# Extract the installer kernel and initrd from disc1 for kickstart booting.
# RHL 7.2 disc1 has them at /images/pxeboot/vmlinuz and initrd.img
set -euo pipefail

ISO=${1:?Usage: extract-boot.sh disc1.iso}
OUTDIR=${2:-/rhl72/isos}

MNT=$(mktemp -d)
cleanup() { umount "$MNT" 2>/dev/null || true; rmdir "$MNT"; }
trap cleanup EXIT

mount -o loop,ro "$ISO" "$MNT"

cp "$MNT/images/pxeboot/vmlinuz"    "$OUTDIR/disc1-vmlinuz"
cp "$MNT/images/pxeboot/initrd.img" "$OUTDIR/disc1-initrd.img"

umount "$MNT"
echo "Extracted kernel and initrd to $OUTDIR"
