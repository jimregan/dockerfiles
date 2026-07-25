#!/bin/bash
# Combine the Fedora Core 3 install discs into one local install tree.
# Usage: prep-tree.sh disc1.iso disc2.iso disc3.iso disc4.iso [output-dir]
set -euo pipefail

[ "$#" -ge 5 ] || { echo "Usage: prep-tree.sh disc1.iso disc2.iso disc3.iso disc4.iso [output-dir]"; exit 1; }

DISC1=$1
DISC2=$2
DISC3=$3
DISC4=$4
OUTDIR=${5:-/rhl72/tree}

MNT=${MNT:-/tmp/rhl72-prep-tree}
STAGE="$OUTDIR.incomplete"
cleanup() {
    local exit_code=$?
    umount "$MNT" 2>/dev/null || true
    rmdir "$MNT" 2>/dev/null || true
    [ "$exit_code" = "0" ] || rm -rf "$STAGE"
}
trap cleanup EXIT

rm -rf "$MNT"
mkdir -p "$MNT"
rm -rf "$STAGE"
mkdir -p "$STAGE"

echo "Extracting $DISC1 (full tree: Fedora/, images/, isolinux/) ..."
mount -o loop,ro "$DISC1" "$MNT"
rsync -a "$MNT/" "$STAGE/"
umount "$MNT"

for ISO in "$DISC2" "$DISC3" "$DISC4"; do
    echo "Extracting $ISO (RPMS only) ..."
    mount -o loop,ro "$ISO" "$MNT"
    rsync -a "$MNT/Fedora/RPMS/" "$STAGE/Fedora/RPMS/"
    umount "$MNT"
done

rm -rf "$OUTDIR"
mv "$STAGE" "$OUTDIR"

echo "Install tree ready at $OUTDIR"
echo "Package count: $(ls "$OUTDIR/Fedora/RPMS/" | wc -l)"
