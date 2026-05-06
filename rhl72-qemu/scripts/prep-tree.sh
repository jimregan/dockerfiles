#!/bin/bash
# Combine the two RHL 7.2 ISOs into a single install tree served over HTTP.
# Usage: prep-tree.sh disc1.iso disc2.iso [output-dir]
set -euo pipefail

DISC1=${1:?Usage: prep-tree.sh disc1.iso disc2.iso}
DISC2=${2:?}
OUTDIR=${3:-/rhl72/tree}

MNT=${MNT:-/tmp/rhl72-prep-tree.$$}
cleanup() { umount "$MNT" 2>/dev/null || true; rmdir "$MNT"; }
trap cleanup EXIT

rm -rf "$MNT"
mkdir -p "$MNT"
mkdir -p "$OUTDIR/RedHat/RPMS" "$OUTDIR/RedHat/base"

for ISO in "$DISC1" "$DISC2"; do
    echo "Extracting $ISO ..."
    mount -o loop,ro "$ISO" "$MNT"

    rsync -a "$MNT/RedHat/RPMS/" "$OUTDIR/RedHat/RPMS/"

    # base dir (hdlist, comps, etc.) only needs to come from disc1
    if [ -d "$MNT/RedHat/base" ]; then
        rsync -a --ignore-existing "$MNT/RedHat/base/" "$OUTDIR/RedHat/base/"
    fi

    umount "$MNT"
done

echo "Install tree ready at $OUTDIR"
echo "Package count: $(ls "$OUTDIR/RedHat/RPMS/" | wc -l)"
