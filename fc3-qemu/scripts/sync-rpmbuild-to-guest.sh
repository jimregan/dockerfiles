#!/bin/bash
# Copy the mounted host rpmbuild tree into the running FC3 guest.
set -euo pipefail

. /fc3/scripts/common.sh

SRC=${1:-/rpmbuild}
DST=${2:-/root/rpmbuild}

[ -d "$SRC" ] || { echo "Missing rpmbuild source directory: $SRC"; exit 1; }

wait_for_ssh 60 5
configure_guest_network

echo "Syncing $SRC to guest:$DST ..."
ssh_cmd "rm -rf '$DST' && mkdir -p '$DST'"
scp_to_guest -r "$SRC"/. "root@${SSH_HOST}:$DST/"
echo "Synced rpmbuild tree into guest:$DST"
