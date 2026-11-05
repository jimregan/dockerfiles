#!/bin/bash
# Copy a Docker-visible path into the running FC3 guest.
set -euo pipefail

. /fc3/scripts/common.sh

[ "$#" -eq 2 ] || { echo "Usage: copy-to-guest.sh CONTAINER_PATH GUEST_PATH"; exit 1; }

SRC=$1
DST=$2

[ -e "$SRC" ] || { echo "Missing source path in Docker container: $SRC"; exit 1; }

wait_for_ssh 60 5
configure_guest_network

echo "Copying $SRC to guest:$DST ..."
ssh_cmd "rm -rf '$DST' && mkdir -p '$DST'"

if [ -d "$SRC" ]; then
    scp_to_guest -r "$SRC"/. "root@${SSH_HOST}:$DST/"
else
    scp_to_guest "$SRC" "root@${SSH_HOST}:$DST/"
fi

echo "Copied to guest:$DST"
