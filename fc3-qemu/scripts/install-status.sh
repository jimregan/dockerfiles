#!/bin/bash
# Report whether the FC3 installer container and guest are still active.
set -euo pipefail

CID=${1:-${INSTALL_CONTAINER:-fc3-install}}

if ! docker inspect "$CID" >/dev/null 2>&1; then
    CID=$(docker ps -a \
        --filter label=fc3-qemu.role=installer \
        --format '{{.ID}} {{.CreatedAt}}' \
        | sort -k2,3 \
        | tail -n 1 \
        | awk '{print $1}')
fi

if [ -z "$CID" ]; then
    echo "No installer container found."
    echo "Start one with: INSTALL_VNC=1 ISO_DIR=fc3 ./build.sh"
    exit 1
fi

echo "Installer container: $CID"
docker ps -a --filter "id=$CID" --format 'state={{.State}} status={{.Status}} image={{.Image}}'

echo
echo "Processes:"
docker exec "$CID" sh -c 'ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E "[q]emu-system|[h]ttp.server|[w]ebsockify" || true' 2>/dev/null || {
    echo "Container is not running; cannot inspect processes."
}

echo
echo "Disk image:"
docker exec "$CID" sh -c 'ls -lh /disk/fc3.qcow2 2>/dev/null || true; qemu-img info /disk/fc3.qcow2 2>/dev/null | sed -n "1,8p" || true' 2>/dev/null || true

echo
echo "Recent installer log lines:"
docker logs --tail 80 "$CID" 2>&1 | grep -E 'Running anaconda|Formatting|Installing|Package|Complete|Traceback|Error|No volume|partition|bootloader|reboot|Install complete' || true

echo
echo "HTTP package fetches:"
docker exec "$CID" sh -c '
    log=/tmp/fc3-install/http.log
    if [ -f "$log" ]; then
        printf "successful RPM GETs: "
        grep -c "GET .*Fedora/RPMS/.* 200 " "$log" || true
        echo "recent HTTP lines:"
        tail -n 20 "$log"
    else
        echo "HTTP log unavailable."
    fi
' 2>/dev/null || true

echo
echo "QEMU monitor:"
docker exec "$CID" sh -c 'printf "info status\ninfo block\nquit\n" | nc -U /tmp/qemu-monitor.sock' 2>/dev/null || {
    echo "QEMU monitor unavailable."
}
