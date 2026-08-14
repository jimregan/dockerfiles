#!/bin/bash
# Boot the final FC3 image with Mozilla + Tcl Plugin installed.
# VNC on :0 (port 5900) — connect with any VNC client.
# The VM will start an X session with Mozilla on boot.
set -euo pipefail

. /fc3/scripts/common.sh

DISK=/disk/fc3.qcow2
require_bootable_disk "$DISK"

KVM_FLAG=$(qemu_kvm_args)
NET_DEVICE=$(qemu_net_device_args)
SOUND_DEVICE=$(qemu_sound_device_args)
LAB_SERVER_PID=""
AUDIO_STREAM_PID=""

echo "noVNC available at: http://localhost:6080/vnc.html"
echo "SSH forwarded to port 2222."
echo "Serving lab files at: http://www.speech.kth.se/labs/analysis/"
echo "Guest audio streamed at: http://localhost:${ICECAST_PORT}/${ICECAST_MOUNT} (also playable via the noVNC page's Audio button)"

python3 /fc3/scripts/lab-server.py &
LAB_SERVER_PID=$!

start_audio_stream

qemu-system-i386 \
    -m 256 \
    -hda "$DISK" \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    $NET_DEVICE \
    $SOUND_DEVICE \
    $KVM_FLAG \
    -no-acpi \
    -vga cirrus \
    -boot order=c \
    -vnc 127.0.0.1:0 &

QEMU_PID=$!

trap "kill $QEMU_PID 2>/dev/null || true; kill $LAB_SERVER_PID 2>/dev/null || true; kill $AUDIO_STREAM_PID 2>/dev/null || true" EXIT

websockify --web /usr/share/novnc/ 6080 127.0.0.1:5900

wait $QEMU_PID
