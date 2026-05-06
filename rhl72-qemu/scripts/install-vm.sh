#!/bin/bash
# Install RHL 7.2 into a qcow2 disk image.
set -euxo pipefail

. /rhl72/scripts/common.sh

DISC1=${DISC1:-/rhl72/isos/disc1.iso}
DISC2=${DISC2:-/rhl72/isos/disc2.iso}
DISK=${DISK:-/disk/rhl72.qcow2}
BOOT_IMAGE=${BOOT_IMAGE:-boot.img}
INSTALL_SCRIPT_REV=20260506-mcopy-1

echo "install-vm.sh revision: $INSTALL_SCRIPT_REV"

for ISO in "$DISC1" "$DISC2"; do
    [ -f "$ISO" ] || { echo "Missing: $ISO"; exit 1; }
done

WORK=${WORK:-/tmp/rhl72-install}
MNT1="$WORK/disc1"
MNT2="$WORK/disc2"
HTTP_PID=""
NOVNC_PID=""
LAST_QEMU_EXIT=""
INSTALL_STATUS="not-started"
INSTALL_ERROR=""

cleanup() {
    local exit_code=$?
    [ -n "$HTTP_PID" ] && kill "$HTTP_PID" 2>/dev/null || true
    [ -n "$NOVNC_PID" ] && kill "$NOVNC_PID" 2>/dev/null || true
    umount "$MNT1" 2>/dev/null || true
    umount "$MNT2" 2>/dev/null || true
    rmdir "$MNT1" "$MNT2" 2>/dev/null || true
    rm -rf "$WORK" 2>/dev/null || true
    if [ "$exit_code" != "0" ]; then
        set +x
        echo
        echo "=== RHL72 INSTALL FAILURE SUMMARY ==="
        echo "script_revision=$INSTALL_SCRIPT_REV"
        echo "status=$INSTALL_STATUS"
        echo "error=${INSTALL_ERROR:-unset}"
        echo "qemu_exit=${LAST_QEMU_EXIT:-unknown}"
        echo "boot_image=$BOOT_IMAGE"
        echo "install_mem=${INSTALL_MEM:-unset}"
        echo "cpu_args=${CPU_FLAG:-unset}"
        echo "disk=$DISK"
        echo "=== END SUMMARY ==="
    fi
    exit "$exit_code"
}
trap cleanup EXIT

rm -rf "$WORK"
mkdir -p "$MNT1" "$MNT2"

mount -o loop,ro "$DISC1" "$MNT1"
mount -o loop,ro "$DISC2" "$MNT2"

cat > "$WORK/serve.py" << EOF
import http.server, os

ROOTS = ["$MNT1", "$MNT2"]

class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        path = path.split('?')[0].lstrip('/')
        for root in ROOTS:
            full = os.path.join(root, path)
            if os.path.exists(full):
                return full
        return os.path.join(ROOTS[0], path)

    def log_message(self, *a):
        pass

http.server.HTTPServer(('', 8080), Handler).serve_forever()
EOF

python3 "$WORK/serve.py" &
HTTP_PID=$!

if [ ! -f "$MNT1/images/$BOOT_IMAGE" ]; then
    echo "Cannot find $MNT1/images/$BOOT_IMAGE"
    find "$MNT1/images" -maxdepth 1 -type f -print || true
    exit 1
fi

BOOT_FLOPPY="$WORK/boot-ks.img"
SYSLINUX_CFG="$WORK/syslinux.cfg"
cp "$MNT1/images/$BOOT_IMAGE" "$BOOT_FLOPPY"

mcopy -i "$BOOT_FLOPPY" /rhl72/kickstart.cfg ::ks.cfg
mcopy -i "$BOOT_FLOPPY" ::syslinux.cfg "$SYSLINUX_CFG"

ORIGINAL_APPEND=$(awk '
    /^[[:space:]]*append[[:space:]]/ {
        sub(/^[[:space:]]*append[[:space:]]+/, "")
        print
        exit
    }
' "$SYSLINUX_CFG")
if [ -z "$ORIGINAL_APPEND" ]; then
    ORIGINAL_APPEND="initrd=initrd.img"
fi

BASE_APPEND=$(printf '%s\n' "$ORIGINAL_APPEND" | sed -E 's/(^| )ks=[^ ]+//g; s/  +/ /g; s/^ //; s/ $//')

cat > "$SYSLINUX_CFG" << EOF
default linux
prompt 0
timeout 1
label linux
  kernel vmlinuz
  append $BASE_APPEND ks=floppy
EOF

mcopy -o -i "$BOOT_FLOPPY" "$SYSLINUX_CFG" ::syslinux.cfg

echo "Boot floppy image: images/$BOOT_IMAGE"
echo "Original syslinux append: $ORIGINAL_APPEND"
echo "Patched syslinux.cfg:"
mtype -i "$BOOT_FLOPPY" ::syslinux.cfg
echo "Boot floppy root:"
mdir -i "$BOOT_FLOPPY" ::

qemu-img create -f qcow2 "$DISK" 8G

CPU_FLAG=$(qemu_install_cpu_args)
INSTALL_MEM=${INSTALL_MEM:-256}

DISPLAY_ARGS="-display none -serial stdio"
if [ "${INSTALL_VNC:-0}" != "0" ]; then
    DISPLAY_ARGS="-vnc 127.0.0.1:0 -serial mon:stdio"
    websockify --web /usr/share/novnc/ 6080 127.0.0.1:5900 &
    NOVNC_PID=$!
    echo "Installer noVNC enabled at http://localhost:6080/vnc.html"
fi

echo "Installer memory: ${INSTALL_MEM}M"
echo "Installer CPU args: $CPU_FLAG"

INSTALL_STATUS="qemu-running"
set +e
qemu-system-i386 \
    -m "$INSTALL_MEM" \
    -drive file="$DISK",format=qcow2,if=ide,index=0,media=disk \
    -drive file="$BOOT_FLOPPY",format=raw,if=floppy,index=0 \
    -boot a \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -device ne2k_isa,netdev=net0,irq=10,iobase=0x300 \
    $CPU_FLAG \
    -no-acpi \
    $DISPLAY_ARGS \
    -no-reboot
LAST_QEMU_EXIT=$?
set -e

if [ "$LAST_QEMU_EXIT" != "0" ]; then
    INSTALL_STATUS="qemu-exited-nonzero"
    INSTALL_ERROR="qemu exited with $LAST_QEMU_EXIT"
    exit "$LAST_QEMU_EXIT"
fi

INSTALL_STATUS="validating-disk"
require_bootable_disk "$DISK"
INSTALL_STATUS="complete"
echo "Install complete: $DISK"
