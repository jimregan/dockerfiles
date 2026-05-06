#!/bin/bash
# Install RHL 7.2 into a qcow2 disk image.
set -euxo pipefail

. /rhl72/scripts/common.sh

DISC1=${DISC1:-/rhl72/isos/disc1.iso}
DISC2=${DISC2:-/rhl72/isos/disc2.iso}
DISK=${DISK:-/disk/rhl72.qcow2}
INSTALL_BOOT=${INSTALL_BOOT:-iso}

for ISO in "$DISC1" "$DISC2"; do
    [ -f "$ISO" ] || { echo "Missing: $ISO"; exit 1; }
done

MNT1=$(mktemp -d)
MNT2=$(mktemp -d)
HTTP_PID=""
KS_PID=""

cleanup() {
    [ -n "$HTTP_PID" ] && kill "$HTTP_PID" 2>/dev/null || true
    [ -n "$KS_PID" ] && kill "$KS_PID" 2>/dev/null || true
    umount "$MNT1" 2>/dev/null || true
    umount "$MNT2" 2>/dev/null || true
    rmdir "$MNT1" "$MNT2" 2>/dev/null || true
}
trap cleanup EXIT

# Mount both discs; serve from a merged view via a small Python script
mount -o loop,ro "$DISC1" "$MNT1"
mount -o loop,ro "$DISC2" "$MNT2"

# Extract installer kernel + initrd from disc1
VMLINUZ=$(mktemp)
INITRD=$(mktemp)

find "$MNT1" -maxdepth 3 \( -name "vmlinuz" -o -name "initrd.img" \) | sort

if   [ -f "$MNT1/isolinux/vmlinuz" ];        then
    cp "$MNT1/isolinux/vmlinuz"    "$VMLINUZ"
    cp "$MNT1/isolinux/initrd.img" "$INITRD"
elif [ -f "$MNT1/images/pxeboot/vmlinuz" ];  then
    cp "$MNT1/images/pxeboot/vmlinuz"    "$VMLINUZ"
    cp "$MNT1/images/pxeboot/initrd.img" "$INITRD"
else
    echo "Cannot find installer kernel. Disc1 contents:"
    find "$MNT1" -maxdepth 3 -type f | sort
    exit 1
fi

echo "Kernel:  $VMLINUZ ($(du -h "$VMLINUZ" | cut -f1))"
echo "Initrd:  $INITRD  ($(du -h "$INITRD"  | cut -f1))"

# Serve both discs over HTTP with a minimal Python merger
# The installer expects RedHat/RPMS/ and RedHat/base/ under the URL root
cat > /tmp/serve.py << EOF
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

    def log_message(self, *a): pass

http.server.HTTPServer(('', 8080), Handler).serve_forever()
EOF

python3 /tmp/serve.py &
HTTP_PID=$!

# Copy kickstart to a known location served by the HTTP server
# Serve it from a temp dir on port 8081
mkdir -p /tmp/ks
cp /rhl72/kickstart.cfg /tmp/ks/ks.cfg
python3 -m http.server 8081 --directory /tmp/ks &
KS_PID=$!

qemu-img create -f qcow2 "$DISK" 8G

CPU_FLAG=$(qemu_install_cpu_args)
DISPLAY_ARGS="-display none -serial stdio"
APPEND_ARGS="text ks=http://10.0.2.2:8081/ks.cfg method=http://10.0.2.2:8080 ksdevice=eth0 ip=dhcp console=ttyS0,9600n8"
NOVNC_PID=""
if [ "${INSTALL_VNC:-0}" != "0" ]; then
    DISPLAY_ARGS="-vnc 127.0.0.1:0 -serial mon:stdio"
    APPEND_ARGS="text ks=http://10.0.2.2:8081/ks.cfg method=http://10.0.2.2:8080 ksdevice=eth0 ip=dhcp"
    websockify --web /usr/share/novnc/ 6080 127.0.0.1:5900 &
    NOVNC_PID=$!
    echo "Installer noVNC enabled at http://localhost:6080/vnc.html"
fi

echo "Installer boot mode: $INSTALL_BOOT"
echo "Installer append args: $APPEND_ARGS"

if [ "$INSTALL_BOOT" = "direct" ]; then
    qemu-system-i386 \
        -m 512 \
        -hda "$DISK" \
        -netdev user,id=net0,hostfwd=tcp::2222-:22 \
        -device ne2k_pci,netdev=net0 \
        $CPU_FLAG \
        -kernel "$VMLINUZ" \
        -initrd "$INITRD" \
        -append "$APPEND_ARGS" \
        $DISPLAY_ARGS \
        -no-reboot
else
    qemu-system-i386 \
        -m 512 \
        -hda "$DISK" \
        -cdrom "$DISC1" \
        -boot d \
        -netdev user,id=net0,hostfwd=tcp::2222-:22 \
        -device ne2k_pci,netdev=net0 \
        $CPU_FLAG \
        $DISPLAY_ARGS \
        -no-reboot
fi

[ -n "$NOVNC_PID" ] && kill "$NOVNC_PID" 2>/dev/null || true
require_bootable_disk "$DISK"
echo "Install complete: $DISK"
