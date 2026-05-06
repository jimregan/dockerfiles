#!/bin/bash
# Install RHL 7.2 into a qcow2 disk image.
set -euxo pipefail

. /rhl72/scripts/common.sh

DISC1=${DISC1:-/rhl72/isos/disc1.iso}
DISC2=${DISC2:-/rhl72/isos/disc2.iso}
DISK=${DISK:-/disk/rhl72.qcow2}
INSTALL_BOOT=${INSTALL_BOOT:-direct}
INSTALL_SCRIPT_REV=20260506-21
KS_ARG=${KS_ARG:-ks=nfs:10.0.2.2:/export/ks/ks.cfg}

echo "install-vm.sh revision: $INSTALL_SCRIPT_REV"

for ISO in "$DISC1" "$DISC2"; do
    [ -f "$ISO" ] || { echo "Missing: $ISO"; exit 1; }
done

MNT1=$(mktemp -d)
MNT2=$(mktemp -d)
HTTP_PID=""
KS_PID=""
KS_FLOPPY=""
BOOT_FLOPPY=""
LAST_QEMU_EXIT=""
INSTALL_STATUS="not-started"
INSTALL_ERROR=""
RPCBIND_PID=""
MOUNTD_PID=""
KS_EXPORT_MOUNTED=0

cleanup() {
    local exit_code=$?
    [ -n "$HTTP_PID" ] && kill "$HTTP_PID" 2>/dev/null || true
    [ -n "$KS_PID" ] && kill "$KS_PID" 2>/dev/null || true
    [ -n "$MOUNTD_PID" ] && kill "$MOUNTD_PID" 2>/dev/null || true
    rpc.nfsd 0 2>/dev/null || true
    exportfs -ua 2>/dev/null || true
    if [ "$KS_EXPORT_MOUNTED" = "1" ]; then
        umount /export/ks 2>/dev/null || true
    fi
    [ -n "$RPCBIND_PID" ] && kill "$RPCBIND_PID" 2>/dev/null || true
    [ -n "$KS_FLOPPY" ] && rm -f "$KS_FLOPPY" 2>/dev/null || true
    [ -n "$BOOT_FLOPPY" ] && rm -f "$BOOT_FLOPPY" 2>/dev/null || true
    umount "$MNT1" 2>/dev/null || true
    umount "$MNT2" 2>/dev/null || true
    rmdir "$MNT1" "$MNT2" 2>/dev/null || true
    if [ "$exit_code" != "0" ]; then
        set +x
        echo
        echo "=== RHL72 INSTALL FAILURE SUMMARY ==="
        echo "script_revision=$INSTALL_SCRIPT_REV"
        echo "status=$INSTALL_STATUS"
        echo "error=${INSTALL_ERROR:-unset}"
        echo "qemu_exit=${LAST_QEMU_EXIT:-unknown}"
        echo "install_boot=$INSTALL_BOOT"
        echo "boot_image=${BOOT_IMAGE:-unset}"
        echo "install_mem=${INSTALL_MEM:-unset}"
        echo "cpu_args=${CPU_FLAG:-unset}"
        echo "append_args=${APPEND_ARGS:-unset}"
        echo "disk=$DISK"
        echo "=== END SUMMARY ==="
    fi
    exit "$exit_code"
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

# Serve kickstart separately. With bootnet.img, HTTP kickstart avoids old
# Anaconda's inconsistent floppy/file kickstart lookup behavior.
mkdir -p /tmp/ks
cp /rhl72/kickstart.cfg /tmp/ks/ks.cfg
python3 -m http.server 8081 --directory /tmp/ks &
KS_PID=$!

run_step() {
    local status=$1
    local output
    local rc
    shift
    INSTALL_STATUS="$status"
    echo "Running step: $status: $*"
    set +e
    output=$("$@" 2>&1)
    rc=$?
    set -e
    if [ "$rc" != "0" ]; then
        printf '%s\n' "$output"
        INSTALL_ERROR="$status failed rc=$rc: $*: $output"
        exit "$rc"
    fi
    [ -n "$output" ] && printf '%s\n' "$output"
}

# RHL 7.2's documented network kickstart path is NFS. QEMU user networking
# exposes the container as 10.0.2.2 to the guest.
INSTALL_STATUS="starting-nfs-kickstart"
mkdir -p /export/ks
mount -t tmpfs -o size=1m tmpfs /export/ks
KS_EXPORT_MOUNTED=1
cp /rhl72/kickstart.cfg /export/ks/ks.cfg
printf '%s\n' '/export/ks 10.0.2.0/24(ro,sync,insecure,no_subtree_check,no_root_squash,fsid=0)' > /etc/exports
mkdir -p /run/rpcbind /proc/fs/nfsd
modprobe nfsd 2>/dev/null || true
if ! mountpoint -q /proc/fs/nfsd; then
    INSTALL_STATUS="mount-nfsd"
    INSTALL_ERROR="mount-nfsd started but did not complete"
    echo "Running step: mount-nfsd: mount -t nfsd nfsd /proc/fs/nfsd"
    set +e
    mount -t nfsd nfsd /proc/fs/nfsd 2>/tmp/mount-nfsd.err
    rc=$?
    set -e
    if [ "$rc" != "0" ]; then
        mount_error=$(cat /tmp/mount-nfsd.err 2>/dev/null || true)
        printf '%s\n' "$mount_error"
        INSTALL_ERROR="mount-nfsd failed rc=$rc: $mount_error"
        exit "$rc"
    fi
    INSTALL_ERROR=""
fi
INSTALL_STATUS="start-rpcbind"
rpcbind -w -f &
RPCBIND_PID=$!
sleep 1
if ! kill -0 "$RPCBIND_PID" 2>/dev/null; then
    INSTALL_ERROR="rpcbind exited during startup"
    INSTALL_STATUS="start-rpcbind"
    exit 1
fi
run_step export-nfs exportfs -ra
run_step start-nfsd rpc.nfsd 8
run_step show-exports exportfs -v
rpc.mountd -F &
MOUNTD_PID=$!
sleep 1
if ! kill -0 "$MOUNTD_PID" 2>/dev/null; then
    INSTALL_ERROR="rpc.mountd exited during startup"
    INSTALL_STATUS="start-mountd"
    exit 1
fi
INSTALL_STATUS="nfs-kickstart-ready"

# Put kickstart on a virtual floppy. RHL 7.2 Anaconda is much more reliable
# with ks=floppy than with fetching ks.cfg over early installer networking.
KS_FLOPPY=$(mktemp)
dd if=/dev/zero of="$KS_FLOPPY" bs=1024 count=1440
mkfs.vfat "$KS_FLOPPY"
KSMNT=$(mktemp -d)
mount -o loop "$KS_FLOPPY" "$KSMNT"
cp /rhl72/kickstart.cfg "$KSMNT/ks.cfg"
umount "$KSMNT"
rmdir "$KSMNT"

qemu-img create -f qcow2 "$DISK" 8G

CPU_FLAG=$(qemu_install_cpu_args)
INSTALL_MEM=${INSTALL_MEM:-256}
DISPLAY_ARGS="-display none -serial stdio"
APPEND_ARGS="text $KS_ARG method=http://10.0.2.2:8080 ksdevice=eth0 ip=dhcp noapic nousb nousbstorage console=ttyS0,9600n8"
NOVNC_PID=""
if [ "${INSTALL_VNC:-0}" != "0" ]; then
    DISPLAY_ARGS="-vnc 127.0.0.1:0 -serial mon:stdio"
    APPEND_ARGS="text $KS_ARG method=http://10.0.2.2:8080 ksdevice=eth0 ip=dhcp noapic nousb nousbstorage"
    websockify --web /usr/share/novnc/ 6080 127.0.0.1:5900 &
    NOVNC_PID=$!
    echo "Installer noVNC enabled at http://localhost:6080/vnc.html"
fi

BOOT_IMAGE=${BOOT_IMAGE:-bootnet.img}
echo "Installer boot floppy image: $BOOT_IMAGE"

if [ -f "$MNT1/images/$BOOT_IMAGE" ]; then
    BOOT_FLOPPY=$(mktemp)
    cp "$MNT1/images/$BOOT_IMAGE" "$BOOT_FLOPPY"
    BOOTMNT=$(mktemp -d)
    mount -o loop "$BOOT_FLOPPY" "$BOOTMNT"
    if [ ! -f "$BOOTMNT/syslinux.cfg" ]; then
        echo "No syslinux.cfg found in $MNT1/images/$BOOT_IMAGE"
        find "$BOOTMNT" -maxdepth 1 -type f -print
        umount "$BOOTMNT"
        rmdir "$BOOTMNT"
        exit 1
    fi

    INITRD_WORK=$(mktemp -d)
    INITRD_GZ="$INITRD_WORK/initrd.img.gz"
    INITRD_IMG="$INITRD_WORK/initrd.img"
    cp "$BOOTMNT/initrd.img" "$INITRD_GZ"
    gunzip "$INITRD_GZ"
    INITRDMNT=$(mktemp -d)
    mount -o loop "$INITRD_IMG" "$INITRDMNT"
    cp /rhl72/kickstart.cfg "$INITRDMNT/ks.cfg"
    mkdir -p "$INITRDMNT/tmp"
    cp /rhl72/kickstart.cfg "$INITRDMNT/tmp/ks.cfg"
    umount "$INITRDMNT"
    rmdir "$INITRDMNT"
    gzip -9 "$INITRD_IMG"
    cp "$INITRD_IMG.gz" "$BOOTMNT/initrd.img"
    rm -rf "$INITRD_WORK"

    ORIGINAL_APPEND=$(awk '
        /^[[:space:]]*append[[:space:]]/ {
            sub(/^[[:space:]]*append[[:space:]]+/, "")
            print
            exit
        }
    ' "$BOOTMNT/syslinux.cfg")
    if [ -z "$ORIGINAL_APPEND" ]; then
        ORIGINAL_APPEND="initrd=initrd.img"
    fi
    RAMDISK_ARG=$(printf '%s\n' "$ORIGINAL_APPEND" | tr ' ' '\n' | grep '^ramdisk_size=' | head -n 1 || true)
    INITRD_ARG=$(printf '%s\n' "$ORIGINAL_APPEND" | tr ' ' '\n' | grep '^initrd=' | head -n 1 || true)
    INITRD_ARG=${INITRD_ARG:-initrd=initrd.img}
    BASE_APPEND="$INITRD_ARG $RAMDISK_ARG"

    echo "Original boot floppy append args: $ORIGINAL_APPEND"
    echo "Base boot floppy append args: $BASE_APPEND"

    cat > "$BOOTMNT/syslinux.cfg" << EOF
DEFAULT linux
PROMPT 0
TIMEOUT 1
LABEL linux
KERNEL vmlinuz
APPEND $APPEND_ARGS $BASE_APPEND
EOF
    cp /rhl72/kickstart.cfg "$BOOTMNT/ks.cfg"
    echo "Patched boot floppy syslinux.cfg:"
    cat "$BOOTMNT/syslinux.cfg"
    echo "Kickstart copied to boot floppy:"
    cat "$BOOTMNT/ks.cfg"
    echo "Patched boot floppy files:"
    find "$BOOTMNT" -maxdepth 1 -type f -printf '%f\n' | sort
    umount "$BOOTMNT"
    rmdir "$BOOTMNT"
else
    echo "Cannot find $MNT1/images/$BOOT_IMAGE"
    find "$MNT1/images" -maxdepth 1 -type f -print || true
    exit 1
fi

echo "Installer boot mode: $INSTALL_BOOT"
echo "Installer kickstart arg: $KS_ARG"
echo "Installer append args: $APPEND_ARGS"
echo "RHL72-supported kickstart forms: ks=floppy | ks=hd:fd0/ks.cfg | ks=file:/ks.cfg | ks=nfs:<server>:/<path> | ks=cdrom:/<path> | ks"
echo "Installer memory: ${INSTALL_MEM}M"
echo "Installer CPU args: $CPU_FLAG"

if [ "$INSTALL_BOOT" = "direct" ]; then
    INSTALL_STATUS="qemu-running"
    set +e
    qemu-system-i386 \
        -m "$INSTALL_MEM" \
        -drive file="$DISK",format=qcow2,if=ide,index=0,media=disk \
        -drive file="$BOOT_FLOPPY",format=raw,if=floppy,index=0 \
        -drive file="$KS_FLOPPY",format=raw,if=floppy,index=1 \
        -boot a \
        -netdev user,id=net0,hostfwd=tcp::2222-:22 \
        -device ne2k_isa,netdev=net0,irq=10,iobase=0x300 \
        $CPU_FLAG \
        -no-acpi \
        $DISPLAY_ARGS \
        -no-reboot
    LAST_QEMU_EXIT=$?
    set -e
else
    INSTALL_STATUS="qemu-running"
    set +e
    qemu-system-i386 \
        -m "$INSTALL_MEM" \
        -drive file="$DISK",format=qcow2,if=ide,index=0,media=disk \
        -drive file="$KS_FLOPPY",format=raw,if=floppy,index=0 \
        -drive file="$DISC1",format=raw,if=ide,index=2,media=cdrom \
        -boot d \
        -netdev user,id=net0,hostfwd=tcp::2222-:22 \
        -device ne2k_isa,netdev=net0,irq=10,iobase=0x300 \
        $CPU_FLAG \
        -no-acpi \
        $DISPLAY_ARGS \
        -no-reboot
    LAST_QEMU_EXIT=$?
    set -e
fi

if [ "$LAST_QEMU_EXIT" != "0" ]; then
    INSTALL_STATUS="qemu-exited-nonzero"
    exit "$LAST_QEMU_EXIT"
fi

[ -n "$NOVNC_PID" ] && kill "$NOVNC_PID" 2>/dev/null || true
INSTALL_STATUS="validating-disk"
require_bootable_disk "$DISK"
INSTALL_STATUS="complete"
echo "Install complete: $DISK"
