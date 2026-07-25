#!/bin/bash
# Install Fedora Core 3 into a qcow2 disk image.
set -euo pipefail

[ "${TRACE:-0}" = "1" ] && set -x

. /rhl72/scripts/common.sh

DISC1=${DISC1:-/rhl72/isos/disc1.iso}
DISK=${DISK:-/disk/rhl72.qcow2}
INSTALL_MEM=${INSTALL_MEM:-256}
TREE_DIR=${TREE_DIR:-/rhl72/tree}
TREE_IMAGE_MB=${TREE_IMAGE_MB:-0}
WORK=/tmp/rhl72-install
MNT1="$WORK/disc1"
KERNEL="$WORK/vmlinuz"
INITRD="$WORK/initrd.img"
KS_FLOPPY="$WORK/ks-floppy.img"
TREE_IMAGE="$WORK/install-tree.img"
NOVNC_PID=""
LAST_QEMU_EXIT=""
INSTALL_STATUS="not-started"
INSTALL_ERROR=""
INSTALL_SCRIPT_REV=20260725-fc3-isolinux

fail() {
    INSTALL_ERROR=$1
    echo "$INSTALL_ERROR"
    exit 1
}

cleanup() {
    local exit_code=$?

    [ -n "$NOVNC_PID" ] && kill "$NOVNC_PID" 2>/dev/null || true
    umount "$MNT1" 2>/dev/null || true
    rm -rf "$WORK" 2>/dev/null || true

    if [ "$exit_code" != "0" ]; then
        set +x
        echo
        echo "=== FC3 INSTALL FAILURE SUMMARY ==="
        echo "script_revision=$INSTALL_SCRIPT_REV"
        echo "status=$INSTALL_STATUS"
        echo "error=${INSTALL_ERROR:-unset}"
        echo "qemu_exit=${LAST_QEMU_EXIT:-unknown}"
        echo "install_tree=$TREE_DIR"
        echo "install_tree_image=$TREE_IMAGE"
        echo "install_mem=$INSTALL_MEM"
        echo "cpu_args=${CPU_FLAG:-unset}"
        echo "disk=$DISK"
        echo "=== END SUMMARY ==="
    fi

    exit "$exit_code"
}
trap cleanup EXIT

require_inputs() {
    [ -f "$DISC1" ] || fail "Missing disc1 ISO: $DISC1"
    [ -f /rhl72/kickstart.cfg ] || fail "Missing kickstart: /rhl72/kickstart.cfg"
    [ -d "$TREE_DIR/Fedora" ] || fail "Missing install tree: $TREE_DIR/Fedora"
}

prepare_workspace() {
    rm -rf "$WORK"
    mkdir -p "$MNT1"
}

extract_installer_kernel() {
    INSTALL_STATUS="extracting-installer-kernel"
    mount -o loop,ro "$DISC1" "$MNT1"

    [ -f "$MNT1/isolinux/vmlinuz" ] || fail "Cannot find $MNT1/isolinux/vmlinuz"
    [ -f "$MNT1/isolinux/initrd.img" ] || fail "Cannot find $MNT1/isolinux/initrd.img"

    cp "$MNT1/isolinux/vmlinuz" "$KERNEL"
    cp "$MNT1/isolinux/initrd.img" "$INITRD"

    umount "$MNT1"
}

make_ks_floppy() {
    INSTALL_STATUS="creating-ks-floppy"

    qemu-img create -f raw "$KS_FLOPPY" 1440k
    mkfs.msdos "$KS_FLOPPY" >/dev/null
    mcopy -o -i "$KS_FLOPPY" /rhl72/kickstart.cfg ::ks.cfg
    mcopy -o -i "$KS_FLOPPY" /rhl72/kickstart.cfg ::KS.CFG

    echo "Kickstart floppy root:"
    mdir -i "$KS_FLOPPY" ::
}

make_install_tree_disk() {
    local tree_mb
    local image_mb

    INSTALL_STATUS="creating-install-tree-disk"

    tree_mb=$(du -sm "$TREE_DIR" | awk '{print $1}')
    if [ "$TREE_IMAGE_MB" -gt 0 ]; then
        image_mb=$TREE_IMAGE_MB
    else
        image_mb=$((tree_mb + 256))
    fi

    if [ "$image_mb" -lt 1024 ]; then
        image_mb=1024
    fi

    echo "Install tree size: ${tree_mb}M"
    echo "Install tree FAT disk: ${image_mb}M"

    qemu-img create -f raw "$TREE_IMAGE" "${image_mb}M"
    mkfs.vfat -F 32 -n FC3TREE "$TREE_IMAGE"
    mcopy -s -i "$TREE_IMAGE" "$TREE_DIR"/* ::
    echo "Install tree FAT disk root:"
    mdir -i "$TREE_IMAGE" ::
}

start_display_proxy() {
    DISPLAY_ARGS="-display none -serial stdio"

    if [ "${INSTALL_VNC:-0}" != "0" ]; then
        DISPLAY_ARGS="-vnc 127.0.0.1:0 -serial mon:stdio"
        websockify --web /usr/share/novnc/ 6080 127.0.0.1:5900 &
        NOVNC_PID=$!
        echo "Installer noVNC enabled at http://localhost:6080/vnc.html"
    fi
}

run_installer() {
    CPU_FLAG=$(qemu_install_cpu_args)
    local qemu_log="$WORK/qemu.log"

    echo "Installer memory: ${INSTALL_MEM}M"
    echo "Installer CPU args: $CPU_FLAG"

    qemu-img create -f qcow2 "$DISK" 8G

    INSTALL_STATUS="qemu-running"
    set +e
    qemu-system-i386 \
        -m "$INSTALL_MEM" \
        -kernel "$KERNEL" \
        -initrd "$INITRD" \
        -append "ks=floppy text" \
        -drive file="$DISK",format=qcow2,if=ide,index=0,media=disk \
        -drive file="$TREE_IMAGE",format=raw,if=ide,index=1,media=disk \
        -drive file="$KS_FLOPPY",format=raw,if=floppy,index=0 \
        -netdev user,id=net0,hostfwd=tcp::2222-:22 \
        -device rtl8139,netdev=net0 \
        $CPU_FLAG \
        -no-acpi \
        $DISPLAY_ARGS \
        -no-reboot >"$qemu_log" 2>&1
    LAST_QEMU_EXIT=$?
    set -e

    if [ "$LAST_QEMU_EXIT" != "0" ]; then
        INSTALL_STATUS="qemu-exited-nonzero"
        echo "QEMU output:"
        cat "$qemu_log" 2>/dev/null || true
        fail "qemu exited with $LAST_QEMU_EXIT"
    fi
}

validate_install() {
    INSTALL_STATUS="validating-disk"
    require_bootable_disk "$DISK"
    INSTALL_STATUS="complete"
    echo "Install complete: $DISK"
}

echo "install-vm.sh revision: $INSTALL_SCRIPT_REV"
require_inputs
prepare_workspace
extract_installer_kernel
make_ks_floppy
make_install_tree_disk
start_display_proxy
run_installer
validate_install
