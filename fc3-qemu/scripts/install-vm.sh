#!/bin/bash
# Install Fedora Core 3 into a qcow2 disk image.
set -euo pipefail

[ "${TRACE:-0}" = "1" ] && set -x

. /fc3/scripts/common.sh

DISC1=${DISC1:-/fc3/isos/disc1.iso}
DISK=${DISK:-/disk/fc3.qcow2}
INSTALL_MEM=${INSTALL_MEM:-256}
TREE_DIR=${TREE_DIR:-/fc3/tree}
INSTALL_HTTP_PORT=${INSTALL_HTTP_PORT:-8000}
INSTALL_NET_MODEL=${INSTALL_NET_MODEL:-pcnet}
INSTALL_URL="http://10.0.2.2:${INSTALL_HTTP_PORT}"
TCLPLUGIN_RPM_URL=${TCLPLUGIN_RPM_URL:-https://github.com/jimregan/tclplugin/releases/download/fc3-rpm-3.1-1/tclplugin-3.1-1.i386.rpm}
SNACK_RPM_URL=${SNACK_RPM_URL:-https://github.com/jimregan/tclplugin/releases/download/fc3-rpm-3.1-1/snack-2.2.10-1.i386.rpm}
WORK=/tmp/fc3-install
MNT1="$WORK/disc1"
KERNEL="$WORK/vmlinuz"
INITRD="$WORK/initrd.img"
HTTP_ROOT="$WORK/http"
HTTP_LOG="$WORK/http.log"
NOVNC_PID=""
HTTP_PID=""
LAST_QEMU_EXIT=""
INSTALL_STATUS="not-started"
INSTALL_ERROR=""
INSTALL_SCRIPT_REV=20260727-fc3-http-method-pcnet

fail() {
    INSTALL_ERROR=$1
    echo "$INSTALL_ERROR"
    exit 1
}

cleanup() {
    local exit_code=$?

    [ -n "$NOVNC_PID" ] && kill "$NOVNC_PID" 2>/dev/null || true
    [ -n "$HTTP_PID" ] && kill "$HTTP_PID" 2>/dev/null || true
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
        echo "install_http=$INSTALL_URL/"
        echo "install_mem=$INSTALL_MEM"
        echo "install_net_model=$INSTALL_NET_MODEL"
        echo "cpu_args=${CPU_FLAG:-unset}"
        echo "disk=$DISK"
        echo "=== END SUMMARY ==="
    fi

    exit "$exit_code"
}
trap cleanup EXIT

require_inputs() {
    [ -f "$DISC1" ] || fail "Missing disc1 ISO: $DISC1"
    [ -f /fc3/kickstart.cfg ] || fail "Missing kickstart: /fc3/kickstart.cfg"
    [ -d "$TREE_DIR/Fedora" ] || fail "Missing install tree: $TREE_DIR/Fedora"
}

prepare_workspace() {
    rm -rf "$WORK"
    mkdir -p "$MNT1" "$HTTP_ROOT"
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

fetch_extra_rpms() {
    INSTALL_STATUS="fetching-extra-rpms"
    mkdir -p "$HTTP_ROOT/extra-rpms"
    echo "Fetching $TCLPLUGIN_RPM_URL"
    curl -fsSL -o "$HTTP_ROOT/extra-rpms/tclplugin.rpm" "$TCLPLUGIN_RPM_URL" \
        || fail "Failed to fetch TCLPLUGIN_RPM_URL=$TCLPLUGIN_RPM_URL"
    echo "Fetching $SNACK_RPM_URL"
    curl -fsSL -o "$HTTP_ROOT/extra-rpms/snack.rpm" "$SNACK_RPM_URL" \
        || fail "Failed to fetch SNACK_RPM_URL=$SNACK_RPM_URL"
    for rpm in tclplugin snack; do
        magic=$(head -c4 "$HTTP_ROOT/extra-rpms/$rpm.rpm" | od -An -tx1 | tr -d ' \n')
        [ "$magic" = "edabeedb" ] \
            || fail "$HTTP_ROOT/extra-rpms/$rpm.rpm is not a valid RPM"
    done
}

start_install_http() {
    local entry

    INSTALL_STATUS="starting-install-http"

    shopt -s dotglob nullglob
    for entry in "$TREE_DIR"/*; do
        ln -s "$entry" "$HTTP_ROOT"/
    done
    shopt -u dotglob nullglob

    fetch_extra_rpms

    sed "s|@INSTALL_URL@|$INSTALL_URL|g" /fc3/kickstart.cfg > "$HTTP_ROOT/ks.cfg"

    python3 -m http.server "$INSTALL_HTTP_PORT" --bind 0.0.0.0 --directory "$HTTP_ROOT" >"$HTTP_LOG" 2>&1 &
    HTTP_PID=$!

    echo "Install HTTP tree: $INSTALL_URL/"
    echo "Install HTTP log: $HTTP_LOG"
}

start_display_proxy() {
    DISPLAY_ARGS="-display none -serial stdio -monitor unix:/tmp/qemu-monitor.sock,server,nowait"

    if [ "${INSTALL_VNC:-0}" != "0" ]; then
        DISPLAY_ARGS="-vnc 127.0.0.1:0 -serial stdio -monitor unix:/tmp/qemu-monitor.sock,server,nowait"
        websockify --web /usr/share/novnc/ 6080 127.0.0.1:5900 &
        NOVNC_PID=$!
        echo "Installer noVNC enabled at http://localhost:6080/vnc.html"
    fi
}

run_installer() {
    CPU_FLAG=$(qemu_install_cpu_args)

    echo "Installer memory: ${INSTALL_MEM}M"
    echo "Installer CPU args: $CPU_FLAG"
    echo "Installer NIC model: $INSTALL_NET_MODEL"
    echo "Installer append args: ks=${INSTALL_URL}/ks.cfg method=${INSTALL_URL}/ ksdevice=eth0 ip=dhcp text console=tty0 console=ttyS0"

    qemu-img create -f qcow2 "$DISK" 8G

    INSTALL_STATUS="qemu-running"
    set +e
    qemu-system-i386 \
        -m "$INSTALL_MEM" \
        -kernel "$KERNEL" \
        -initrd "$INITRD" \
        -append "ks=${INSTALL_URL}/ks.cfg method=${INSTALL_URL}/ ksdevice=eth0 ip=dhcp text console=tty0 console=ttyS0" \
        -drive file="$DISK",format=qcow2,if=ide,index=0,media=disk \
        -netdev user,id=net0,hostfwd=tcp::2222-:22 \
        -device "$INSTALL_NET_MODEL",netdev=net0 \
        $CPU_FLAG \
        -no-acpi \
        $DISPLAY_ARGS \
        -no-reboot
    LAST_QEMU_EXIT=$?
    set -e

    if [ "$LAST_QEMU_EXIT" != "0" ]; then
        INSTALL_STATUS="qemu-exited-nonzero"
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
start_install_http
start_display_proxy
run_installer
validate_install
