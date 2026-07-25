#!/bin/bash
set -euo pipefail

ISO_DIR=${ISO_DIR:?Please set ISO_DIR to the directory containing your RHL 7.2 ISOs}
ROOT_PASSWORD=${ROOT_PASSWORD:-rootpassword}
INSTALL_USE_KVM=${INSTALL_USE_KVM:-0}
INSTALL_VNC=${INSTALL_VNC:-0}
BOOT_IMAGE=${BOOT_IMAGE:-boot.img}
INSTALL_CPU=${INSTALL_CPU:-pentium2}
INSTALL_MEM=${INSTALL_MEM:-256}
TREE_DIR=${TREE_DIR:-$(pwd)/tree}

case "$ISO_DIR" in
    /*) ISO_DIR_ABS=$ISO_DIR ;;
    *) ISO_DIR_ABS="$(pwd)/$ISO_DIR" ;;
esac

case "$TREE_DIR" in
    /*) TREE_DIR_ABS=$TREE_DIR ;;
    *) TREE_DIR_ABS="$(pwd)/$TREE_DIR" ;;
esac

TREE_PARENT=$(dirname "$TREE_DIR_ABS")
TREE_BASE=$(basename "$TREE_DIR_ABS")
DOCKER_DNS_ARGS=${DOCKER_DNS_ARGS:-}

[ -f "$ISO_DIR_ABS/disc1.iso" ] || { echo "Missing ISO: $ISO_DIR_ABS/disc1.iso"; exit 1; }
[ -f "$ISO_DIR_ABS/disc2.iso" ] || { echo "Missing ISO: $ISO_DIR_ABS/disc2.iso"; exit 1; }

mkdir -p output rpmbuild/BUILD rpmbuild/BUILDROOT rpmbuild/RPMS rpmbuild/SOURCES rpmbuild/SPECS rpmbuild/SRPMS

# Step 1: base image with QEMU and scripts
docker build -f Dockerfile.base -t rhl72-base .

# Step 2: build the merged install tree if it is not already present
if [ ! -d "$TREE_DIR_ABS/RedHat" ]; then
    echo "Preparing merged RHL 7.2 install tree in Docker..."
    mkdir -p "$TREE_PARENT"
    docker run --rm --privileged \
        $DOCKER_DNS_ARGS \
        -v "$ISO_DIR_ABS":/rhl72/isos:ro \
        -v "$TREE_PARENT":/rhl72/tree-parent \
        rhl72-base \
        bash /rhl72/scripts/prep-tree.sh \
            /rhl72/isos/disc1.iso \
            /rhl72/isos/disc2.iso \
            "/rhl72/tree-parent/$TREE_BASE"
fi

# Step 3: run the RHL 7.2 installer, commit disk into rhl72-installed
echo "Running RHL 7.2 installer (no KVM = slow)..."
KVM=""
[ -e /dev/kvm ] && KVM="--device /dev/kvm:/dev/kvm"
PORTS=""
[ "$INSTALL_VNC" != "0" ] && PORTS="-p 6080:6080"

CID=$(docker run -d --privileged $KVM $PORTS \
    $DOCKER_DNS_ARGS \
    -v "$ISO_DIR_ABS":/rhl72/isos:ro \
    -v "$TREE_DIR_ABS":/rhl72/tree:ro \
    -v "$(pwd)/kickstart.cfg":/rhl72/kickstart.cfg:ro \
    -e ROOT_PASSWORD="$ROOT_PASSWORD" \
    -e INSTALL_USE_KVM="$INSTALL_USE_KVM" \
    -e INSTALL_VNC="$INSTALL_VNC" \
    -e BOOT_IMAGE="$BOOT_IMAGE" \
    -e INSTALL_CPU="$INSTALL_CPU" \
    -e INSTALL_MEM="$INSTALL_MEM" \
    -e TREE_DIR=/rhl72/tree \
    rhl72-base \
    bash /rhl72/scripts/install-vm.sh)

docker logs -f "$CID"
EXIT=$(docker wait "$CID")
if [ "$EXIT" != "0" ]; then
    echo
    echo "=== LAST INSTALLER LOGS TO PASTE ==="
    docker logs --tail 80 "$CID" 2>/dev/null || true
    docker rm "$CID" >/dev/null
    echo "=== END INSTALLER LOGS exit=$EXIT ==="
    exit 1
fi
docker commit "$CID" rhl72-installed
docker rm "$CID"

# Step 4: images used after the guest OS is installed
docker build -f Dockerfile.interactive -t rhl72-interactive .
docker build -f Dockerfile.builder -t rhl72-builder .

echo ""
echo "Done. Next steps:"
echo "  Develop/debug interactively: docker compose up interactive"
echo "  Build RPM:                   docker compose --profile build run --rm builder"
echo "  Build final image:           docker compose --profile final build final"
echo "  Run final image:             docker compose --profile final up final"
