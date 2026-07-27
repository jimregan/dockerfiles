#!/bin/bash
set -euo pipefail

ISO_DIR=${ISO_DIR:?Please set ISO_DIR to the directory containing your Fedora Core 3 ISOs}
ROOT_PASSWORD=${ROOT_PASSWORD:-rootpassword}
INSTALL_USE_KVM=${INSTALL_USE_KVM:-0}
INSTALL_VNC=${INSTALL_VNC:-0}
INSTALL_CPU=${INSTALL_CPU:-pentium2}
INSTALL_MEM=${INSTALL_MEM:-256}
INSTALL_NET_MODEL=${INSTALL_NET_MODEL:-pcnet}
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
[ -f "$ISO_DIR_ABS/disc3.iso" ] || { echo "Missing ISO: $ISO_DIR_ABS/disc3.iso"; exit 1; }
[ -f "$ISO_DIR_ABS/disc4.iso" ] || { echo "Missing ISO: $ISO_DIR_ABS/disc4.iso"; exit 1; }

mkdir -p output rpmbuild/BUILD rpmbuild/BUILDROOT rpmbuild/RPMS rpmbuild/SOURCES rpmbuild/SPECS rpmbuild/SRPMS

# Step 1: base image with QEMU and scripts
docker build -f Dockerfile.base -t fc3-base .

# Step 2: build the merged install tree if it is not already present
if [ ! -d "$TREE_DIR_ABS/Fedora" ]; then
    echo "Preparing merged Fedora Core 3 install tree in Docker..."
    mkdir -p "$TREE_PARENT"
    docker run --rm --privileged \
        $DOCKER_DNS_ARGS \
        -v "$ISO_DIR_ABS":/fc3/isos:ro \
        -v "$TREE_PARENT":/fc3/tree-parent \
        fc3-base \
        bash /fc3/scripts/prep-tree.sh \
            /fc3/isos/disc1.iso \
            /fc3/isos/disc2.iso \
            /fc3/isos/disc3.iso \
            /fc3/isos/disc4.iso \
            "/fc3/tree-parent/$TREE_BASE"
fi

# Step 3: run the Fedora Core 3 installer, commit disk into fc3-installed
echo "Running Fedora Core 3 installer (no KVM = slow)..."
KVM=""
[ -e /dev/kvm ] && KVM="--device /dev/kvm:/dev/kvm"
PORTS=""
[ "$INSTALL_VNC" != "0" ] && PORTS="-p 6080:6080"

INSTALL_CONTAINER=${INSTALL_CONTAINER:-fc3-install}
docker rm -f "$INSTALL_CONTAINER" >/dev/null 2>&1 || true

CID=$(docker run -d --privileged $KVM $PORTS \
    --name "$INSTALL_CONTAINER" \
    --label fc3-qemu.role=installer \
    $DOCKER_DNS_ARGS \
    -v "$ISO_DIR_ABS":/fc3/isos:ro \
    -v "$TREE_DIR_ABS":/fc3/tree:ro \
    -v "$(pwd)/kickstart.cfg":/fc3/kickstart.cfg:ro \
    -e ROOT_PASSWORD="$ROOT_PASSWORD" \
    -e INSTALL_USE_KVM="$INSTALL_USE_KVM" \
    -e INSTALL_VNC="$INSTALL_VNC" \
    -e INSTALL_CPU="$INSTALL_CPU" \
    -e INSTALL_MEM="$INSTALL_MEM" \
    -e INSTALL_NET_MODEL="$INSTALL_NET_MODEL" \
    -e TREE_DIR=/fc3/tree \
    fc3-base \
    bash /fc3/scripts/install-vm.sh)

echo "Installer container: $CID"
echo "Status check: ./scripts/install-status.sh"

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
docker commit "$CID" fc3-installed
docker rm "$CID"

# Step 4: images used after the guest OS is installed
docker build -f Dockerfile.interactive -t fc3-interactive .
docker build -f Dockerfile.builder -t fc3-builder .

echo ""
echo "Done. Next steps:"
echo "  Develop/debug interactively: docker compose up interactive"
echo "  Build RPM:                   docker compose --profile build run --rm builder"
echo "  Build final image:           docker compose --profile final build final"
echo "  Run final image:             docker compose --profile final up final"
