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

if [ ! -d "$TREE_DIR/RedHat" ]; then
    echo "Missing install tree: $TREE_DIR/RedHat"
    echo "Create it with: sudo ./scripts/prep-tree.sh \"$ISO_DIR/disc1.iso\" \"$ISO_DIR/disc2.iso\" \"$TREE_DIR\""
    exit 1
fi

mkdir -p output rpmbuild/BUILD rpmbuild/BUILDROOT rpmbuild/RPMS rpmbuild/SOURCES rpmbuild/SPECS rpmbuild/SRPMS

# Step 1: base image with QEMU and scripts
docker build -f Dockerfile.base -t rhl72-base .

# Step 2: run the RHL 7.2 installer, commit disk into rhl72-installed
echo "Running RHL 7.2 installer (no KVM = slow)..."
KVM=""
[ -e /dev/kvm ] && KVM="--device /dev/kvm:/dev/kvm"
PORTS=""
[ "$INSTALL_VNC" != "0" ] && PORTS="-p 6080:6080"

CID=$(docker run -d --privileged $KVM $PORTS \
    -v "$ISO_DIR":/rhl72/isos:ro \
    -v "$TREE_DIR":/rhl72/tree:ro \
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

# Step 3: images used after the guest OS is installed
docker build -f Dockerfile.interactive -t rhl72-interactive .
docker build -f Dockerfile.builder -t rhl72-builder .

echo ""
echo "Done. Next steps:"
echo "  Develop/debug interactively: docker compose up interactive"
echo "  Build RPM:                   docker compose --profile build run --rm builder"
echo "  Build final image:           docker compose --profile final build final"
echo "  Run final image:             docker compose --profile final up final"
