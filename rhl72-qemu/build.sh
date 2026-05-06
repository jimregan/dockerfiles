#!/bin/bash
set -euo pipefail

ISO_DIR=${ISO_DIR:?Please set ISO_DIR to the directory containing your RHL 7.2 ISOs}

# Step 1: base image with QEMU and scripts
docker build -f Dockerfile.base -t rhl72-base .

# Step 2: run the RHL 7.2 installer, commit disk into rhl72-installed
echo "Running RHL 7.2 installer (no KVM = slow)..."
KVM=""
[ -e /dev/kvm ] && KVM="--device /dev/kvm:/dev/kvm"

CID=$(docker run -d --privileged $KVM \
    -v "$ISO_DIR":/rhl72/isos:ro \
    -v "$(pwd)/kickstart.cfg":/rhl72/kickstart.cfg:ro \
    rhl72-base \
    bash /rhl72/scripts/install-vm.sh)

docker logs -f "$CID"
EXIT=$(docker wait "$CID")
if [ "$EXIT" != "0" ]; then
    echo "Installer failed (exit $EXIT)"
    docker rm "$CID"
    exit 1
fi
docker commit "$CID" rhl72-installed
docker rm "$CID"

# Step 3: interactive image (for spec development)
docker build -f Dockerfile.interactive -t rhl72-interactive .

echo ""
echo "Done. Next steps:"
echo "  Develop spec:  docker compose up interactive"
echo "  Build final:   docker compose build final"
echo "  Run final:     docker compose --profile final up final"
