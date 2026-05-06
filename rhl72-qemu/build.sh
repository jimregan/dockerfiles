#!/bin/bash
set -euo pipefail

ISO_DIR=${ISO_DIR:?Please set ISO_DIR to the directory containing your RHL 7.2 ISOs}

# Step 1: build the base image
docker build -f Dockerfile.base -t rhl72-base .

# Step 2: run the installer in a privileged container, commit the result
echo "Running RHL 7.2 installer..."
KVM=""
[ -e /dev/kvm ] && KVM="--device /dev/kvm:/dev/kvm"

CID=$(docker run -d --privileged $KVM \
    -v "$ISO_DIR":/rhl72/isos:ro \
    rhl72-base \
    bash /rhl72/scripts/install-vm.sh)

docker logs -f "$CID"

EXIT=$(docker wait "$CID")
if [ "$EXIT" != "0" ]; then
    echo "Installer exited with code $EXIT"
    docker rm "$CID"
    exit 1
fi

docker commit "$CID" rhl72-installed
docker rm "$CID"
echo "Done — rhl72-installed is ready. Run: docker compose up interactive"
