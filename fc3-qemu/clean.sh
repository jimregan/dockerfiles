#!/bin/bash
set -euo pipefail

IMAGES=(
    fc3-final
    fc3-builder
    fc3-interactive
    fc3-installed
    fc3-base
)

containers=$(docker ps -aq --filter "ancestor=fc3-base" \
    --filter "ancestor=fc3-installed" \
    --filter "ancestor=fc3-interactive" \
    --filter "ancestor=fc3-builder" \
    --filter "ancestor=fc3-final")

if [ -n "$containers" ]; then
    docker rm -f $containers
fi

for image in "${IMAGES[@]}"; do
    image_ids=$(docker image ls -q "$image")
    if [ -n "$image_ids" ]; then
        docker image rm -f $image_ids
    fi
done

echo "Removed FC3 containers and images."
