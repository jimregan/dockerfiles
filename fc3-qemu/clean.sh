#!/bin/bash
set -euo pipefail

IMAGES=(
    rhl72-final
    rhl72-builder
    rhl72-interactive
    rhl72-installed
    rhl72-base
)

containers=$(docker ps -aq --filter "ancestor=rhl72-base" \
    --filter "ancestor=rhl72-installed" \
    --filter "ancestor=rhl72-interactive" \
    --filter "ancestor=rhl72-builder" \
    --filter "ancestor=rhl72-final")

if [ -n "$containers" ]; then
    docker rm -f $containers
fi

for image in "${IMAGES[@]}"; do
    image_ids=$(docker image ls -q "$image")
    if [ -n "$image_ids" ]; then
        docker image rm -f $image_ids
    fi
done

echo "Removed RHL72 containers and images."
