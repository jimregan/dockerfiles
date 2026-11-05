#!/bin/bash
set -e

until curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; do
    echo "Waiting for Ollama at ${OLLAMA_HOST}..."
    sleep 2
done

exec "$@"
