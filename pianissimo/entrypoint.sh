#!/bin/sh
set -eu

if [ "${1:-}" = "batch" ]; then
    shift
    exec python /app/batch_transcribe.py "$@"
fi

exec python /app/transcribe.py "$@"
