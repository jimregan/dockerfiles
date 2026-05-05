#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 /absolute/path/to/data /absolute/path/to/output [image-name]" >&2
  exit 1
fi

DATA_DIR=$1
OUTPUT_DIR=$2
IMAGE_NAME=${3:-falcon-pdf-ocr}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

docker build -t "${IMAGE_NAME}" "${SCRIPT_DIR}"
docker run --rm \
  --gpus all \
  -v "${DATA_DIR}:/data:ro" \
  -v "${OUTPUT_DIR}:/output" \
  "${IMAGE_NAME}"
