#!/bin/bash
set -euo pipefail

docker compose build base
docker compose build interactive
docker compose build --profile build build
