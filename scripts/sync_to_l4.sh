#!/usr/bin/env bash
# Sync local maba-sparse codebase to remote 4x L4 GPU server
set -euo pipefail

echo "=== Syncing maba-sparse to remote 4x L4 GPU server ==="
ssh l4-server "mkdir -p /root/maba-sparse"
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='.pytest_cache' \
    -e "ssh" /workspaces/123123/maba-sparse/ l4-server:/root/maba-sparse/

echo "=== Sync complete. Code deployed to l4-server:/root/maba-sparse ==="
