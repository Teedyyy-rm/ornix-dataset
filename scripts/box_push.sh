#!/usr/bin/env bash
# Copy a local file to the rented Ornix worker box.
#
#   ORNIX_SSH_PASS=... scripts/box_push.sh <local-file> <remote-path>
#
# The file is streamed over SSH; nothing is written to the remote disk except
# the destination itself, and the caller is responsible for `chmod 600` on any
# secret it pushes.
set -euo pipefail

HOST="${ORNIX_SSH_HOST:-n2.ckey.vn}"
PORT="${ORNIX_SSH_PORT:-3027}"
USER="${ORNIX_SSH_USER:-root}"

if [ -z "${ORNIX_SSH_PASS:-}" ]; then
  echo "ORNIX_SSH_PASS is not set" >&2
  exit 2
fi
if [ "$#" -ne 2 ]; then
  echo "usage: $0 <local-file> <remote-path>" >&2
  exit 2
fi

export SSHPASS="${ORNIX_SSH_PASS}"
exec sshpass -e scp \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o LogLevel=ERROR \
  -P "${PORT}" "$1" "${USER}@${HOST}:$2"