#!/usr/bin/env bash
# Remote helper for the rented Ornix worker box.
#
# Credentials come from the environment (ORNIX_SSH_PASS), never from disk or
# the command line, so they do not leak into `ps` or shell history.
#
#   ORNIX_SSH_PASS=... scripts/box_ssh.sh            # interactive session
#   ORNIX_SSH_PASS=... scripts/box_ssh.sh < remote.sh # run a script over stdin
#
# Host/port/user come from ORNIX_SSH_HOST / ORNIX_SSH_PORT / ORNIX_SSH_USER;
# nothing about the box is hardcoded here.
set -u

HOST="${ORNIX_SSH_HOST:-n2.ckey.vn}"
PORT="${ORNIX_SSH_PORT:-3027}"
USER="${ORNIX_SSH_USER:-root}"

if [ -z "${ORNIX_SSH_PASS:-}" ]; then
  echo "ORNIX_SSH_PASS is not set" >&2
  exit 2
fi
export SSHPASS="${ORNIX_SSH_PASS}"

exec sshpass -e ssh \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o LogLevel=ERROR \
  -o ConnectTimeout=25 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=4 \
  -p "${PORT}" "${USER}@${HOST}" 'bash -s'