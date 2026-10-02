#!/usr/bin/env bash
# Watch the box's fan-out publish progress (run LOCALLY, talks to the box).
#
#   ORNIX_SSH_PASS=... bash scripts/watch_publish.sh [interval_seconds]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
INTERVAL="${1:-120}"

while :; do
  out="$(ORNIX_SSH_PASS="${ORNIX_SSH_PASS:-}" timeout 90 bash "$HERE/box_ssh.sh" <<'REMOTE' 2>/dev/null | grep -v "Warning: Permanently added"
ROOT=/opt/ornix/work/camp7
n_ok=$(grep -l '"status": "PUBLISHED_VERIFIED"' $ROOT/releases/*REMOTE_VERIFIED.json 2>/dev/null | wc -l)
n_fail=$(grep -c '"ok": false' /opt/ornix/publish_rest.log 2>/dev/null || echo 0)
alive=$(ps -p "$(cat /opt/ornix/publish_rest.pid 2>/dev/null)" >/dev/null 2>&1 && echo 1 || echo 0)
echo "verified=$n_ok failed=$n_fail running=$alive"
REMOTE
)"
  echo "[$(date +%H:%M:%S)] ${out:-unreachable}"
  case "$out" in
    *"running=0") echo "publish finished"; break ;;
  esac
  sleep "$INTERVAL"
done