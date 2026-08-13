#!/usr/bin/env bash
# Dumb retry loop around run_model.sh — steps are idempotent, so re-running always resumes.
# Usage: supervisor.sh <hf_model_id> <gpu_id> <short_key>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$1"; GPU="$2"; KEY="$3"

for attempt in $(seq 1 8); do
  state "[$KEY] supervisor attempt $attempt/8"
  if bash "$SCRIPTS_DIR/run_model.sh" "$MODEL" "$GPU" "$KEY"; then
    state "[$KEY] supervisor: success on attempt $attempt"
    exit 0
  fi
  rc=$?
  {
    echo "=== $(date -u +%FT%TZ) [$KEY] attempt $attempt failed (exit $rc) ==="
    for f in "$EXP_ROOT/$KEY/logs/"*.log; do echo "--- tail $f ---"; tail -20 "$f"; done
  } >> "$EXP_ROOT/attempts.log" 2>/dev/null
  sleep 120
done
state "[$KEY] supervisor: GAVE UP after 8 attempts — see attempts.log"
exit 1
