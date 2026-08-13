#!/usr/bin/env bash
# Shared config for all scripts. Override any of these via environment variables.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AXIS_DIR="$REPO_DIR/assistant-axis"
EXP_ROOT="${EXP_ROOT:-/workspace/exp}"

QUESTION_COUNT="${QUESTION_COUNT:-120}"   # 120 -> 600 responses/role (~overnight); 240 -> full paper scale (~2x time)
MIN_COUNT="${MIN_COUNT:-25}"              # min fully-role-playing responses for a role vector (paper: 50 at 1200/role)
BATCH_SIZE="${BATCH_SIZE:-16}"            # activation extraction batch size (H200: 16 is conservative for 27-31B)
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-4.1-mini}"   # OpenRouter model id — NOT the bare openai name
JUDGE_BATCH="${JUDGE_BATCH:-50}"          # concurrent judge requests
JUDGE_RPS="${JUDGE_RPS:-60}"              # judge rate limit (OpenRouter-friendly; script default is 100)
PRUNE_ACTIVATIONS="${PRUNE_ACTIVATIONS:-1}"  # delete raw activations after upload — ~220GB/model at 275 roles
HF_RESULTS_REPO="${HF_RESULTS_REPO:-timf34/gemma-assistant-axis-results}"

if [[ ! -f "$AXIS_DIR/.env" ]]; then
  echo "FATAL: $AXIS_DIR/.env missing (see .env.example)" >&2
  exit 1
fi
set -a; source "$AXIS_DIR/.env"; set +a
if [[ -z "${OPENAI_API_KEY:-}" || -z "${OPENAI_BASE_URL:-}" ]]; then
  echo "FATAL: OPENAI_API_KEY / OPENAI_BASE_URL not set in .env (OpenRouter creds)" >&2
  exit 1
fi

mkdir -p "$EXP_ROOT"
state() { echo "$(date -u +%FT%TZ)  $*" >> "$EXP_ROOT/STATE.md"; }
