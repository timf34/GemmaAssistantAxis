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

# Credentials. Accept .env in either the repo root or assistant-axis/ (the pipeline's own
# load_dotenv() only finds the latter, so we export everything here for all steps), and accept
# the key under OPENROUTER_API_KEY as well as OPENAI_API_KEY — the openai SDK reads the latter.
# Two runs have now died on this exact mismatch; be forgiving about it, not clever.
set -a
for envfile in "$REPO_DIR/.env" "$AXIS_DIR/.env"; do
  [[ -f "$envfile" ]] && source "$envfile"
done
set +a
: "${OPENAI_API_KEY:=${OPENROUTER_API_KEY:-}}"
: "${OPENAI_BASE_URL:=https://openrouter.ai/api/v1}"
export OPENAI_API_KEY OPENAI_BASE_URL
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "FATAL: no OpenRouter key found. Put OPENAI_API_KEY (or OPENROUTER_API_KEY) in" >&2
  echo "       $AXIS_DIR/.env or $REPO_DIR/.env — see .env.example" >&2
  exit 1
fi
# The judge scripts call load_dotenv() themselves and only look next to the pipeline, so make sure
# a root-only .env is also visible there.
if [[ ! -f "$AXIS_DIR/.env" && -f "$REPO_DIR/.env" ]]; then
  { echo "OPENAI_API_KEY=$OPENAI_API_KEY"; echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"; } > "$AXIS_DIR/.env"
  [[ -n "${HF_TOKEN:-}" ]] && echo "HF_TOKEN=$HF_TOKEN" >> "$AXIS_DIR/.env"
  echo "note: mirrored credentials from $REPO_DIR/.env to $AXIS_DIR/.env (pipeline needs them there)"
fi

mkdir -p "$EXP_ROOT"
state() { echo "$(date -u +%FT%TZ)  $*" >> "$EXP_ROOT/STATE.md"; }
