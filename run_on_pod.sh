#!/usr/bin/env bash
# One-shot runner: computes the Assistant Axis for gemma-3-27b-it and gemma-4-31B-it
# IN PARALLEL, one model per GPU, on a 2x H200/H100/A100-80GB pod.
#
# Everything runs locally on the pod — no laptop needs to stay awake. Every pipeline step is
# idempotent (skips existing outputs), each model is wrapped in a retry supervisor, and the
# two models are independent: one failing never blocks the other.
#
# Usage:
#   git clone https://github.com/timf34/GemmaAssistantAxis.git && cd GemmaAssistantAxis
#   # scp your .env to assistant-axis/.env (see .env.example), then:
#   bash run_on_pod.sh
#
#   QUESTION_COUNT=60 bash run_on_pod.sh          # half scale (~2x faster) if the night is short
#   MODELS_ONLY=gemma-3-27b bash run_on_pod.sh    # one model only
#   SKIP_PREFLIGHT=1 bash run_on_pod.sh           # resume after a crash (preflight already passed)
#   SHUTDOWN=stop bash run_on_pod.sh              # pause the pod when done (stops GPU billing)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export QUESTION_COUNT="${QUESTION_COUNT:-120}"    # 120 -> 600 responses/role
export MIN_COUNT="${MIN_COUNT:-25}"               # scales with QUESTION_COUNT (paper ratio ~4%)
export BATCH_SIZE="${BATCH_SIZE:-16}"
export EXP_ROOT="${EXP_ROOT:-/workspace/exp}"
export PRUNE_ACTIVATIONS="${PRUNE_ACTIVATIONS:-1}"   # delete raw activations after upload (~220GB/model)

echo "== [1/5] dependencies =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
(cd assistant-axis && uv sync)
# vLLM needs a driver supporting CUDA >= 12.8 (check nvidia-smi, not nvcc).
if ! (cd assistant-axis && uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"); then
  echo "!! torch cannot use this GPU — driver/CUDA mismatch."
  nvidia-smi | grep -oE 'CUDA Version: [0-9.]+' || true
  exit 1
fi
command -v tmux >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq tmux; } || true
echo "  deps OK"

echo "== [2/5] preflight =="
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  bash scripts/preflight.sh
else
  echo "  skipped (SKIP_PREFLIGHT=1)"
fi

echo "== [3/5] launching both models in parallel (one per GPU) =="
mkdir -p "$EXP_ROOT"
declare -a PIDS=() KEYS=()
launch() {  # launch <hf_id> <gpu> <key>
  if [[ -n "${MODELS_ONLY:-}" && "${MODELS_ONLY}" != "$3" ]]; then echo "  skipping $3"; return; fi
  echo "  $3: $1 on GPU $2  (log: $EXP_ROOT/$3/logs/)"
  bash scripts/supervisor.sh "$1" "$2" "$3" > "$EXP_ROOT/$3.supervisor.log" 2>&1 &
  PIDS+=($!); KEYS+=("$3")
}
launch google/gemma-3-27b-it 0 gemma-3-27b
launch google/gemma-4-31B-it 1 gemma-4-31b

echo "== [4/5] waiting (tail $EXP_ROOT/STATE.md for progress) =="
FAILED=()
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then echo "  ${KEYS[$i]}: OK"; else echo "  ${KEYS[$i]}: FAILED"; FAILED+=("${KEYS[$i]}"); fi
done

echo "== [5/5] cross-generation comparison =="
# Includes the paper's published Gemma 2 27B vectors, so the table spans three generations.
CMP=()
for k in gemma-3-27b gemma-4-31b; do
  [[ -d "$EXP_ROOT/$k/release/$k" ]] && CMP+=("$k=$EXP_ROOT/$k/release/$k")
done
if [[ ${#CMP[@]} -gt 0 ]]; then
  (cd assistant-axis && uv run python ../scripts/analyze_axis.py \
     --outdir "$EXP_ROOT/_compare" --roles90 ../roles_90.json \
     --fetch-gemma2 --compare "${CMP[@]}") || echo "  (comparison failed — per-model RESULTS.md still written)"
  (cd assistant-axis && uv run python ../scripts/upload_results.py \
     --exp "$EXP_ROOT" --key _compare --repo "${HF_RESULTS_REPO:-timf34/gemma-assistant-axis-results}") || true
fi

echo
echo "== DONE =="
echo "  per model:  $EXP_ROOT/<key>/RESULTS.md   (+ release/<key>/ : assistant_axis.pt, default_vector.pt, role_vectors/)"
echo "  comparison: $EXP_ROOT/COMPARISON.md"
echo "  progress log: $EXP_ROOT/STATE.md ; failures: $EXP_ROOT/attempts.log"
[[ ${#FAILED[@]} -gt 0 ]] && echo "  !! failed: ${FAILED[*]}"

# Optional self-shutdown so an unattended run doesn't keep billing. Requires runpodctl authed
# (runpodctl config --apiKey <KEY>); RUNPOD_POD_ID is set automatically on RunPod pods.
#   SHUTDOWN=stop      -> pause pod (billing stops, disk kept)
#   SHUTDOWN=terminate -> destroy pod (results must already be on HF — see upload step)
case "${SHUTDOWN:-}" in
  stop) RP_ACTION="stop" ;;
  terminate) RP_ACTION="remove" ;;
  *) RP_ACTION="" ;;
esac
# Optionally push the lightweight results (reports, plots, summaries — NOT the .pt vectors, which
# are ~370MB/model and live on HF) to GitHub before shutting down. Needs non-interactive git auth,
# e.g.  git remote set-url origin https://<TOKEN>@github.com/timf34/GemmaAssistantAxis.git
# SAFETY: if the push fails, a pending 'terminate' is downgraded to 'stop' so nothing is lost.
if [[ "${SAVE_TO_GIT:-0}" == "1" ]]; then
  echo "== saving reports to git =="
  mkdir -p results
  for k in gemma-3-27b gemma-4-31b; do
    [[ -d "$EXP_ROOT/$k" ]] || continue
    mkdir -p "results/$k"
    cp -f "$EXP_ROOT/$k"/{RESULTS.md,summary.json} "results/$k/" 2>/dev/null || true
    cp -f "$EXP_ROOT/$k"/*.png "results/$k/" 2>/dev/null || true
  done
  cp -f "$EXP_ROOT/COMPARISON.md" "$EXP_ROOT/STATE.md" results/ 2>/dev/null || true
  git add -f results/ 2>/dev/null || true
  git -c user.name="gemma-axis-pod" -c user.email="pod@gemma-axis.local" \
    commit -q -m "results: run finished $(date -u +%FT%TZ)" || echo "  (nothing new to commit)"
  git pull --no-rebase --no-edit 2>/dev/null || true
  if git push; then
    echo "  reports pushed to remote"
  elif [[ "$RP_ACTION" == "remove" ]]; then
    echo "  !! git push FAILED — refusing to terminate; downgrading to 'stop' to keep data."
    RP_ACTION="stop"
  fi
fi

if [[ -n "$RP_ACTION" ]]; then
  # Never terminate if a model failed or nothing was uploaded — downgrade to a pause instead.
  if [[ "$RP_ACTION" == "remove" && ${#FAILED[@]} -gt 0 ]]; then
    echo "!! failures present — downgrading terminate to stop so data survives"; RP_ACTION="stop"
  fi
  if command -v runpodctl >/dev/null 2>&1 && [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    echo "== SHUTDOWN=$SHUTDOWN -> runpodctl $RP_ACTION pod $RUNPOD_POD_ID =="
    runpodctl "$RP_ACTION" pod "$RUNPOD_POD_ID"
  else
    echo "!! cannot self-shutdown (runpodctl missing or RUNPOD_POD_ID unset) — pod left running."
  fi
fi
