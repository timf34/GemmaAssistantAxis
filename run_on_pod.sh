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
#   DOCTOR_ONLY=1 bash run_on_pod.sh              # validate a fresh pod in ~3 min, then exit
#   SKIP_PREFLIGHT=1 bash run_on_pod.sh           # resume after a crash (preflight already passed)
#   SHUTDOWN=stop bash run_on_pod.sh              # pause the pod when done (stops GPU billing)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export QUESTION_COUNT="${QUESTION_COUNT:-120}"    # 120 -> 600 responses/role
export MIN_COUNT="${MIN_COUNT:-25}"               # scales with QUESTION_COUNT (paper ratio ~4%)
export BATCH_SIZE="${BATCH_SIZE:-16}"
export EXP_ROOT="${EXP_ROOT:-$(pwd)/exp}"   # INSIDE the repo, so small artifacts are git-tracked
export PRUNE_ACTIVATIONS="${PRUNE_ACTIVATIONS:-0}"   # keep raw activations (uploaded to a separate public dataset); 1 = delete after upload

# STEP 0 — before installing anything: is this host's driver usable at all?
# The driver is a HOST property (RunPod H200 hosts have shipped driver 550 = CUDA 12.4 across three
# templates, including the cu1281 image — the template's CUDA version is the container toolkit and
# cannot raise the driver). vLLM >= 0.19 wheels need CUDA 12.8+. Fail here, in one second, before a
# multi-minute uv sync — or enable NVIDIA forward-compat if the host supports it.
echo "== [0/6] host driver check (before anything else) =="
# Uses scripts/cuda_compat.sh so the required CUDA version comes from the INSTALLED TORCH, not a
# constant. This step originally hardcoded cuda-compat-12-8 / /usr/local/cuda-12.8/compat; on this
# pod torch is cu130, so that path can never exist, and worse, this step then OVERWROTE the working
# .cuda_compat.env with the wrong path on every launch. It also tested `-d` on the directory, which
# passes on the stripped, library-less cuda-compat package this image ships (see TROUBLESHOOTING.md).
# At step 0 the venv may not exist yet, so torch may be unqueryable: in that case fall back to any
# real compat dir on disk (glob, highest version first) and let the doctor -- which runs after deps
# install and CAN ask torch -- make the authoritative call and (re)write .cuda_compat.env.
source scripts/cuda_compat.sh
# If a previous run already enabled compat, honour it now so nvidia-smi/torch below see it.
[[ -f "$EXP_ROOT/.cuda_compat.env" ]] && source "$EXP_ROOT/.cuda_compat.env"
DRV_CUDA=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
TORCH_CUDA="$(cuda_compat_torch_cuda "$(pwd)/assistant-axis")"
MIN_CUDA="${MIN_CUDA:-${TORCH_CUDA:-12.8}}"
if [[ -z "$DRV_CUDA" ]]; then
  echo "!! nvidia-smi shows no driver CUDA version — is this a GPU pod?"; exit 1
fi
if [[ "$(printf '%s\n%s\n' "$MIN_CUDA" "$DRV_CUDA" | sort -V | head -1)" != "$MIN_CUDA" ]]; then
  echo "  driver CUDA $DRV_CUDA < $MIN_CUDA (driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1))"
  echo "  -> attempting NVIDIA forward-compatibility ($(cuda_compat_pkg "$MIN_CUDA"), supported on datacenter GPUs)"
  COMPAT_DIR="$(cuda_compat_find_dir "$MIN_CUDA")"
  if [[ -z "$COMPAT_DIR" ]]; then
    cuda_compat_install "$MIN_CUDA" || true
    COMPAT_DIR="$(cuda_compat_find_dir "$MIN_CUDA")"
  fi
  if [[ -z "$COMPAT_DIR" ]]; then
    echo "=========================================================================="
    echo " UNUSABLE HOST: driver CUDA $DRV_CUDA < $MIN_CUDA and $(cuda_compat_pkg "$MIN_CUDA") not installable."
    echo " Nothing inside the pod can raise the driver. Rent a host with driver >= 570"
    echo " (ask RunPod which region/template) — do NOT judge by the template's CUDA version."
    echo "=========================================================================="
    exit 1
  fi
  export LD_LIBRARY_PATH="$COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  mkdir -p "$EXP_ROOT"
  # Only write the env file when we could not read torch's CUDA version (fresh pod, no venv). When
  # torch IS queryable the doctor writes it after verifying the GPU actually works, so a wrong guess
  # here can never clobber a verified-good config.
  if [[ -z "$TORCH_CUDA" && ! -f "$EXP_ROOT/.cuda_compat.env" ]]; then
    echo "export LD_LIBRARY_PATH=\"$COMPAT_DIR\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"" > "$EXP_ROOT/.cuda_compat.env"
  fi
  echo "  forward-compat libs present at $COMPAT_DIR — verified against torch by the doctor after deps install"
else
  echo "  driver CUDA $DRV_CUDA >= $MIN_CUDA — ok"
fi

echo "== [1/6] dependencies =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
(cd assistant-axis && uv sync)
echo "  deps installed"

echo "== [2/6] doctor: fail-fast environment checks =="
# Every cheap check lives in scripts/doctor.sh so a broken pod is caught in the first minute,
# not overnight. Prints an explicit "SAFE TO LEAVE IT RUNNING" banner when everything passes.
if ! bash scripts/doctor.sh; then
  echo "!! environment checks failed — aborting before any GPU work. Fix the [FAIL] lines above."
  exit 1
fi
[[ "${DOCTOR_ONLY:-0}" == "1" ]] && { echo "DOCTOR_ONLY=1 -> stopping after checks."; exit 0; }

command -v tmux >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq tmux; } || true
echo "  deps OK"

echo "== [3/6] preflight =="
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  bash scripts/preflight.sh
else
  echo "  skipped (SKIP_PREFLIGHT=1)"
fi

# How many GPUs do we actually have? Never assume 2 — a single-GPU pod makes vLLM fail with
# NVMLError_InvalidArgument because CUDA_VISIBLE_DEVICES points at a device that does not exist.
NGPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
[[ "$NGPU" -ge 1 ]] || { echo "!! no GPUs visible"; exit 1; }
if [[ "$NGPU" -ge 2 ]]; then
  echo "== [4/6] $NGPU GPUs: running both models IN PARALLEL, one per GPU =="
  PARALLEL=1
else
  echo "== [4/6] 1 GPU: running the two models SEQUENTIALLY on GPU 0 (roughly 2x wall-clock) =="
  PARALLEL=0
fi

mkdir -p "$EXP_ROOT"
declare -a PIDS=() KEYS=() FAILED=()
launch() {  # launch <hf_id> <gpu> <key>
  if [[ -n "${MODELS_ONLY:-}" && "${MODELS_ONLY}" != "$3" ]]; then echo "  skipping $3"; return; fi
  echo "  $3: $1 on GPU $2  (log: $EXP_ROOT/$3/logs/)"
  bash scripts/supervisor.sh "$1" "$2" "$3" > "$EXP_ROOT/$3.supervisor.log" 2>&1 &
  local pid=$!
  if [[ "$PARALLEL" == "1" ]]; then
    PIDS+=("$pid"); KEYS+=("$3")
  else
    # One GPU: finish this model before starting the next, so they never contend for memory.
    if wait "$pid"; then echo "  $3: OK"; else echo "  $3: FAILED"; FAILED+=("$3"); fi
  fi
}
launch google/gemma-3-27b-it 0 gemma-3-27b
launch google/gemma-4-31B-it $(( NGPU >= 2 ? 1 : 0 )) gemma-4-31b

echo "== [5/6] waiting (tail $EXP_ROOT/STATE.md for progress) =="
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then echo "  ${KEYS[$i]}: OK"; else echo "  ${KEYS[$i]}: FAILED"; FAILED+=("${KEYS[$i]}"); fi
done

echo "== [6/6] cross-generation comparison =="
# A fresh pod has no memory of earlier runs, so restore any prior model's release from the HF
# results dataset first — otherwise the table quietly covers only what ran on THIS pod.
# Includes the paper's published Gemma 2 27B vectors, so the table spans three generations.
for k in gemma-3-27b gemma-4-31b; do
  if [[ ! -d "$EXP_ROOT/$k/release/$k" ]]; then
    (cd assistant-axis && uv run python ../scripts/fetch_prior_results.py --key "$k" \
       --exp-root "$EXP_ROOT" --repo "${HF_RESULTS_REPO:-timf34/gemma-assistant-axis-results}") \
       || echo "  (no prior results for $k — it will be absent from the comparison)"
  fi
done
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
  echo "== saving experiment artifacts to git =="
  # exp/ lives inside the repo; .gitignore excludes only the bulky binaries (activations, responses,
  # vectors, .pt) which go to HF. Everything else — plots, reports, scores, gate tables, comparison,
  # logs — is committed wholesale, so nothing small is ever left behind on the pod.
  git add -A exp/ 2>/dev/null || true
  # legacy layout: also keep results/ populated for anything reading the old path
  mkdir -p results && cp -rf exp/COMPARISON.md exp/STATE.md results/ 2>/dev/null || true
  git add -A results/ 2>/dev/null || true
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
