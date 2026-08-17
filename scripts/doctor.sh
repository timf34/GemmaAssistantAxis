#!/usr/bin/env bash
# Fail-fast environment validation. EVERY check here is cheap (seconds), so a broken pod is
# caught in the first minute — while you are still watching — instead of after you go to bed.
#
# Rule for this file: if a failure mode has ever cost us a run, it gets a check here.
#   - missing/misplaced .env, key under the wrong name          (cost: one aborted Gemma preflight)
#   - HF_HUB_ENABLE_HF_TRANSFER=1 with no hf_transfer in venv   (cost: one full night)
#   - gated/unauthorized HF repo                                (cost: one aborted preflight)
#   - transformers too old for the model architecture           (cost: gemma-4 never ran)
#   - chat template silently dropping persona instructions      (would have invalidated results)
#
# Run standalone any time:  bash scripts/doctor.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# MODELS_ONLY=gemma-4-31b narrows the checks to that model (used when unblocking gemma-4 alone).
case "${MODELS_ONLY:-}" in
  gemma-3-27b) DEFAULT_DOCTOR_MODELS="google/gemma-3-27b-it" ;;
  gemma-4-31b) DEFAULT_DOCTOR_MODELS="google/gemma-4-31B-it" ;;
  *) DEFAULT_DOCTOR_MODELS="google/gemma-3-27b-it google/gemma-4-31B-it" ;;
esac
DOCTOR_MODELS="${DOCTOR_MODELS:-$DEFAULT_DOCTOR_MODELS}"
MIN_DISK_GB="${MIN_DISK_GB:-600}"
FAILED=0
ok()   { echo "  [ok]   $*"; }
fail() { echo "  [FAIL] $*"; FAILED=$((FAILED+1)); }

echo "== doctor: driver / CUDA (checked FIRST — nothing else matters if torch cannot load) =="
# vLLM >= 0.19 wheels are built for CUDA 12.8/12.9/13.0. The DRIVER is a host property (RunPod's H200
# hosts have shipped driver 550 = CUDA 12.4 across three separate templates — the template's CUDA
# version is the container toolkit and cannot raise the driver). Two ways to proceed on an old driver:
#   1. NVIDIA forward-compat: cuda-compat-<ver> installs newer user-space libcuda under
#      /usr/local/cuda-<ver>/compat; datacenter GPUs (H100/H200/A100) support this. Try it.
#   2. If that fails: a host with a newer driver (ask RunPod which regions run 570+), or pin an
#      older vllm/torch (which cannot load gemma-4).
# The required CUDA version is read from the INSTALLED TORCH, not hardcoded: this pod resolved to
# torch 2.13.0+cu130, so the answer was cuda-compat-13-0, and the old hardcoded 12.8 path could
# never have appeared no matter how many times it was installed. See scripts/cuda_compat.sh.
source "$(dirname "${BASH_SOURCE[0]}")/cuda_compat.sh"
# HEADS UP when reading the log: nvidia-smi reports the version of whichever libcuda it loads, so
# once forward-compat is active (common.sh sources .cuda_compat.env before this runs) this prints
# "driver supports CUDA 13.0" on a box whose KERNEL driver is still 550.144.03 / CUDA 12.4 --
# `nvidia-smi` alone says 12.4, `LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat nvidia-smi` says 13.0.
# That is the compat layer working as intended, not the host driver changing (it cannot change from
# inside the container). The check below is therefore self-consistent: delete .cuda_compat.env and
# it drops back to 12.4 and re-runs the compat branch. True kernel version: /proc/driver/nvidia/version.
DRV_CUDA=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
TORCH_CUDA="$(cuda_compat_torch_cuda "$AXIS_DIR")"
MIN_CUDA="${MIN_CUDA:-${TORCH_CUDA:-12.8}}"
COMPAT_VER="${COMPAT_VER:-${MIN_CUDA//./-}}"
[[ -n "$TORCH_CUDA" ]] && echo "  venv torch is built for CUDA $TORCH_CUDA -> needs driver CUDA >= $MIN_CUDA"
torch_ok() { (cd "$AXIS_DIR" && uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1); }
if [[ -z "$DRV_CUDA" ]]; then
  fail "nvidia-smi reports no driver CUDA version"
elif [[ "$(printf '%s\n%s\n' "$MIN_CUDA" "$DRV_CUDA" | sort -V | head -1)" != "$MIN_CUDA" ]]; then
  echo "  driver supports CUDA $DRV_CUDA < $MIN_CUDA — trying NVIDIA forward-compatibility (cuda-compat-$COMPAT_VER)"
  # Locate real compat libs. Test for libcuda.so.1, NOT just a directory: this image shipped a
  # cuda-compat-12-8 dpkg entry with every library stripped out (its version is not even in NVIDIA's
  # repo), so "apt says installed" proved nothing and a bare -d test would have passed on an empty dir.
  COMPAT_DIR="$(cuda_compat_find_dir "$MIN_CUDA")"
  if [[ -z "$COMPAT_DIR" ]]; then
    cuda_compat_install "$MIN_CUDA" || true
    COMPAT_DIR="$(cuda_compat_find_dir "$MIN_CUDA")"
  fi
  if [[ -n "$COMPAT_DIR" ]]; then
    export LD_LIBRARY_PATH="$COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if torch_ok; then
      ok "forward-compat active: $COMPAT_DIR on driver CUDA $DRV_CUDA — torch sees the GPU"
      # Persist for every later step in this run (run_model.sh, supervisor.sh source common.sh).
      echo "export LD_LIBRARY_PATH=\"$COMPAT_DIR\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"" > "$EXP_ROOT/.cuda_compat.env"
      state "doctor: cuda forward-compat enabled ($COMPAT_DIR)"
    else
      fail "installed $COMPAT_DIR but torch still cannot use the GPU on driver $DRV_CUDA"
      echo "         This driver branch may not support forward-compat to $MIN_CUDA. Options: a host with"
      echo "         driver >= 570 (ask RunPod which region/template), or MIN_CUDA=12.4 with an older stack"
      echo "         (cannot load gemma-4)."
      state "doctor: driver CUDA $DRV_CUDA, forward-compat failed"
      echo; echo "=========================================================="; echo " DRIVER TOO OLD — STOPPING BEFORE ANY OTHER CHECK "; echo "=========================================================="
      exit 1
    fi
  else
    fail "driver CUDA $DRV_CUDA < $MIN_CUDA and cuda-compat-$COMPAT_VER could not be installed"
    echo "         Need a host with driver >= 570 (CUDA 12.8+). The template's CUDA version does NOT change the driver."
    state "doctor: driver CUDA $DRV_CUDA, no compat package"
    echo; echo "=========================================================="; echo " DRIVER TOO OLD — STOPPING BEFORE ANY OTHER CHECK "; echo "=========================================================="
    exit 1
  fi
else
  ok "driver supports CUDA $DRV_CUDA (>= $MIN_CUDA)"
fi

echo "== doctor: credentials =="
# common.sh already resolved/normalised these or exited.
ok ".env resolved; judge key present (${OPENAI_API_KEY:0:8}...), base URL $OPENAI_BASE_URL"
case "$OPENAI_BASE_URL" in
  *openrouter.ai*) ok "judge routed via OpenRouter (no OpenAI credits needed)" ;;
  *) fail "OPENAI_BASE_URL is not OpenRouter: $OPENAI_BASE_URL" ;;
esac
resp=$(curl -sS --max-time 30 "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
  -d "{\"model\":\"$JUDGE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"say ok\"}],\"max_tokens\":5}" 2>&1)
if echo "$resp" | grep -q '"choices"'; then ok "live judge call to $JUDGE_MODEL succeeded"
else fail "judge call failed: $(echo "$resp" | head -c 300)"; fi

echo "== doctor: HF downloaders =="
# A host image can enable accelerated downloads that a clean venv cannot satisfy.
for pkg_var in "hf_transfer:HF_HUB_ENABLE_HF_TRANSFER" "hf_xet:HF_HUB_ENABLE_XET"; do
  pkg="${pkg_var%%:*}"; var="${pkg_var##*:}"
  if [[ "${!var:-0}" == "1" ]]; then
    if (cd "$AXIS_DIR" && uv run python -c "import $pkg" 2>/dev/null); then
      ok "$var=1 and $pkg importable"
    else
      (cd "$AXIS_DIR" && uv pip install -q "$pkg") >/dev/null 2>&1 || true
      if (cd "$AXIS_DIR" && uv run python -c "import $pkg" 2>/dev/null); then
        ok "$var=1; installed missing $pkg"
      else
        export "$var=0"; ok "$var could not be satisfied -> disabled it (downloads will still work)"
      fi
    fi
  else
    ok "$var not enabled"
  fi
done

echo "== doctor: models reachable + loadable =="
for m in $DOCTOR_MODELS; do
  out=$(cd "$AXIS_DIR" && uv run python - "$m" <<'PY' 2>&1
import sys
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer
m = sys.argv[1]
hf_hub_download(m, "config.json")                       # auth + network + downloader
cfg = AutoConfig.from_pretrained(m)                      # architecture supported by installed transformers
n = getattr(cfg, "num_hidden_layers", None) or getattr(getattr(cfg, "text_config", None), "num_hidden_layers", None)
tok = AutoTokenizer.from_pretrained(m)                   # tokenizer (custom vocab / added tokens)
print(f"OK layers={n} vocab={len(tok)} type={cfg.model_type}")
PY
)
  if [[ "$out" == OK* ]]; then ok "$m — $out"
  else fail "$m — $(echo "$out" | tail -3 | tr '\n' ' ')"; fi
done

echo "== doctor: persona delivery =="
for m in $DOCTOR_MODELS; do
  if (cd "$AXIS_DIR" && uv run python "$SCRIPTS_DIR/check_chat_template.py" --model "$m" >/dev/null 2>&1); then
    ok "$m — persona instructions reach the model"
  else fail "$m — persona instructions would be dropped"; fi
done

echo "== doctor: hardware + disk =="
NGPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if (cd "$AXIS_DIR" && uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null); then
  ok "torch sees $NGPU GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd', ' -)"
else fail "torch cannot use the GPU (driver/CUDA mismatch: $(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | head -1))"; fi
# Never assume 2 GPUs: a single-GPU pod crashes vLLM with NVMLError_InvalidArgument because
# CUDA_VISIBLE_DEVICES=1 points at a device that does not exist. State the plan explicitly.
if [[ "$NGPU" -ge 2 ]]; then ok "execution plan: 2 models in PARALLEL, one per GPU"
elif [[ "$NGPU" -eq 1 ]]; then ok "execution plan: 1 GPU -> 2 models SEQUENTIALLY on GPU 0 (~2x wall-clock)"
else fail "no GPUs visible"; fi
avail=$(df -BG --output=avail /workspace 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -n "$avail" && "$avail" -ge "$MIN_DISK_GB" ]]; then ok "${avail}GB free on /workspace (need ~${MIN_DISK_GB}GB)"
else fail "only ${avail:-?}GB free on /workspace, need ~${MIN_DISK_GB}GB"; fi

echo
if [[ "$FAILED" -eq 0 ]]; then
  echo "=========================================================="
  echo " ALL ENVIRONMENT CHECKS PASSED — SAFE TO LEAVE IT RUNNING "
  echo "=========================================================="
  state "doctor: all checks passed"
  exit 0
fi
echo "=========================================================="
echo " $FAILED CHECK(S) FAILED — FIX BEFORE WALKING AWAY        "
echo "=========================================================="
state "doctor: $FAILED check(s) failed"
exit 1
