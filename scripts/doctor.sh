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

DOCTOR_MODELS="${DOCTOR_MODELS:-google/gemma-3-27b-it google/gemma-4-31B-it}"
MIN_DISK_GB="${MIN_DISK_GB:-600}"
FAILED=0
ok()   { echo "  [ok]   $*"; }
fail() { echo "  [FAIL] $*"; FAILED=$((FAILED+1)); }

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
if (cd "$AXIS_DIR" && uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null); then
  ok "torch sees $(nvidia-smi --list-gpus | wc -l) GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd', ' -)"
else fail "torch cannot use the GPU (driver/CUDA mismatch: $(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | head -1))"; fi
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
