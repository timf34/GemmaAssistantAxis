#!/usr/bin/env bash
# Unblock gemma-4-31B-it, then PROVE it works before any GPU hours are spent.
#
# Why this exists: the first attempt died because the venv had transformers 4.57.5 while
# gemma-4's config declares model_type "gemma4" (transformers 5.x). The model card says
# "You can use all Gemma 4 models with the latest version of Transformers", so upgrade —
# but vLLM pins transformers versions, so the upgrade must be verified end to end, not assumed.
#
# Usage:  bash scripts/upgrade_for_gemma4.sh
# Then:   MODELS_ONLY=gemma-4-31b bash run_on_pod.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
AXIS_DIR="$(pwd)/assistant-axis"
MODEL="${GEMMA4_MODEL:-google/gemma-4-31B-it}"

echo "== gemma-4 unblock: $MODEL =="
# Load credentials WITHOUT sourcing common.sh: common.sh calls `exit 1` when .env is missing, and
# because `source` runs in this shell that would kill this script — silently, if stderr were hidden.
# (That exact combination once made this script return instantly with no output at all.)
for f in .env assistant-axis/.env; do
  [[ -f "$f" ]] && { set -a; source "$f"; set +a; echo "  loaded $f"; }
done
: "${HF_TOKEN:=${HUGGING_FACE_HUB_TOKEN:-}}"
export HF_TOKEN
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "!! no HF_TOKEN found in .env — gemma-4 is licence-gated and the download will 403."
  echo "   Put HF_TOKEN=hf_... in assistant-axis/.env (see .env.example) and re-run."
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "!! uv not found — run 'bash run_on_pod.sh' once (or install uv) before this script."
  exit 1
fi
if [[ ! -d "$AXIS_DIR/.venv" ]]; then
  echo "  no venv yet -> running uv sync first (a few minutes)"
  (cd "$AXIS_DIR" && uv sync) || { echo "!! uv sync failed"; exit 1; }
fi

echo "== current versions =="
(cd "$AXIS_DIR" && uv run python -c "
import transformers; print('transformers', transformers.__version__)
try:
    import vllm; print('vllm', vllm.__version__)
except Exception as e: print('vllm import failed:', e)")

echo
echo "== upgrading transformers (and vLLM, which pins it) =="
(cd "$AXIS_DIR" && uv pip install -U transformers) || { echo "!! transformers upgrade failed"; exit 1; }
if [[ "${UPGRADE_VLLM:-1}" == "1" ]]; then
  (cd "$AXIS_DIR" && uv pip install -U vllm) || echo "  (vllm upgrade failed — continuing; may still work)"
fi

echo
echo "== new versions =="
(cd "$AXIS_DIR" && uv run python -c "
import transformers; print('transformers', transformers.__version__)
try:
    import vllm; print('vllm', vllm.__version__)
except Exception as e: print('vllm import failed:', e)")

echo
echo "== verify: config + tokenizer load (this is what failed before) =="
if ! (cd "$AXIS_DIR" && uv run python - "$MODEL" <<'PY'
import sys
from transformers import AutoConfig, AutoTokenizer
m = sys.argv[1]
cfg = AutoConfig.from_pretrained(m)
n = getattr(cfg, "num_hidden_layers", None) or getattr(getattr(cfg, "text_config", None), "num_hidden_layers", None)
tok = AutoTokenizer.from_pretrained(m)
print(f"OK model_type={cfg.model_type} layers={n} vocab={len(tok)}")
PY
); then
  echo "!! still cannot load $MODEL — do NOT rent GPU time for this yet."
  exit 1
fi

echo
echo "== verify: vLLM can actually serve it (the real test — loads weights, needs the GPU) =="
if [[ "${SKIP_VLLM_TEST:-0}" == "1" ]]; then
  echo "  skipped (SKIP_VLLM_TEST=1)"
else
  if ! (cd "$AXIS_DIR" && uv run python - "$MODEL" <<'PY'
import sys
from vllm import LLM, SamplingParams
m = sys.argv[1]
llm = LLM(model=m, max_model_len=2048, gpu_memory_utilization=0.90, trust_remote_code=True)
out = llm.generate(["Say ok."], SamplingParams(max_tokens=8))
print("VLLM OK:", out[0].outputs[0].text.strip()[:40])
PY
  ); then
    echo "!! vLLM cannot serve $MODEL (likely a vllm/transformers version conflict)."
    echo "   Options: pin a vllm nightly with gemma4 support, or run generation via transformers."
    exit 1
  fi
fi

echo
echo "=========================================================="
echo " GEMMA 4 UNBLOCKED — run: MODELS_ONLY=gemma-4-31b bash run_on_pod.sh"
echo "=========================================================="
