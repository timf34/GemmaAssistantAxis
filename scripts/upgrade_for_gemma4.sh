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
# Check the HOST driver before touching a single package. vLLM >= 0.19 wheels are built for CUDA
# 12.8/12.9/13.0 only; on a 12.4 driver the new torch cannot import (libcusparseLt.so.0). That is a
# host property — no pip install fixes it — so refuse immediately and say which pod to rent.
source "$(dirname "${BASH_SOURCE[0]}")/cuda_compat.sh"
DRV_CUDA=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
# Required CUDA comes from the installed torch, not a constant — see scripts/cuda_compat.sh.
MIN_CUDA="${MIN_CUDA:-$(cuda_compat_torch_cuda "$AXIS_DIR")}"; MIN_CUDA="${MIN_CUDA:-12.8}"
if [[ -z "$DRV_CUDA" ]]; then
  echo "!! nvidia-smi shows no driver CUDA version — is this a GPU pod?"; exit 1
fi
if [[ "$(printf '%s\n%s\n' "$MIN_CUDA" "$DRV_CUDA" | sort -V | head -1)" != "$MIN_CUDA" ]]; then
  echo "  driver CUDA $DRV_CUDA < $MIN_CUDA — will rely on NVIDIA forward-compat ($(cuda_compat_pkg "$MIN_CUDA"))."
  echo "  The doctor (run_on_pod.sh) installs and verifies it; this script only checks the package installs."
  COMPAT_DIR="$(cuda_compat_find_dir "$MIN_CUDA")"
  if [[ -z "$COMPAT_DIR" ]]; then
    cuda_compat_install "$MIN_CUDA" || true
    COMPAT_DIR="$(cuda_compat_find_dir "$MIN_CUDA")"
  fi
  if [[ -n "$COMPAT_DIR" ]]; then
    export LD_LIBRARY_PATH="$COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "  forward-compat libs present at $COMPAT_DIR (LD_LIBRARY_PATH set for this script)"
  else
    echo "=========================================================================="
    echo " driver CUDA $DRV_CUDA < $MIN_CUDA and $(cuda_compat_pkg "$MIN_CUDA") could not be installed."
    echo " Need a host with driver >= 570. The template's CUDA version does NOT change the driver."
    echo "=========================================================================="
    exit 1
  fi
else
  echo "  driver CUDA $DRV_CUDA >= $MIN_CUDA — ok"
fi
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
echo "== upgrading vLLM + transformers TOGETHER =="
# Resolve both in ONE command. Installing them separately fails: vLLM <0.19 pins transformers<5,
# so a later `uv pip install -U vllm` silently DOWNGRADES transformers back below the gemma4
# threshold (observed: transformers 5.x -> 4.57.5, vllm 0.13.0, gemma4 still unloadable).
# vLLM 0.19.1+ ships day-0 Gemma 4 support and itself requires transformers>=5.5.3.
VLLM_FLOOR="${VLLM_FLOOR:-0.19.1}"
TRANSFORMERS_FLOOR="${TRANSFORMERS_FLOOR:-5.5.3}"
# CEILING, not optional: transformers 5.15.0 makes Gemma4Config heterogeneous, and vLLM 0.27.1
# reads config.head_dim directly -> AmbiguousGlobalPerLayerAttributeError before any weights load.
# Without this cap `-U` pulls 5.15.0 and the serve test below fails. See pyproject.toml.
TRANSFORMERS_CEIL="${TRANSFORMERS_CEIL:-5.15}"
echo "  targeting vllm>=$VLLM_FLOOR and transformers>=$TRANSFORMERS_FLOOR,<$TRANSFORMERS_CEIL"
if ! (cd "$AXIS_DIR" && uv pip install -U "vllm>=$VLLM_FLOOR" "transformers>=$TRANSFORMERS_FLOOR,<$TRANSFORMERS_CEIL"); then
  echo "!! could not resolve vllm>=$VLLM_FLOOR with transformers>=$TRANSFORMERS_FLOOR."
  echo "   Most likely a torch/CUDA constraint on this image. Check the resolver output above."
  echo "   Fallbacks: newer base image (CUDA 12.8+), or VLLM_FLOOR=<other> to try another release."
  exit 1
fi

echo "== new versions =="
(cd "$AXIS_DIR" && uv run python -c "
import transformers; print('transformers', transformers.__version__)
try:
    import vllm; print('vllm', vllm.__version__)
except Exception as e: print('vllm import failed:', e)")

echo
# Guard: the whole failure mode here is "the install reported success but versions moved back".
if ! (cd "$AXIS_DIR" && uv run python - "$VLLM_FLOOR" "$TRANSFORMERS_FLOOR" <<'PY'
import sys
from packaging.version import Version
import transformers, vllm
tf_ok = Version(transformers.__version__) >= Version(sys.argv[2])
vl_ok = Version(vllm.__version__) >= Version(sys.argv[1])
print(f"transformers {transformers.__version__} {'>=' if tf_ok else '<'} {sys.argv[2]}")
print(f"vllm         {vllm.__version__} {'>=' if vl_ok else '<'} {sys.argv[1]}")
sys.exit(0 if (tf_ok and vl_ok) else 1)
PY
); then
  echo "!! versions did not land above the required floors — gemma4 will not load. Stopping."
  exit 1
fi

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

# APPLY THE CHAT TEMPLATE. This test used to call llm.generate() on the raw string "Say ok." and
# print whatever came back. On gemma-4 that returns "ok. ok. ok. ok. ..." forever — an untemplated
# instruct model just continues the text — and the test happily printed "VLLM OK: ok. ok. ok." and
# exited 0. It green-lit an overnight run on output that would have been garbage. The pipeline
# templates every prompt (generation.py:227), so the test must too, and must CHECK the result:
# "it returned a string" is not the same as "it generated language".
m = sys.argv[1]
llm = LLM(model=m, max_model_len=2048, gpu_memory_utilization=0.90, trust_remote_code=True)
tok = llm.get_tokenizer()
prompts = [
    tok.apply_chat_template([{"role": "user", "content": q}],
                            tokenize=False, add_generation_prompt=True)
    for q in ("Say ok.", "What is the capital of France? Answer in one sentence.")
]
outs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.9, max_tokens=64))
texts = [o.outputs[0].text.strip() for o in outs]
for t in texts:
    print("VLLM OUT:", repr(t[:120]))
# Degeneracy guard: a repetition loop collapses to a handful of distinct tokens.
for t in texts:
    if not t:
        print("!! empty generation"); sys.exit(1)
    w = t.split()
    if len(w) >= 6 and len(set(w)) / len(w) < 0.25:
        print(f"!! degenerate output: only {len(set(w))}/{len(w)} distinct tokens"); sys.exit(1)
if "paris" not in texts[1].lower():
    print(f"!! model failed a basic factual prompt; got: {texts[1][:120]!r}"); sys.exit(1)
print("VLLM OK: chat-templated generation is coherent")
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
