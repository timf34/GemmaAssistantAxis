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
DRV_CUDA=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
MIN_CUDA="${MIN_CUDA:-12.8}"
if [[ -z "$DRV_CUDA" ]]; then
  echo "!! nvidia-smi shows no driver CUDA version — is this a GPU pod?"; exit 1
fi
if [[ "$(printf '%s\n%s\n' "$MIN_CUDA" "$DRV_CUDA" | sort -V | head -1)" != "$MIN_CUDA" ]]; then
  echo "  driver CUDA $DRV_CUDA < $MIN_CUDA — will rely on NVIDIA forward-compat (cuda-compat-${COMPAT_VER:-12-8})."
  echo "  The doctor (run_on_pod.sh) installs and verifies it; this script only checks the package installs."
  COMPAT_VER="${COMPAT_VER:-12-8}"; COMPAT_DIR="/usr/local/cuda-${COMPAT_VER/-/.}/compat"
  if [[ ! -d "$COMPAT_DIR" ]]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >/dev/null 2>&1 || true
    if ! apt-get install -y -qq "cuda-compat-$COMPAT_VER" >/dev/null 2>&1; then
      . /etc/os-release; distro="${ID}${VERSION_ID//./}"
      curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb" -o /tmp/cuda-keyring.deb 2>/dev/null \
        && dpkg -i /tmp/cuda-keyring.deb >/dev/null 2>&1 && apt-get update -qq >/dev/null 2>&1 \
        && apt-get install -y -qq "cuda-compat-$COMPAT_VER" >/dev/null 2>&1 || true
    fi
  fi
  if [[ -d "$COMPAT_DIR" ]]; then
    export LD_LIBRARY_PATH="$COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "  forward-compat libs present at $COMPAT_DIR (LD_LIBRARY_PATH set for this script)"
  else
    echo "=========================================================================="
    echo " driver CUDA $DRV_CUDA < $MIN_CUDA and cuda-compat-$COMPAT_VER could not be installed."
    echo " Need a host with driver >= 570. The template's CUDA version does NOT change the driver."
    echo "=========================================================================="
    exit 1
  fi
fi
echo "  driver CUDA $DRV_CUDA >= $MIN_CUDA — ok"
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
echo "  targeting vllm>=$VLLM_FLOOR and transformers>=$TRANSFORMERS_FLOOR"
if ! (cd "$AXIS_DIR" && uv pip install -U "vllm>=$VLLM_FLOOR" "transformers>=$TRANSFORMERS_FLOOR"); then
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
