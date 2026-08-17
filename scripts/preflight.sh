#!/usr/bin/env bash
# End-to-end preflight. Run before the overnight launch; aborts loudly on any failure.
# Also pre-downloads both models (the slowest part, ~120GB total).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODELS=("google/gemma-3-27b-it" "google/gemma-4-31B-it")
# Map both models onto whatever GPUs exist (a 1-GPU pod crashes vLLM otherwise).
NGPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
[[ "$NGPU" -ge 1 ]] || { echo "FATAL: no GPUs visible" >&2; exit 1; }
GPUS=(0 $(( NGPU >= 2 ? 1 : 0 )))
KEYS=("gemma-3-27b" "gemma-4-31b")
# Honour MODELS_ONLY the same way run_on_pod.sh's launch() does. Without this, a
# MODELS_ONLY=gemma-4-31b run still pre-downloaded gemma-3 (~54GB) and put it through a full
# generate/activations/judge cycle -- GPU hours and judge spend for a model this run will not touch.
INDICES=(0 1)
if [[ -n "${MODELS_ONLY:-}" ]]; then
  INDICES=()
  for i in 0 1; do [[ "${KEYS[$i]}" == "$MODELS_ONLY" ]] && INDICES+=("$i"); done
  if [[ ${#INDICES[@]} -eq 0 ]]; then
    echo "FATAL: MODELS_ONLY='$MODELS_ONLY' matches none of: ${KEYS[*]}" >&2; exit 1
  fi
  echo "MODELS_ONLY=$MODELS_ONLY -> preflighting only: ${KEYS[${INDICES[0]}]}"
fi

echo "== 1/4 OpenRouter judge check =="
resp=$(curl -sS "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
  -d "{\"model\":\"$JUDGE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ok\"}],\"max_tokens\":5}")
echo "$resp" | grep -q '"choices"' || { echo "FATAL: OpenRouter test call failed: $resp" >&2; exit 1; }
echo "ok"

echo "== 2/4 GPU / disk =="
# Require 1 GPU, not 2. This check used to hard-fail with "need 2 GPUs" and killed the whole
# overnight launch on a healthy 1xH200 pod -- AFTER the doctor had already printed "ALL ENVIRONMENT
# CHECKS PASSED". It contradicted every other part of the repo: line 11 above already maps both
# models onto whatever GPUs exist precisely so a 1-GPU pod works, doctor.sh reports the 1-GPU plan
# as ok ("2 models SEQUENTIALLY on GPU 0"), and run_on_pod.sh has an explicit PARALLEL=0 branch for
# it. Leftover from when the design assumed a 2-GPU pod. One GPU is slower, not invalid.
[[ "$NGPU" -ge 1 ]] || { echo "FATAL: no GPUs visible" >&2; exit 1; }
[[ "$NGPU" -ge 2 ]] || echo "  note: 1 GPU — models run sequentially (~2x wall-clock), not in parallel"
df -h /workspace
avail_gb=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
[[ "$avail_gb" -ge 600 ]] || echo "WARNING: <600GB free — full run needs ~450GB activations + ~120GB models. Consider a bigger volume or LAYER subset."

echo "== 3/4 HF access (license-gated repos) =="
for i in "${INDICES[@]}"; do
  m="${MODELS[$i]}"
  uv run --project "$AXIS_DIR" python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('$m', 'config.json')
print('access ok: $m')" || { echo "FATAL: cannot access $m — accept the Gemma license on huggingface.co with the HF_TOKEN account" >&2; exit 1; }
done

echo "== 4/4 Tiny end-to-end per model (pirate + default, 3 questions) =="
for i in "${INDICES[@]}"; do
  m="${MODELS[$i]}"; key="${KEYS[$i]}"; gpu="${GPUS[$i]}"
  out="$EXP_ROOT/preflight/$key"
  echo "--- $m on GPU $gpu ---"
  uv run --project "$AXIS_DIR" python -c "
import sys; sys.path.insert(0, '$AXIS_DIR')
from assistant_axis.models import get_config
c = get_config('$m'); print('inferred config:', c)
assert c['total_layers'] > 20, 'layer count looks wrong (text_config issue?)'"
  CUDA_VISIBLE_DEVICES=$gpu uv run --project "$AXIS_DIR" python "$AXIS_DIR/pipeline/1_generate.py" \
    --model "$m" --roles pirate default --question_count 3 \
    --roles_dir "$AXIS_DIR/data/roles/instructions" --questions_file "$AXIS_DIR/data/extraction_questions.jsonl" \
    --output_dir "$out/responses"
  CUDA_VISIBLE_DEVICES=$gpu uv run --project "$AXIS_DIR" python "$AXIS_DIR/pipeline/2_activations.py" \
    --model "$m" --responses_dir "$out/responses" --output_dir "$out/activations" --batch_size 4
  uv run --project "$AXIS_DIR" python "$AXIS_DIR/pipeline/3_judge.py" \
    --responses_dir "$out/responses" --output_dir "$out/scores" --judge_model "$JUDGE_MODEL"
  uv run --project "$AXIS_DIR" python - <<EOF
import torch, json, pathlib
acts = torch.load(sorted(pathlib.Path('$out/activations').glob('*.pt'))[0], weights_only=False)
first = next(iter(acts.values())) if isinstance(acts, dict) else acts[0]
print('activation entry shape:', tuple(first.shape))
scores = json.load(open(sorted(pathlib.Path('$out/scores').glob('*.json'))[0]))
print('sample scores:', list(scores.items())[:3] if isinstance(scores, dict) else scores[:3])
EOF
  state "preflight OK: $m (GPU $gpu)"
done
echo "PREFLIGHT PASSED"
