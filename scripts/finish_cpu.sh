#!/usr/bin/env bash
# Finish a run WITHOUT a GPU: judge -> vectors -> axis -> package -> analyze -> upload.
#
# Only steps 1 (generate) and 2 (activations) need a GPU. Everything after is API traffic and
# numpy — so it should never run on a rented GPU pod. Stop the pod once activations are done,
# rsync the run directory to a laptop or a cheap CPU box, and finish here.
#
# Typical use, from your laptop:
#   rsync -avP root@<POD>:/workspace/exp/gemma-3-27b  ./exp/       # responses + activations + scores
#   EXP_ROOT=./exp bash scripts/finish_cpu.sh gemma-3-27b
#
# What you actually need locally per model:
#   responses/   (needed only if the judge has not finished)
#   activations/ (needed for the vectors step — this is the bulky one, ~220GB for a 27B model)
#   scores/      (whatever the judge already produced; it resumes)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

KEY="${1:?usage: finish_cpu.sh <key>   e.g. finish_cpu.sh gemma-3-27b}"
OUT="$EXP_ROOT/$KEY"
mkdir -p "$OUT/logs"
[[ -d "$OUT" ]] || { echo "FATAL: $OUT not found"; exit 1; }

EFF_MIN_COUNT="$MIN_COUNT"

step() { local name="$1"; shift; state "[$KEY] (cpu) $name: start"; "$@" >> "$OUT/logs/$name.log" 2>&1; state "[$KEY] (cpu) $name: done"; echo "  $name done"; }

echo "== finishing $KEY on CPU (no GPU required) =="

if [[ -d "$OUT/responses" ]]; then
  echo "  judging (resumes; skips roles already scored)"
  step judge_cpu uv run --project "$AXIS_DIR" python "$AXIS_DIR/pipeline/3_judge.py" \
    --responses_dir "$OUT/responses" --output_dir "$OUT/scores" --judge_model "$JUDGE_MODEL" \
    --roles_dir "$AXIS_DIR/data/roles/instructions" --batch_size "$JUDGE_BATCH" --requests_per_second "$JUDGE_RPS"
else
  echo "  no responses/ locally — assuming scores/ is already complete"
fi

[[ -d "$OUT/activations" ]] || { echo "FATAL: $OUT/activations missing — the vectors step needs it."; exit 1; }

step vectors uv run --project "$AXIS_DIR" python "$AXIS_DIR/pipeline/4_vectors.py" \
  --activations_dir "$OUT/activations" --scores_dir "$OUT/scores" --output_dir "$OUT/vectors" --min_count "$EFF_MIN_COUNT"
step axis uv run --project "$AXIS_DIR" python "$AXIS_DIR/pipeline/5_axis.py" \
  --vectors_dir "$OUT/vectors" --output "$OUT/axis.pt"
step package uv run --project "$AXIS_DIR" python "$SCRIPTS_DIR/package_release.py" \
  --vectors_dir "$OUT/vectors" --axis "$OUT/axis.pt" --out "$OUT/release/$KEY"
step analyze uv run --project "$AXIS_DIR" python "$SCRIPTS_DIR/analyze_axis.py" \
  --release "$OUT/release/$KEY" --key "$KEY" --outdir "$OUT" --roles90 "$REPO_DIR/roles_90.json"

if [[ "${UPLOAD:-1}" == "1" ]]; then
  uv run --project "$AXIS_DIR" python "$SCRIPTS_DIR/upload_results.py" \
    --exp "$OUT" --key "$KEY" --repo "$HF_RESULTS_REPO" || echo "  (upload failed — results are local at $OUT)"
fi

echo "== $KEY done: $OUT/RESULTS.md =="
