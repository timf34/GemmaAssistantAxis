#!/usr/bin/env bash
# Full pipeline for ONE model on ONE GPU. Idempotent: every step skips existing outputs,
# so re-running after a crash resumes where it left off.
# Usage: run_model.sh <hf_model_id> <gpu_id> <short_key>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="$1"; GPU="$2"; KEY="$3"
OUT="$EXP_ROOT/$KEY"
mkdir -p "$OUT/logs"
export CUDA_VISIBLE_DEVICES="$GPU"
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

run_step() {  # run_step <name> <timeout> <cmd...>
  local name="$1" tmo="$2"; shift 2
  state "[$KEY] $name: start"
  timeout "$tmo" "$@" >> "$OUT/logs/$name.log" 2>&1
  state "[$KEY] $name: done"
}

# 1. Generate — all 275 roles + default (no --roles flag = full set), QUESTION_COUNT questions x 5 prompts
run_step generate 14h uv run --project "$AXIS_DIR" python "$AXIS_DIR/pipeline/1_generate.py" \
  --model "$MODEL" --question_count "$QUESTION_COUNT" \
  --roles_dir "$AXIS_DIR/data/roles/instructions" --questions_file "$AXIS_DIR/data/extraction_questions.jsonl" \
  --output_dir "$OUT/responses"

# 2+3. Activations (GPU) and judge (API) in parallel
run_step activations 12h uv run --project "$AXIS_DIR" python "$AXIS_DIR/pipeline/2_activations.py" \
  --model "$MODEL" --responses_dir "$OUT/responses" --output_dir "$OUT/activations" --batch_size "$BATCH_SIZE" &
ACT_PID=$!
run_step judge 12h uv run --project "$AXIS_DIR" python "$AXIS_DIR/pipeline/3_judge.py" \
  --responses_dir "$OUT/responses" --output_dir "$OUT/scores" --judge_model "$JUDGE_MODEL" \
  --roles_dir "$AXIS_DIR/data/roles/instructions" --batch_size "$JUDGE_BATCH" --requests_per_second "$JUDGE_RPS" &
JUDGE_PID=$!
wait "$ACT_PID"; wait "$JUDGE_PID"

# Judge cleanup pass (repo README: rerun once to catch malformed/rate-limited responses)
run_step judge_rerun 4h uv run --project "$AXIS_DIR" python "$AXIS_DIR/pipeline/3_judge.py" \
  --responses_dir "$OUT/responses" --output_dir "$OUT/scores" --judge_model "$JUDGE_MODEL" \
  --roles_dir "$AXIS_DIR/data/roles/instructions" --batch_size "$JUDGE_BATCH" --requests_per_second "$JUDGE_RPS"

# 4. Per-role vectors from fully-role-playing (score=3) responses only — keep the judge filter
run_step vectors 2h uv run --project "$AXIS_DIR" python "$AXIS_DIR/pipeline/4_vectors.py" \
  --activations_dir "$OUT/activations" --scores_dir "$OUT/scores" --output_dir "$OUT/vectors" --min_count "$MIN_COUNT"

# 5. Axis = mean(default) - mean(role vectors)
run_step axis 1h uv run --project "$AXIS_DIR" python "$AXIS_DIR/pipeline/5_axis.py" \
  --vectors_dir "$OUT/vectors" --output "$OUT/axis.pt"

# 6. Package into the release layout: <key>/assistant_axis.pt, <key>/default_vector.pt, <key>/role_vectors/<role>.pt
run_step package 1h uv run --project "$AXIS_DIR" python "$SCRIPTS_DIR/package_release.py" \
  --vectors_dir "$OUT/vectors" --axis "$OUT/axis.pt" --out "$OUT/release/$KEY"

# 7. Analysis: PCA metrics, plots, RESULTS.md, integrity check cos(default-mean(roles), axis)=1
run_step analyze 2h uv run --project "$AXIS_DIR" python "$SCRIPTS_DIR/analyze_axis.py" \
  --release "$OUT/release/$KEY" --key "$KEY" --outdir "$OUT" --roles90 "$REPO_DIR/roles_90.json"

# 8. Backup off the ephemeral pod (release + reports + scores; NOT raw activations/responses)
run_step upload 2h uv run --project "$AXIS_DIR" python "$SCRIPTS_DIR/upload_results.py" \
  --exp "$OUT" --key "$KEY" --repo "$HF_RESULTS_REPO"

# 9. Prune raw activations (~220GB/model at 275 roles) now that vectors are saved and uploaded.
#    Re-running step 4 with a different --min_count after this requires re-extracting activations.
if [[ "$PRUNE_ACTIVATIONS" == "1" && -d "$OUT/release/$KEY" ]]; then
  state "[$KEY] pruning activations"
  rm -rf "$OUT/activations"
fi

state "[$KEY] COMPLETE"
