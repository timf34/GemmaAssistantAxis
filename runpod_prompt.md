# Prompt for Claude Code (run on the RunPod pod)

Copy everything below the line into a Claude Code session running on the RunPod pod.

Pod assumptions (set up before starting Claude Code):
- **2× A100/H100 80GB (tensor parallel), or a single H200 141GB.** gemma-4-31B-it is ~62GB of bf16 weights — a single 80GB card is too tight to run it well. ≥300GB volume at `/workspace` (two big models + activations).
- `.env` copied (scp, not git) to `/workspace/GemmaAssistantAxis/assistant-axis/.env` — see `.env.example` in this repo. OpenRouter key under `OPENAI_API_KEY` AND `OPENROUTER_API_KEY`, `OPENAI_BASE_URL=https://openrouter.ai/api/v1`, plus `HF_TOKEN` for the license-gated Gemma repos (the account must have accepted the Gemma licenses on huggingface.co — check gemma-3 AND gemma-4 families).
- Claude Code running as a non-root user in tmux (`--dangerously-skip-permissions` refuses root).

---

## Goal

Compute the **Assistant Axis** (Lu et al. 2026, arXiv:2601.10387 — paper at `assistant-axis/og-paper.md`, pipeline vendored in `assistant-axis/`) for the newest Gemma generations, which postdate the paper (it used Gemma **2** 27B):

1. `google/gemma-3-27b-it` — same scale as the paper's Gemma 2 27B, one generation newer. Run this first: known-good scale, lowest risk.
2. `google/gemma-4-31B-it` — the dense Gemma 4 flagship (deliberately NOT the 26B-A4B MoE — untested tooling assumptions).

These are strong, heavily post-trained role-players, so no capability gate is needed — a short smoke test per model, then the full pipeline. Headline result per model: does PC1 of the 90-persona vector space align with the Assistant Axis (paper: cosine >0.71 at the middle layer), with the default assistant at one extreme?

## Setup

1. ```bash
   cd /workspace
   git clone https://github.com/timf34/GemmaAssistantAxis.git
   cd GemmaAssistantAxis/assistant-axis && uv sync
   ```
2. Verify `.env` (fail fast if missing/incomplete) and make a one-off test call to `openai/gpt-4.1-mini` through the OpenRouter base URL. There are NO OpenAI credits — if any judge call ever errors against `api.openai.com`, env loading broke: stop and fix, don't retry. Only the judge scripts auto-load `.env`; start every shell/tmux session with `set -a; source /workspace/GemmaAssistantAxis/assistant-axis/.env; set +a`.
3. Verify HF access: `huggingface-cli download google/gemma-3-27b-it config.json` (and same for gemma-4-31B-it) before any long work — a 403 means the license wasn't accepted; report and stop.
4. All experiment artifacts go in `/workspace/exp/` (`/workspace/exp/<model>/{responses,activations,scores,vectors,logs}`, reports, driver scripts). Keep the vendored `assistant-axis/` clean; any code patch must be minimal, recorded in `assistant-axis/UPSTREAM.md`, and dumped to `/workspace/exp/patches.diff`.

## Pipeline (read the code before changing anything)

5 restartable steps in `assistant-axis/pipeline/` (each skips existing outputs — re-running is always safe):
1. `1_generate.py` (vLLM) → 2. `2_activations.py` (all layers, post-MLP residual) + 3. `3_judge.py` in parallel (**always `--judge_model openai/gpt-4.1-mini`** — the default `gpt-4.1-mini` is not a valid OpenRouter ID) → 4. `4_vectors.py` (score-3 responses only) → 5. `5_axis.py` (`axis = mean(default) − mean(roles)`).

**Personas: use the committed `roles_90.json` at this repo's root** (90 roles + `default`; pass via `--roles`). This exact list is shared with the SPP-3B sprint track for cross-track comparability — do not re-derive or substitute roles.

**Scale: `--question_count 120` (600 responses/role), `--min_count 25`** — matching the SPP track so metrics are comparable.

## Gemma-specific checks (before committing GPU-hours)

- **Multimodal wrappers:** both targets are multimodal (`*ForConditionalGeneration`). Verify vLLM generation AND the transformers-based activation extraction (`assistant_axis/internals/model.py`) hook the **language-model backbone's** decoder layers — hooks on the wrong submodule fail silently. Confirm layer count: `get_config` reads `config.num_hidden_layers`, which for multimodal configs may sit under `text_config`; verify the inferred count matches the actual decoder depth (gemma-3-27b: 62 layers) and that `target_layer` lands mid-stack. Note `MODEL_CONFIGS` only knows gemma-**2**-27b-it; these models take the auto-infer path.
- **Gemma 4 is a new architecture:** the vendored `uv.lock` may pin vllm/transformers versions that predate it. Try loading first; if unsupported, upgrade vllm+transformers (record exact versions in RESULTS.md). If it still fails after 2–3 genuine attempts, finish gemma-3-27b fully and report — don't spend the night on dependency archaeology.
- **Memory:** TP=2 for generation on 2× 80GB (`--tensor_parallel_size 2`); activations extraction with conservative batch size (start 8 for 27B/31B, raise if headroom). `--max_model_len 2048` suffices (2048-token conversations).
- Chat template: Gemma folds system prompts into the first user turn; the pipeline's `format_conversation` detection handles it — confirm which path triggers and record it.

## Protocol

1. **Phase 0 — smoke test per model (~20 min):** 2 roles (`pirate`, plus `default`), `--question_count 10`, full 5-step chain. Eyeball a few responses; include 2–3 in the report.
2. **Phase 1 — full run, gemma-3-27b-it first:** all 90 roles + default. Steps 2+3 in parallel after step 1. Budget estimate: 91 roles × 600 ≈ 55k generations — measure throughput in Phase 0 and project honestly in `/workspace/exp/STATE.md`. Judge ≈ 55k calls/model (~$8–15).
3. **Phase 2 — gemma-4-31B-it:** same, after gemma-3 completes (or in parallel only if you have 4 GPUs).
4. **Analysis per model** (write and test the script during Phase 0, run automatically): PCA on standardized role vectors at the target layer; report PC1↔axis cosine (headline), variance explained (paper: 4–19 PCs for 70%), default's position, top/bottom-10 roles by axis projection (helpers positive, demons/wraiths negative expected), layer sweep of PC1↔axis cosine; plots (scree, PC1-vs-PC2 colored by axis projection, ranked projections). Write `/workspace/exp/<model>/RESULTS.md` and a combined `/workspace/exp/COMPARISON.md` (gemma-3-27b vs gemma-4-31B, per-model metrics only — never compare raw directions across models).
5. **Backup after each model completes:** upload `/workspace/exp/<model>/` (vectors, axis.pt, PCA outputs, plots, reports — NOT raw activations/responses) to a **private** HF dataset `timf34/gemma-assistant-axis-results`.

## Ops

- **Supervisor:** don't launch the driver directly. `supervisor.sh`: up to 8 attempts of `overnight.sh` (idempotent — steps skip existing work); on failure log attempt + exit code + last 20 log lines to `/workspace/exp/attempts.log`, sleep 120s, retry. Generous `timeout` per step (8h generation, 6h activations for these sizes) so hangs become retries. `overnight.sh` appends timestamped step start/finish lines to `/workspace/exp/STATE.md`. A model failing all retries is logged and skipped, never blocking the queue.
- Everything long-running in tmux with per-step logs. Launch, watch ~10 min of throughput/memory, then leave it. End your session with: what was launched, where logs are, the morning checklist.
