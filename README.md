# GemmaAssistantAxis

Computing the **Assistant Axis** (Lu et al. 2026, [arXiv:2601.10387](https://arxiv.org/abs/2601.10387)) for the newest Gemma generations — `google/gemma-3-27b-it` and `google/gemma-4-31B-it` (dense) — which postdate the paper (it used Gemma 2 27B, Qwen3 32B, Llama 3.3 70B).

Because the paper published its **Gemma 2 27B** vectors, running the same 275-role protocol gives a clean **three-generation comparison**: Gemma 2 → 3 → 4.

Companion track to [SynthenticPersonaPretrainingSprint](https://github.com/timf34/SynthenticPersonaPretrainingSprint) (same pipeline on 3B synthetic-persona-pretrained models); `roles_90.json` is the 90-persona subset shared with that track, and analysis reports subset metrics alongside the full-275 ones so the two are comparable.

## Run it

```bash
git clone https://github.com/timf34/GemmaAssistantAxis.git && cd GemmaAssistantAxis
# scp your filled-in .env to assistant-axis/.env  (see .env.example)
bash run_on_pod.sh
```

That's the whole thing: it installs deps, preflights (OpenRouter judge call, GPU/disk, HF license access, tiny end-to-end per model), then runs **both models in parallel, one per GPU**, each wrapped in a retry supervisor, and finally builds the cross-generation comparison including the paper's Gemma 2 vectors.

**Pod:** 2× H200 141GB (ideal — one model per card, no tensor parallelism) or 2× H100/A100 80GB. **1TB volume** at `/workspace`: raw activations run ~220GB per model at full scale and are kept + uploaded to a separate public dataset by default (`PRUNE_ACTIVATIONS=1` to delete after a confirmed upload).

Useful overrides:

| var | default | meaning |
|---|---|---|
| `QUESTION_COUNT` | 120 | 600 responses/role. Use 60 for a short night, 240 for full paper scale |
| `MIN_COUNT` | 25 | min fully-role-playing responses per role vector (scale with QUESTION_COUNT) |
| `MODELS_ONLY` | – | `gemma-3-27b` or `gemma-4-31b` to run just one |
| `DOCTOR_ONLY` | 0 | run the cheap environment checks (~3 min) and exit — validate a fresh pod before committing to a night |
| `SKIP_PREFLIGHT` | 0 | resume after a crash |
| `SHUTDOWN` | – | `stop` pauses the pod when done (billing stops); `terminate` destroys it (auto-downgraded to `stop` if anything failed) |
| `SAVE_TO_GIT` | 0 | push reports/plots (not the .pt vectors) to this repo before shutdown; needs a PAT in the git remote |

## Gemma 4

**Driver requirement: CUDA >= 12.8, and the DRIVER is a host property.** Gemma 4 needs vLLM >= 0.19 /
transformers >= 5.5, whose wheels are built for CUDA 12.8+. RunPod's H200 hosts have shipped **driver 550
(CUDA 12.4)** across three different templates — including the `cu1281` one — because the template's CUDA
version is the *container toolkit* and cannot raise the *host driver*. Check with `nvidia-smi | grep "CUDA Version"`.

On an old driver the doctor now automatically tries **NVIDIA forward-compatibility** (`cuda-compat-12-8`,
newer user-space `libcuda` under `/usr/local/cuda-12.8/compat`; supported on datacenter GPUs like H100/H200/A100)
and verifies torch can see the GPU through it. If that works, the run proceeds on the same pod. If it does
not, the only fix is a host with driver >= 570 — ask RunPod which region/template has it — or an older
vllm/torch stack, which cannot load Gemma 4.

```bash
bash scripts/upgrade_for_gemma4.sh          # upgrades, then loads config+tokenizer AND serves it via vLLM
MODELS_ONLY=gemma-4-31b bash run_on_pod.sh  # only if the above prints GEMMA 4 UNBLOCKED
```

**GPU:** 30.7B dense ≈ 62GB in bf16. A single **H200 141GB** is the comfortable, fastest choice; an H100 80GB works with less KV-cache headroom; an A100 80GB fits but runs roughly 2–3× slower (Ampere, no FP8) and is likelier to carry a driver too old for a current vLLM.

## Storage policy: everything public, nothing only-on-the-pod

Results go to the **public** HF dataset `timf34/gemma-assistant-axis-results` (`HF_PRIVATE=1` to override). Private HF repos share a
small LFS storage quota, and a 403 from that quota is what stopped an earlier run's `.pt` tensors from
ever being backed up. What is preserved per model:

| artifact | where | how |
|---|---|---|
| vectors: `assistant_axis.pt`, `default_vector.pt`, `role_vectors/*.pt` | HF | automatic (`upload_results.py`) |
| judge scores, reports, plots, `summary.json`, logs | HF + git (`results/`) | automatic |
| raw transcripts joined to scores (`responses_<key>.jsonl.gz`) | HF | **manual**: `scripts/archive_responses.py --upload` |
| raw activations (per-response, all layers) | HF, separate dataset `*-activations` | automatic (`upload_activations.py`, resumable); ~57–220GB/model. Needed for any per-response analysis (the vectors are per-role means). `PRUNE_ACTIVATIONS=1` deletes them after a confirmed upload; never deleted otherwise |

Run the archive step before terminating any pod. `SHUTDOWN=stop` keeps the disk, so a missed archive is recoverable; `terminate` is not.

## Cost discipline: GPUs only for GPU work

Only steps 1 (generate) and 2 (activations) need a GPU. The judge is API traffic and everything after
it is numpy — on a 3B pair the GPU phase is ~45 minutes while the judge can run for hours, so a
naive single-pod run bills expensive cards to sit idle.

Two mitigations are built in:

- **The judge is parallel by default** (`JUDGE_BATCH=200`, `JUDGE_RPS=250`). Lower them if OpenRouter
  returns 429s; raise them if it keeps up.
- **`scripts/finish_cpu.sh` runs everything after activations without a GPU.** Once `STATE.md` shows
  `activations: done`, you can stop the pod, pull the run down, and finish on a laptop:

  ```bash
  rsync -avP root@<POD>:/workspace/exp/gemma-3-27b ./exp/
  EXP_ROOT=./exp bash scripts/finish_cpu.sh gemma-3-27b
  ```

  Note the trade-off: the vectors step needs `activations/`, which is the bulky artifact. Compare the
  download size against the GPU-hours saved before moving — for a small model that finishes in an hour
  it is usually not worth it; for a long judge tail it is.

## Before you walk away

`bash run_on_pod.sh` runs `scripts/doctor.sh` before any GPU work: credentials + a live OpenRouter call, HF downloader flags (auto-installs `hf_transfer`/`hf_xet` or disables them), HF auth, config + tokenizer load for both models, chat-template persona delivery, GPU, and disk. It prints **`ALL ENVIRONMENT CHECKS PASSED — SAFE TO LEAVE IT RUNNING`** when everything is green; until you see that line, stay at the terminal. `DOCTOR_ONLY=1 bash run_on_pod.sh` runs just the checks (~3 min) on a fresh pod.

## What it produces

Per model, in the same layout as the paper's HF release (`lu-christina/assistant-axis-vectors`), so anything written against that release works unchanged:

```
<key>/assistant_axis.pt        (n_layers, hidden)
<key>/default_vector.pt        (n_layers, hidden)
<key>/role_vectors/<role>.pt   (n_layers, hidden)   — all 275 roles that pass the judge filter
```

Plus `RESULTS.md` (PC1↔axis cosine, variance explained, default separation, ranked role projections, layer sweep), `summary.json`, three plots, and a top-level `COMPARISON.md` across Gemma 2/3/4. Everything except raw activations/responses is uploaded to the **public** HF dataset `timf34/gemma-assistant-axis-results` as each model finishes.

**Integrity check:** every run reports `cos(default − mean(saved role vectors), axis)`, which must be **1.000** — it proves the axis was built from exactly the vectors that were saved (the same check that validates against the paper's Gemma 2 release).

## Monitoring

`/workspace/exp/STATE.md` is a timestamped step-by-step log; `/workspace/exp/attempts.log` records any retries. Optionally run Claude Code on the pod as a *monitor* (not a driver) using `claude_monitor_prompt.md`.

## Layout

- `run_on_pod.sh` — the one-shot entry point
- `scripts/` — `doctor.sh` (fail-fast env checks), `preflight.sh`, `supervisor.sh` (retry loop), `run_model.sh` (5-step pipeline + packaging), `package_release.py`, `analyze_axis.py`, `upload_results.py`
- `assistant-axis/` — vendored pipeline ([safety-research/assistant-axis](https://github.com/safety-research/assistant-axis); provenance in `assistant-axis/UPSTREAM.md`, paper in `assistant-axis/og-paper.md`)
- `roles_90.json` — 90-persona subset shared with the SPP track
