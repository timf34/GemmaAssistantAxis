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

**Pod:** 2× H200 141GB (ideal — one model per card, no tensor parallelism) or 2× H100/A100 80GB. **1TB volume** at `/workspace`: raw activations run ~220GB per model at full scale (pruned automatically after upload; set `PRUNE_ACTIVATIONS=0` to keep them).

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

## What it produces

Per model, in the same layout as the paper's HF release (`lu-christina/assistant-axis-vectors`), so anything written against that release works unchanged:

```
<key>/assistant_axis.pt        (n_layers, hidden)
<key>/default_vector.pt        (n_layers, hidden)
<key>/role_vectors/<role>.pt   (n_layers, hidden)   — all 275 roles that pass the judge filter
```

Plus `RESULTS.md` (PC1↔axis cosine, variance explained, default separation, ranked role projections, layer sweep), `summary.json`, three plots, and a top-level `COMPARISON.md` across Gemma 2/3/4. Everything except raw activations/responses is uploaded to the private HF dataset `timf34/gemma-assistant-axis-results` as each model finishes.

**Integrity check:** every run reports `cos(default − mean(saved role vectors), axis)`, which must be **1.000** — it proves the axis was built from exactly the vectors that were saved (the same check that validates against the paper's Gemma 2 release).

## Monitoring

`/workspace/exp/STATE.md` is a timestamped step-by-step log; `/workspace/exp/attempts.log` records any retries. Optionally run Claude Code on the pod as a *monitor* (not a driver) using `claude_monitor_prompt.md`.

## Layout

- `run_on_pod.sh` — the one-shot entry point
- `scripts/` — `preflight.sh`, `supervisor.sh` (retry loop), `run_model.sh` (5-step pipeline + packaging), `package_release.py`, `analyze_axis.py`, `upload_results.py`
- `assistant-axis/` — vendored pipeline ([safety-research/assistant-axis](https://github.com/safety-research/assistant-axis); provenance in `assistant-axis/UPSTREAM.md`, paper in `assistant-axis/og-paper.md`)
- `roles_90.json` — 90-persona subset shared with the SPP track
