# GemmaAssistantAxis

Computing the **Assistant Axis** (Lu et al. 2026, [arXiv:2601.10387](https://arxiv.org/abs/2601.10387)) for the newest Gemma generations — `google/gemma-3-27b-it` and `google/gemma-4-31B-it` (dense) — which postdate the paper (it used Gemma 2 27B, Qwen3 32B, Llama 3.3 70B).

Companion track to [SynthenticPersonaPretrainingSprint](https://github.com/timf34/SynthenticPersonaPretrainingSprint), which runs the same pipeline on 3B synthetic-persona-pretrained models. Both tracks use the identical 60-persona subset (`roles_60.json`) and identical scale settings so per-model metrics are directly comparable; final cross-track comparison is assembled from the two tracks' `COMPARISON.md` files.

## Layout

- `assistant-axis/` — vendored pipeline from [safety-research/assistant-axis](https://github.com/safety-research/assistant-axis) (provenance: `assistant-axis/UPSTREAM.md`; paper: `assistant-axis/og-paper.md`)
- `roles_60.json` — canonical 60-persona subset (20 helpers / 20 humans / 20 non-human), shared across tracks
- `runpod_prompt.md` — the prompt to paste into Claude Code on the RunPod pod; its header lists pod requirements (2× 80GB or 1× H200, `.env` from `.env.example`)

## Run

1. Rent the pod, clone this repo to `/workspace`, scp your filled-in `.env` to `assistant-axis/.env`.
2. Non-root user + tmux + `claude --dangerously-skip-permissions`.
3. Paste everything below the line in `runpod_prompt.md`.

Results land in `/workspace/exp/` and are backed up to the private HF dataset `timf34/gemma-assistant-axis-results` as each model completes.
