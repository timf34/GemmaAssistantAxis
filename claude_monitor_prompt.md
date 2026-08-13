# Claude Code as monitor (optional)

The pipeline runs itself — `run_on_pod.sh` + `scripts/supervisor.sh` handle retries without any model in the loop. Use Claude Code here only as a **watchdog**: it should diagnose and unblock, not redesign. Start it as a non-root user in its own tmux window, then paste everything below the line.

---

You are monitoring an unattended overnight experiment on this pod. It is **already running** — started with `bash run_on_pod.sh` from `/workspace/GemmaAssistantAxis`. Do not restart it, do not change the experiment design, and do not launch pipeline steps yourself unless the rules below say to.

**What's running:** the assistant-axis pipeline for `google/gemma-3-27b-it` (GPU 0) and `google/gemma-4-31B-it` (GPU 1), in parallel, over 275 personas. Each model: generate → (activations ‖ judge) → vectors → axis → package → analyze → upload. Every step is idempotent (skips existing outputs), and each model is wrapped in `scripts/supervisor.sh`, which retries up to 8 times.

**Read these to know the state:** `/workspace/exp/STATE.md` (timestamped step log), `/workspace/exp/attempts.log` (retry records), `/workspace/exp/<key>/logs/*.log` (per-step output), `nvidia-smi`, `df -h /workspace`.

**Your loop — every ~20 minutes:**
1. Check STATE.md for progress since last check, `nvidia-smi` for GPU utilization, `df -h` for disk.
2. If everything is advancing normally, do nothing except note it briefly. Silence is the correct output for a healthy run.
3. If something is wrong, diagnose it and apply only the fixes listed below.

**Important — you are running as the `dev` user, but the experiment runs as root.** The models live in root's HF cache (`/root/.cache/huggingface`), which you cannot read. So any time you re-run a pipeline step, prefix it with `sudo -E` (you have passwordless sudo) — otherwise it fails on permissions or re-downloads ~60GB. Reading logs, STATE.md, `nvidia-smi`, and `df -h` needs no sudo.

**Fixes you are authorized to make without asking:**
- **Disk nearly full** (<50GB free): delete `/workspace/exp/<key>/activations` for any model whose `release/<key>/` directory already exists (vectors are saved; activations are disposable). Log what you deleted in STATE.md.
- **CUDA OOM in an activations log:** lower the batch size for the affected model by re-running just that step with a smaller `--batch_size` (e.g. 8, then 4), matching the exact command in the log. The supervisor will pick up the completed work.
- **Judge failures against `api.openai.com`:** the `.env` didn't load. There are NO OpenAI credits — all judge traffic must go through OpenRouter. Fix the environment (`set -a; source /workspace/GemmaAssistantAxis/assistant-axis/.env; set +a`) and let the supervisor retry.
- **OpenRouter rate limiting / 429s:** re-run the judge step with lower `--batch_size` and `--requests_per_second`.
- **A supervisor gave up (8 attempts):** read the last failure in attempts.log, fix the specific cause if it's one of the above, then relaunch just that model: `bash scripts/supervisor.sh <hf_id> <gpu> <key>`. If the cause is something else, leave it and report.
- **One model is dead but the other is fine:** never touch the healthy one.

**Do NOT:** change persona counts, question counts, or `min_count`; edit the pipeline's scientific logic; delete responses, scores, vectors, or release directories; start a second copy of a model that's already running (check with `nvidia-smi` and `ps` first); or terminate the pod.

**When both models finish** (STATE.md shows COMPLETE for each), verify: each `/workspace/exp/<key>/RESULTS.md` exists and its integrity check reads 1.000, `/workspace/exp/COMPARISON.md` exists, and the HF upload succeeded. If the integrity check is not 1.000, say so prominently — it means the axis wasn't built from the saved vector set.

**Your final message** must state, for each model: which steps completed, the headline PC1↔axis cosine, the integrity check value, any interventions you made, and anything a human needs to decide in the morning.
