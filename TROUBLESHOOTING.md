# Troubleshooting: getting gemma-4-31B-it running on a pod

Every issue that has cost this project a run, with the symptom, the real cause, and the fix.
Written 2026-08-17 while unblocking gemma-4-31B-it on a 1xH200 RunPod pod.

**The theme.** Almost nothing here was a Gemma 4 bug. Gemma 4 serves fine under vLLM. The failures
were (a) a driver/toolkit mismatch, (b) two upstream version-floor bugs, and (c) this repo's own
assumptions, written for Gemma 2/3 on transformers 4.x, meeting a `*ForConditionalGeneration` model
on transformers 5.x. They surfaced one at a time because each was only reachable after the previous
was fixed. If you are bringing up a new model here, expect the same shape of problem.

**Read this first if you are in a hurry:** the working combination on driver 550 is
`cuda-compat-13-0` + `torch 2.13.0+cu130` + `vllm 0.27.1` + `transformers 5.14.1` +
`openai 2.54.0`, and `uv.lock` already pins it. Run `bash scripts/doctor.sh` — it checks most of
this in about a minute.

---

## 1. `torch.cuda.is_available()` is False — "driver too old"

**Symptom**
```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12040).
torch.cuda.is_available() -> False
```
`import torch` itself SUCCEEDS. Only CUDA init fails.

**Cause.** The host kernel driver is 550.144.03 (CUDA 12.4) and cannot be changed from inside the
container — it is a host property, and the pod template's "CUDA version" does not affect it. The
venv's torch is built for **CUDA 13.0** (`torch 2.13.0+cu130`).

**Fix.** NVIDIA forward compatibility: a newer USER-mode libcuda that drives the older KERNEL
module. Supported for data center GPUs (H100/H200/A100) — this is a documented NVIDIA
configuration, not a hack. See <https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html>.

```bash
sudo apt-get install -y cuda-compat-13-0        # libcuda 580.x, installs to /usr/local/cuda-13.0/compat
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
```
Persist it in `/workspace/exp/.cuda_compat.env` (sourced by `scripts/common.sh`, so every pipeline
step inherits it). Verify:
```bash
cd assistant-axis && uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA H200
```

**Pick the compat version from TORCH, not from a constant.** The package must match
`torch.version.cuda` (13.0 here → `cuda-compat-13-0`). `scripts/cuda_compat.sh` derives this
automatically; `doctor.sh` and `upgrade_for_gemma4.sh` use it. Earlier versions of those scripts
hardcoded `cuda-compat-12-8` / `/usr/local/cuda-12.8/compat`, a path that could never appear on this
pod no matter how many times it was installed.

**Gotcha — `nvidia-smi` starts reporting 13.0.** Once compat is on `LD_LIBRARY_PATH`, `nvidia-smi`
reports the version of the libcuda it loaded, so it prints CUDA 13.0 on a box whose kernel driver is
still 12.4. Not a bug and not the host driver changing:
```bash
nvidia-smi | grep CUDA                                              # 12.4
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat nvidia-smi | grep CUDA  # 13.0
cat /proc/driver/nvidia/version                                     # the truth: 550.144.03
```

**Gotcha — an installed `cuda-compat` package containing no libraries.**
```bash
dpkg -L cuda-compat-12-8    # only /usr/share/doc entries; no libcuda.so.*
```
On this image `cuda-compat-12-8` was registered at version `575.57.08`, which exists **only in
`/var/lib/dpkg/status`** and in NVIDIA's repo at *no* version (`apt-cache policy` shows the repo
candidate as `570.211.01`). It is a stripped phantom entry baked into the image, so
`apt-get install --reinstall` can never fix it — apt has no matching version to reinstall. Always
test for the actual file, never for the directory or the dpkg entry:
```bash
test -e /usr/local/cuda-13.0/compat/libcuda.so.1
```

**Not the problem: `libcusparseLt.so.0`.** It ships as the `nvidia-cusparselt-cu13` wheel and is
already in the venv. If you see it named as a blocker, that report is stale.

**If forward-compat genuinely fails**, you need a host with driver >= 570. Nothing installable
inside the container will help.

---

## 2. `ImportError: cannot import name 'NamespaceTool' from 'openai.types.responses'`

**Symptom.** `import vllm` works, but `from vllm import LLM` dies. (The broken module is imported
lazily, which is why a bare `import vllm` smoke test passes and hides this.)

**Cause.** vLLM 0.27.1 declares `openai>=2.0.0`, but `vllm/tool_parsers/utils.py` imports
`NamespaceTool`, which first exists in **openai 2.25.0** (verified by bisecting wheels: 2.24.0 does
not have it). The lock had resolved openai 2.15.0 — satisfying vLLM's declared floor while breaking
its actual imports. Upstream class of bug: <https://github.com/vllm-project/vllm/issues/22482>.

**Fix.** In `assistant-axis/pyproject.toml`:
```toml
"openai>=2.25.0,<3",   # capped so the resolver cannot move the judge client across a major version
```
then `uv lock`. Currently resolves to 2.54.0.

**Do not fix this with `uv pip install`.** `uv run` and `uv sync` re-sync the environment to
`uv.lock`, so a bare pip install is silently reverted — and `run_on_pod.sh` runs `uv sync` at step
1/6. The constraint has to live in `pyproject.toml` + `uv.lock`.

---

## 3. `AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute`

**Symptom.** vLLM dies during engine init, before loading any weights:
```
File "vllm/transformers_utils/model_arch_config_convertor.py", line 608, in get_head_size
    head_dim = getattr(self.hf_text_config, "head_dim", 0)
AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute and may vary across layers.
```

**Cause.** transformers **5.15.0** rewrote `Gemma4Config` to synthesize a `per_layer_config` that
overrides `head_dim` with `global_head_dim`, marking the config heterogeneous — so any plain read of
`config.head_dim` raises. vLLM 0.27.1 reads it directly. In **5.14.1** `head_dim=256` and
`global_head_dim=512` are flat attributes, which is the layout vLLM's gemma4 code expects.
Note `getattr(..., "head_dim", 0)` does NOT absorb this: the default only catches `AttributeError`,
and this is a `RuntimeError` subclass. Upstream: <https://github.com/vllm-project/vllm/issues/51744>.

**Fix.** Cap transformers below 5.15 in `assistant-axis/pyproject.toml`:
```toml
"transformers>=5.5.3,<5.15",   # resolves to 5.14.1
```
vLLM 0.27.1 is the LATEST release and declares only `transformers>=5.5.3` with no cap, so the
resolver picks 5.15.0 and breaks itself. **Revisit this cap** once a vLLM release ships the
`per_layer_config`-aware `head_dim` lookup (PRs #49797 / #49959) — then transformers can move up.

Downgrading costs nothing here: 5.14.1 reports identical config values (60 layers, vocab 262144).

---

## 4. The vLLM serve test passes while the model emits gibberish

**Symptom.** `scripts/upgrade_for_gemma4.sh` printed `VLLM OK: a. a. a. a.` and exited **0**,
green-lighting an overnight run on degenerate output.

**Cause.** The test called `llm.generate(["Say ok."])` on the **raw string**. An instruct model with
no chat template just continues the text: `"ok. ok. ok. ok. ..."` (1/256 distinct tokens). The
pipeline templates every prompt (`generation.py:227`), so the test was not exercising the real path.
Generation was fine all along.

**Fix.** The serve test now applies `apply_chat_template(..., add_generation_prompt=True)`, checks a
distinct-token ratio to catch repetition loops, and asserts a factual answer ("Paris"). "It returned
a string" is not the same as "it generated language".

**Lesson.** Any smoke test that will authorize hours of GPU time must assert on *content*, not just
absence of an exception.

---

## 5. Doctor: "persona instructions would be dropped" (false alarm)

**Symptom.** `[FAIL] google/gemma-4-31B-it — persona instructions would be dropped`, blocking the
run. Gemma 3 failed the identical check.

**Cause.** `scripts/check_chat_template.py` asked "does the template render a system turn?" and
treated **yes** as unsafe, on the premise that these models were trained without a system role. That
inverted the meaning of its own signal:
* gemma-3 has no system role but its template MERGES system text into the user turn, so the probe
  string appears — flagged unsafe, when merging is exactly the safe behaviour.
* gemma-4 has a genuine system role (`<|turn>system ... <turn|>`) — flagged unsafe, when that is the
  model's native persona channel.

Verified by generation on the pipeline's own path: a "speak like a pirate" system turn returns
*"Ahoy there, matey! ... the capital be **Paris**"*, and a Shakespearean-bard persona answers 2+2 in
Elizabethan English. Persona obeyed AND question answered. Nothing was being dropped.

**Fix.** The check now calls the pipeline's own `format_conversation()`, renders what it produces,
and asserts the persona text is present in the final prompt — testing delivery instead of guessing
from a substring.

**Do NOT "fix" this with `ASSISTANT_AXIS_FORCE_USER_CONCAT=1`.** No pipeline code reads that
variable — it only flips the doctor's verdict, silencing a results-integrity check without changing
what the model receives.

---

## 6. `FATAL: need 2 GPUs, found 1`

**Symptom.** `run_on_pod.sh` aborts in preflight on a healthy 1xH200 pod — **after** the doctor has
already printed "ALL ENVIRONMENT CHECKS PASSED — SAFE TO LEAVE IT RUNNING".

**Cause.** A leftover 2-GPU assertion in `scripts/preflight.sh`, contradicting the rest of the repo:
line 11 of that same file already maps models onto "whatever GPUs exist", `doctor.sh` reports the
1-GPU plan as `[ok]`, and `run_on_pod.sh` has an explicit `PARALLEL=0` branch. One GPU is slower
(models run sequentially, ~2x wall-clock), not invalid.

**Fix.** Require `>= 1` GPU and print a note when running sequentially.

**Also fixed here:** preflight ran its tiny end-to-end on BOTH models via a hardcoded `for i in 0 1`,
ignoring `MODELS_ONLY`. A `MODELS_ONLY=gemma-4-31b` run still downloaded gemma-3 (~54GB) and put it
through a full generate/activations/judge cycle. It now honours `MODELS_ONLY` like `run_on_pod.sh`.

---

## 7. `'Gemma4Config' object has no attribute 'num_hidden_layers'`

**Symptom.** `ValueError: Could not infer config for model google/gemma-4-31B-it` from
`assistant_axis/models.py:get_config()` — moments after the doctor reported `layers=60` for the same
model.

**Cause.** gemma-3 and gemma-4 are `*ForConditionalGeneration` (multimodal) models: the
language-model hyperparameters live under `config.text_config`, not at the top level. `doctor.sh`
already read it with a fallback; `models.py` did not.

**Fix.** Read the top level, then fall back to `text_config`:
```python
total_layers = getattr(config, "num_hidden_layers", None) or getattr(
    getattr(config, "text_config", None), "num_hidden_layers", None)
```
Sanity values: gemma-4 60 layers / target 30, gemma-3 62 / 31, gemma-2 46 / 22. `target_layer` is
`total // 2`, matching every hand-entered entry in `MODEL_CONFIGS`.

---

## 8. `Could not find transformer layers for model ... (class: Gemma4ForConditionalGeneration)`

**Symptom.** Generation succeeds, then `pipeline/2_activations.py` dies at `pm.get_layers()`.

**Cause.** transformers 5.x moved the VLM text stack one level down. `get_layers()` knew the 4.x
path `m.language_model.layers`, but under 5.14.1 both gemma models expose it at
`m.model.language_model.layers`:

| model | path | layers |
|---|---|---|
| Gemma4ForConditionalGeneration | `model.model.language_model.layers` | 60 x `Gemma4TextDecoderLayer` |
| Gemma3ForConditionalGeneration | `model.model.language_model.layers` | 62 x `Gemma3DecoderLayer` |

**Fix.** Added that path to `layer_paths` in `assistant_axis/internals/model.py`.

**DANGER — do not make this match loosely.** Both models also carry a vision tower at
`model.vision_tower.encoder.layers` (27 layers). A glob or a broader fallback would silently probe
the **image encoder** and produce role vectors that are correctly shaped, pass every downstream
check, and are completely meaningless. Keep the path pinned to `language_model`, and assert the
resolved layer class is a text decoder, not `*VisionEncoderLayer` / `SiglipEncoderLayer`.

To inspect a new model's layout cheaply (no weights downloaded):
```python
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights
import torch.nn as nn
cfg = AutoConfig.from_pretrained(M)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(cfg)
for name, mod in model.named_modules():
    if isinstance(mod, nn.ModuleList) and len(mod) > 0:
        print(name, len(mod), type(mod[0]).__name__)
```

**Note on existing gemma-3 results.** The gemma-3 axis (`results/gemma-3-27b/`, committed
2026-08-14) was produced on transformers 4.x, where the old path was correct, and is unaffected —
its integrity cosine is 0.9999. Only a RE-RUN of gemma-3 would have hit this, which this fix
prevents.

---

## 9. `TypeError: unsupported operand type(s) for +: 'BatchEncoding' and 'list'`

**Symptom.** Generation succeeds and `get_layers()` finds the model, then activation extraction
dies at `activations.py:308` (`padded_ids = ids + [pad] * n`).

**Cause.** On transformers 5.x, `tokenizer.apply_chat_template(..., tokenize=True)` returns a
`BatchEncoding` (`{'input_ids': [...], 'attention_mask': [...]}`); on 4.x it returned a plain
`list[int]`. `assistant_axis/internals/conversation.py` had **nine** call sites feeding that result
into span arithmetic — `len()`, longest-common-prefix, subsequence search, list concatenation.

**Why this one is nasty.** `len(BatchEncoding)` is **2** — the number of *keys* — so every offset
would be computed as though the conversation were two tokens long. It happened to crash at the `+`.
It could just as easily have produced silently corrupt spans that flowed straight into the role
vectors and passed every downstream check.

**Fix.** One `_chat_ids()` helper on `ConversationEncoder` that always returns `list[int]`
(unwraps `input_ids`, `.tolist()`s tensors, unwraps a `[[...]]` batch), and all nine sites routed
through it. Do not call `apply_chat_template(tokenize=True)` directly anywhere else in that file.

**Verify** — the test that matters is that spans decode back to their own content, not merely that
nothing crashes:
```python
full_ids, spans = enc.build_turn_spans(conv)
for s in spans:
    print(s['role'], tok.decode(full_ids[s['start']:s['end']]))
# user      -> 'What is the relationship between law and morality?'
# assistant -> 'Arr, the law be the map and morality be the compass, matey.'
```

---

## 10. Step-0 driver check clobbers a working compat config

**Symptom.** A launch on an already-working pod suddenly fails the driver check, or
`/workspace/exp/.cuda_compat.env` points at `/usr/local/cuda-12.8/compat` after having pointed at
`13.0`.

**Cause.** Commit `344a707` added an early "step 0" driver check to `run_on_pod.sh` — a good idea
(fail in one second, before the multi-minute `uv sync`) — but as written it re-hardcoded
`cuda-compat-12-8`, tested `-d` on the directory (passes on the phantom package, see #1), and
**overwrote** `.cuda_compat.env` with that wrong path on every launch.

**Fix.** Step 0 now goes through `scripts/cuda_compat.sh` (version from torch; glob fallback on a
fresh pod with no venv) and only writes `.cuda_compat.env` when torch is unqueryable *and* no
verified file exists. The doctor — which runs after deps install and can ask torch — remains the
authority and (re)writes the file after confirming the GPU actually works.

**Lesson.** Any new code path that decides the compat version must go through `cuda_compat.sh`.
There have now been three independent hardcodings of `12-8` in this repo; each one broke the pod.

---

## 11. Preflight: `IndexError: list index out of range` after "Skipping <role>: no role file found"

**Symptom.** Activation extraction succeeds (`activation entry shape: (60, 5376)`), then the judge
logs `Skipping default: no role file found` / `Skipping pirate: no role file found`, writes no
scores, and preflight's sanity read dies with a bare `IndexError`.

**Cause.** Preflight invoked `3_judge.py` without `--roles_dir`. The judge's default is
`../data/roles/instructions` — a **relative** path resolved against the CWD, which `run_on_pod.sh`
sets to the repo root — so it looked in `/workspace/data/roles/instructions` and found nothing.
Preflight-only: the real run (`run_model.sh`) always passes the flag.

**Fix.** Pass `--roles_dir "$AXIS_DIR/data/roles/instructions"` in preflight, exactly as
`run_model.sh` does. The sanity read now names its failure ("judge wrote no score files") instead
of an `IndexError`, so a silently-empty judge can never pass the gate again.

**Sanity values once it works.** Pirate responses score **3/3** ("fully playing the role") across
all 15 preflight samples — that is correct, not a stuck judge: 3 is the top of the 0–3 rubric and
Gemma 4 inhabits personas strongly. `default` is skipped with "no eval_prompt in role file" by
design; it is the assistant baseline, not a persona.

---

## Self-shutdown (`SHUTDOWN=stop`) silently does nothing

`run_on_pod.sh` requires `RUNPOD_POD_ID` **and** an authenticated `runpodctl`. On this pod neither
was present: no `RUNPOD_*` env vars at all (the hostname is a Docker container ID, not a pod ID), and
`~/.runpod/config.toml` held an empty `apikey`. The run completes and leaves the pod **billing**.

```bash
runpodctl config --apiKey <KEY>
runpodctl get pod                       # read-only check; find your pod's ID here
RUNPOD_POD_ID=<id> SHUTDOWN=stop bash run_on_pod.sh
```
Verify auth with `runpodctl get pod` — never test with `runpodctl stop pod`, which stops the pod you
are working on. Note the API key persists in `~/.runpod/config.toml` across a `stop` (disk is kept).

---

## Quick reference

| Symptom | Cause | Fix |
|---|---|---|
| `cuda.is_available()` False, "driver too old" | driver 550 (12.4) vs torch cu130 | `cuda-compat-13-0` + `LD_LIBRARY_PATH` |
| `dpkg -L cuda-compat-*` shows only docs | phantom dpkg entry, not in repo | install the real `.deb`; test for `libcuda.so.1` |
| `NamespaceTool` ImportError | vLLM's openai floor too low | `openai>=2.25.0,<3` |
| `AmbiguousGlobalPerLayerAttributeError` | transformers 5.15 heterogeneous config | `transformers>=5.5.3,<5.15` |
| serve test "OK" but output is `a. a. a.` | no chat template in the test | template the prompt + assert coherence |
| "persona instructions would be dropped" | inverted heuristic in the doctor | check actual delivery via `format_conversation()` |
| `FATAL: need 2 GPUs, found 1` | stale 2-GPU assertion | require `>= 1` |
| `no attribute 'num_hidden_layers'` | multimodal config nests under `text_config` | getattr fallback |
| `Could not find transformer layers` | 5.x moved VLM text stack | add `model.model.language_model.layers` |
| `BatchEncoding + list` TypeError | 5.x `apply_chat_template(tokenize=True)` returns BatchEncoding | `_chat_ids()` helper → `list[int]` |
| `.cuda_compat.env` reverts to 12.8 | step-0 check re-hardcoded the version | route through `cuda_compat.sh` |
| pod keeps billing after run | no `RUNPOD_POD_ID` / unauth `runpodctl` | configure both |
