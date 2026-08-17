# Vendored from upstream

This directory is a vendored copy of https://github.com/safety-research/assistant-axis
at commit `a98961956072224eaf244eb289d6c01700b63795` (vendored 2026-08-13), tracked
directly in this repo so modifications persist and clone with it.

Local additions/changes vs upstream:
- `og-paper.md` — markdown copy of the paper (arXiv:2601.10387), not in upstream.
- `pyproject.toml` — added `hf_transfer`/`hf_xet` (RunPod images enable the matching env flags) and
  raised floors to `transformers>=5.5.3` + `vllm>=0.19.1`, which are required for gemma-4's
  `model_type: gemma4`. Earlier vLLM releases pin `transformers<5`, so the two MUST be resolved in
  a single install command or vLLM silently downgrades transformers back below the threshold.
- (record any pipeline code changes here as they happen)
