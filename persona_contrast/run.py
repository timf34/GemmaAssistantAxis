#!/usr/bin/env python3
"""Persona-vs-Assistant contrast across tokens x layers, projected onto the existing Assistant axis.

Reuses, rather than reimplements:
  - assistant_axis.internals.ProbingModel      (model + tokenizer loading, get_layers())
  - the same hook site as assistant_axis/internals/activations.py:
        forward hook on model.layers[i], output[0]  == post-decoder-layer residual stream
  - assistant_axis.internals.ConversationEncoder (chat template application)
  - assistant_axis.axis.load_axis / the project() convention (normalise axis, dot, no centering)

Usage (on the GPU pod, from the repo root):
    uv run --project assistant-axis python -m persona_contrast.run --config persona_contrast/config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "assistant-axis"))
from assistant_axis.axis import load_axis                     # noqa: E402
from assistant_axis.internals import ProbingModel, ConversationEncoder  # noqa: E402

from persona_contrast.core import (                            # noqa: E402
    align_sequences, find_boundaries, project_all, contrast_matrix, destripe_per_layer_mean,
    assistant_start_stats, symmetric_limit, plot_stack, plot_start_lines, save_json,
)


def build_messages(cfg, persona_phrase: str):
    system = cfg["system_template"].format(persona=persona_phrase)
    # Gemma has no system role: the instruction goes at the head of the first user turn.
    # ConversationEncoder.format_chat handles that folding for gemma models; we build the same
    # single-user-turn structure explicitly so both conditions are token-identical outside the slot.
    return [{"role": "user", "content": f"{system}\n\n{cfg['user_message']}"}]


@torch.no_grad()
def forward_all_layers(pm: ProbingModel, input_ids: torch.Tensor):
    """Return activations (n_layers, n_tokens, hidden) from the SAME hook site the axis used."""
    layers = pm.get_layers()
    acts = [None] * len(layers)
    handles = []

    def mk(i):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            acts[i] = t[0].detach().float().cpu()          # (n_tokens, hidden)
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(mk(i)))
    try:
        pm.model(input_ids=input_ids.to(pm.model.device))
    finally:
        for h in handles:
            h.remove()
    return torch.stack(acts).numpy()                          # (L, T, H)


@torch.no_grad()
def greedy_extend(pm: ProbingModel, input_ids: torch.Tensor, n: int) -> torch.Tensor:
    if n <= 0:
        return input_ids
    out = pm.model.generate(input_ids=input_ids.to(pm.model.device), max_new_tokens=n, do_sample=False,
                            pad_token_id=pm.tokenizer.pad_token_id or pm.tokenizer.eos_token_id)
    return out[0:1].cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="tokenize + align + diagnostics only, no model")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out_dir = REPO / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- axis -----------------------------------------------------------------------------------
    axis_path = REPO / cfg["axis_file"]
    if not axis_path.exists():
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(cfg["axis_hf_repo"], cfg["axis_hf_path"], repo_type="dataset",
                            token=os.environ.get("HF_TOKEN"))
        axis_path.parent.mkdir(parents=True, exist_ok=True)
        Path(p).replace(axis_path) if False else __import__("shutil").copy2(p, axis_path)
    axis = load_axis(str(axis_path)).float().numpy()          # (L, H)
    print(f"[axis] {axis_path.name}: shape {axis.shape}  (layers x hidden)")

    # ---- model ----------------------------------------------------------------------------------
    torch.manual_seed(0)
    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model"])
        enc = ConversationEncoder(tok, cfg["model"])
        pm = None
    else:
        pm = ProbingModel(cfg["model"])
        pm.model.eval()
        tok = pm.tokenizer
        enc = ConversationEncoder(tok, cfg["model"])
        n_layers = len(pm.get_layers())
        hidden = pm.model.config.text_config.hidden_size if hasattr(pm.model.config, "text_config") \
            else pm.model.config.hidden_size
        assert axis.shape == (n_layers, hidden), \
            f"axis {axis.shape} does not match model (layers={n_layers}, hidden={hidden})"
        print(f"[model] {cfg['model']}: {n_layers} layers, hidden {hidden}, eval={not pm.model.training}")

    # ---- conditions -----------------------------------------------------------------------------
    control_msgs = build_messages(cfg, cfg["control_persona_word"])
    ctrl_ids = enc.token_ids(control_msgs, add_generation_prompt=True)
    ctrl_toks = tok.convert_ids_to_tokens(ctrl_ids)
    ctrl_ids_t = torch.tensor([ctrl_ids])
    if pm and cfg.get("generate_tokens", 0) > 0:
        ctrl_ids_t = greedy_extend(pm, ctrl_ids_t, cfg["generate_tokens"])
        ctrl_ids = ctrl_ids_t[0].tolist(); ctrl_toks = tok.convert_ids_to_tokens(ctrl_ids)
    ctrl_proj = None
    if pm:
        ctrl_acts = forward_all_layers(pm, ctrl_ids_t)
        ctrl_proj = project_all(ctrl_acts, axis, normalize=cfg["normalize_axis"])
        np.save(out_dir / "proj_control.npy", ctrl_proj)

    # sign sanity check: the control (assistant) condition should project HIGHER than the personas
    # at the assistant-start token in high layers, if the axis is oriented assistant-positive.
    panels, stats, diagnostics, raw_store = [], {}, {"control": {"ids": ctrl_ids, "tokens": ctrl_toks}}, {}
    for persona in cfg["personas"]:
        phrase = cfg.get("persona_articles", {}).get(persona, f"a {persona}")
        msgs = build_messages(cfg, phrase)
        ids = enc.token_ids(msgs, add_generation_prompt=True)
        ids_t = torch.tensor([ids])
        if pm and cfg.get("generate_tokens", 0) > 0:
            ids_t = greedy_extend(pm, ids_t, cfg["generate_tokens"])
            ids = ids_t[0].tolist()
        toks = tok.convert_ids_to_tokens(ids)
        al = align_sequences(ids, ctrl_ids, toks, ctrl_toks)
        al.boundaries = find_boundaries(toks, ids)
        diagnostics[persona] = {
            "phrase": phrase, "ids": ids, "tokens": toks, "boundaries": al.boundaries,
            "persona_only_positions": al.persona_only, "control_only_positions": al.control_only,
            "n_matched": al.n_matched, "n_tokens": len(ids),
        }
        print(f"\n[{persona}] {len(ids)} tokens, {al.n_matched} matched to control, "
              f"persona-only cols {al.persona_only}, boundaries {al.boundaries}")
        print("   persona:", " ".join(t.replace('▁', '␣') for t in toks))
        print("   control:", " ".join(t.replace('▁', '␣') for t in ctrl_toks))
        if pm is None:
            continue
        acts = forward_all_layers(pm, ids_t)
        proj = project_all(acts, axis, normalize=cfg["normalize_axis"])
        np.save(out_dir / f"proj_{persona}.npy", proj)
        diff = contrast_matrix(proj, ctrl_proj, al)
        np.save(out_dir / f"diff_raw_{persona}.npy", diff)
        raw_store[persona] = (diff, al)
        panels.append((persona, diff, al))
        stats[persona] = assistant_start_stats(diff, al)
        # sign check at the assistant-start token, high layers
        s = al.boundaries.get("assistant_start")
        if s is not None and al.persona_to_control[s] >= 0:
            L = proj.shape[0]; hi = slice(int(L * 0.75), L)
            pv, cv = float(proj[hi, s].mean()), float(ctrl_proj[hi, al.persona_to_control[s]].mean())
            print(f"   sign check @assistant-start high layers: persona {pv:.3f} vs control {cv:.3f} "
                  f"-> {'OK (control more assistant-like)' if cv > pv else 'NOTE: persona >= control here'}")

    save_json(diagnostics, out_dir / "tokens_and_boundaries.json")
    save_json({"config": cfg}, out_dir / "config_used.json")
    if pm is None:
        print(f"\n[dry-run] diagnostics written to {out_dir}/tokens_and_boundaries.json")
        return

    # ---- metadata-token exclusion (logged; full version always saved) --------------------------
    exclude_cols = {}
    if cfg.get("exclude_metadata_tokens"):
        for persona, (diff, al) in raw_store.items():
            cols = [i for i, t in enumerate(al.persona_tokens) if t in ("<bos>", "<eos>", "<pad>")]
            exclude_cols[persona] = cols
            print(f"[{persona}] excluding metadata token columns {cols}: {[al.persona_tokens[c] for c in cols]}")

    # ---- plots ----------------------------------------------------------------------------------
    cs = cfg["color_scale"]
    vlim = symmetric_limit([d for _, d, _ in panels], cs["method"], cs.get("q", 0.98))
    note = f"colour limit ±{vlim:.2f} ({cs['method']} q={cs.get('q', '')} of |diff|), zero-centred"
    plot_stack(panels, str(out_dir / "heatmap_raw.png"), "Assistant Axis Difference (raw)", vlim, note, exclude_cols)
    if cfg.get("destripe") == "per_layer_mean_over_context":
        ds_panels = []
        for persona, diff, al in panels:
            s = al.boundaries.get("assistant_start", diff.shape[1])
            ctx = [t for t in range(s) if al.persona_to_control[t] >= 0]
            dd = destripe_per_layer_mean(diff, ctx)
            np.save(out_dir / f"diff_destriped_{persona}.npy", dd)
            ds_panels.append((persona, dd, al))
        vlim2 = symmetric_limit([d for _, d, _ in ds_panels], cs["method"], cs.get("q", 0.98))
        plot_stack(ds_panels, str(out_dir / "heatmap_destriped.png"),
                   "Assistant Axis Difference (de-striped: per-layer mean over shared context removed)",
                   vlim2, f"colour limit ±{vlim2:.2f}; d'[l,t] = d[l,t] − mean_context d[l,·]", exclude_cols)
    plot_start_lines(stats, str(out_dir / "assistant_start_by_layer.png"))
    save_json(stats, out_dir / "assistant_start_stats.json")

    print("\n== assistant-start effect (negative = high layers shift toward the persona) ==")
    for persona, s in stats.items():
        print(f"  {persona:10s} effect={s['assistant_start_effect']:+.3f}   "
              f"start(high)={s['mean_high_layers_at_start']:+.3f}  ctx(high)={s['mean_high_layers_preceding_context']:+.3f}  "
              f"strongest persona layer={s['strongest_persona_layer']} ({s['strongest_persona_value']:+.3f})")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
