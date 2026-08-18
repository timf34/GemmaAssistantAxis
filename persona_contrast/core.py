"""Model-free core of the persona-vs-assistant contrast experiment: token alignment, contrast
matrices, de-striping, the assistant-start statistic, and plotting. Everything here runs on
numpy arrays so it can be unit-tested without a GPU; run.py supplies the activations.

Conventions (verified against assistant_axis/axis.py):
    projection[l, t] = h[l, t] · (axis[l] / ||axis[l]||)      # no centering
    higher projection = more Assistant-like  (axis = mean(default) - mean(roles))
    difference[l, t]  = proj_persona[l, t] - proj_control[l, t]
        > 0 (red)  : shifted toward the Assistant representation
        < 0 (blue) : shifted away from Assistant / toward the persona
"""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ------------------------------------------------------------------ token alignment ---------------

@dataclass
class Alignment:
    """Alignment of a persona-condition token sequence against its control-condition sequence."""
    persona_ids: List[int]
    control_ids: List[int]
    persona_tokens: List[str]
    control_tokens: List[str]
    # for each persona-side position, the matched control-side position or -1 if unmatched
    persona_to_control: List[int]
    persona_only: List[int]                    # persona positions with no control match (persona words)
    control_only: List[int]                    # control positions with no persona match
    boundaries: Dict[str, int] = field(default_factory=dict)   # persona-side indices of key tokens

    @property
    def n_matched(self) -> int:
        return sum(1 for m in self.persona_to_control if m >= 0)


def align_sequences(persona_ids: Sequence[int], control_ids: Sequence[int],
                    persona_tokens: Sequence[str], control_tokens: Sequence[str]) -> Alignment:
    """Align two token-id sequences that differ in a small inserted/substituted region.

    Uses difflib on the id sequences: shared runs are matched one-to-one; the persona-word region
    (where the sequences differ) is left unmatched on both sides. Everything after the differing
    region — the user message and the assistant-start tokens — is matched exactly, so it is
    directly subtractable. We never pretend differently tokenized sequences are one-to-one.
    """
    sm = difflib.SequenceMatcher(a=list(persona_ids), b=list(control_ids), autojunk=False)
    p2c = [-1] * len(persona_ids)
    matched_c = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                p2c[i1 + k] = j1 + k
                matched_c.add(j1 + k)
    persona_only = [i for i, m in enumerate(p2c) if m < 0]
    control_only = [j for j in range(len(control_ids)) if j not in matched_c]
    return Alignment(list(persona_ids), list(control_ids), list(persona_tokens), list(control_tokens),
                     p2c, persona_only, control_only)


def find_boundaries(tokens: Sequence[str], ids: Sequence[int], tokenizer=None) -> Dict[str, int]:
    """Locate chat-template boundaries by inspecting the ACTUAL tokens (not assumed names).

    Gemma's template: <bos><start_of_turn>user\\n{content}<end_of_turn>\\n<start_of_turn>model\\n
    We record: first user-turn start, end of user turn, and the assistant-start token
    (the '<start_of_turn>' that precedes 'model'), plus the position of the 'model' token itself.
    Falls back to searching decoded strings so it works across tokenizer versions.
    """
    b: Dict[str, int] = {}
    toks = [t.replace("▁", " ") for t in tokens]
    sot = [i for i, t in enumerate(toks) if "start_of_turn" in t]
    eot = [i for i, t in enumerate(toks) if "end_of_turn" in t]
    if sot:
        b["user_turn_start"] = sot[0]
    if eot:
        b["user_turn_end"] = eot[0]
    # assistant start = the LAST <start_of_turn> (the one opening the model turn)
    if len(sot) >= 2:
        b["assistant_start"] = sot[-1]
        # the 'model' role token typically follows immediately
        for k in range(sot[-1] + 1, min(sot[-1] + 3, len(toks))):
            if toks[k].strip() == "model":
                b["assistant_role_token"] = k
                break
    return b


# ------------------------------------------------------------------ contrast ----------------------

def project_all(acts: np.ndarray, axis: np.ndarray, normalize: bool = True) -> np.ndarray:
    """acts: (n_layers, n_tokens, hidden); axis: (n_layers, hidden) -> (n_layers, n_tokens)."""
    assert acts.shape[0] == axis.shape[0], f"layer mismatch: acts {acts.shape} vs axis {axis.shape}"
    assert acts.shape[2] == axis.shape[1], f"hidden mismatch: acts {acts.shape} vs axis {axis.shape}"
    ax = axis.astype(np.float64)
    if normalize:
        ax = ax / (np.linalg.norm(ax, axis=1, keepdims=True) + 1e-8)
    return np.einsum("lth,lh->lt", acts.astype(np.float64), ax)


def contrast_matrix(proj_persona: np.ndarray, proj_control: np.ndarray, al: Alignment) -> np.ndarray:
    """difference[l, t] over PERSONA-side token positions. Matched positions: persona - control.
    Unmatched (persona-word) positions: NaN — plotted as hatched, never silently subtracted
    against a different token."""
    L, T = proj_persona.shape
    out = np.full((L, T), np.nan)
    for t, m in enumerate(al.persona_to_control):
        if m >= 0:
            out[:, t] = proj_persona[:, t] - proj_control[:, m]
    return out


def destripe_per_layer_mean(diff: np.ndarray, context_cols: Sequence[int]) -> np.ndarray:
    """Subtract each layer's mean difference over the given context columns (shared, matched
    prompt tokens). Removes a per-layer offset so that a layer that is uniformly shifted does not
    read as a horizontal stripe. Transformation: d'[l, t] = d[l, t] - mean_{c in context} d[l, c]."""
    cols = [c for c in context_cols if not np.all(np.isnan(diff[:, c]))]
    base = np.nanmean(diff[:, cols], axis=1, keepdims=True) if cols else 0.0
    return diff - base


# ------------------------------------------------------------------ statistics --------------------

def assistant_start_stats(diff: np.ndarray, al: Alignment, high_frac: float = 0.25,
                          n_context: int = 8) -> Dict:
    """Quantify the assistant-start effect.

    - contrast at the assistant-start token, every layer
    - strongest persona-like (most negative) layer there
    - mean over the final `high_frac` of layers at that token
    - same over the `n_context` matched user-prompt tokens immediately preceding it
    - assistant_start_effect = mean_high(start) - mean_high(preceding)  (negative = persona-ward)
    """
    L = diff.shape[0]
    hi = list(range(int(round(L * (1 - high_frac))), L))
    s = al.boundaries.get("assistant_start")
    if s is None:
        return {"error": "assistant_start boundary not found"}
    col = diff[:, s]
    prev = [c for c in range(max(0, s - n_context), s) if not np.all(np.isnan(diff[:, c]))]
    prev_mat = diff[:, prev] if prev else np.full((L, 1), np.nan)
    start_hi = float(np.nanmean(col[hi]))
    prev_hi = float(np.nanmean(prev_mat[hi, :]))
    return {
        "assistant_start_index": int(s),
        "assistant_start_token": al.persona_tokens[s],
        "contrast_by_layer_at_start": [float(x) for x in col],
        "strongest_persona_layer": int(np.nanargmin(col)),
        "strongest_persona_value": float(np.nanmin(col)),
        "high_layers": [hi[0], hi[-1]],
        "mean_high_layers_at_start": start_hi,
        "mean_high_layers_preceding_context": prev_hi,
        "n_preceding_context_tokens": len(prev),
        "assistant_start_effect": start_hi - prev_hi,
        "interpretation": "negative assistant_start_effect = high layers shift toward the persona at "
                          "the assistant-start token relative to the preceding user-prompt tokens",
    }


# ------------------------------------------------------------------ plotting ----------------------

def symmetric_limit(mats: Sequence[np.ndarray], method: str = "quantile", q: float = 0.98) -> float:
    vals = np.concatenate([m[np.isfinite(m)].ravel() for m in mats])
    if vals.size == 0:
        return 1.0
    if method == "quantile":
        return float(np.quantile(np.abs(vals), q)) or 1.0
    if method == "max":
        return float(np.abs(vals).max()) or 1.0
    raise ValueError(method)


def plot_stack(panels: List[Tuple[str, np.ndarray, Alignment]], out_path: str, title: str,
               vlim: float, scale_note: str, exclude_cols: Optional[Dict[str, List[int]]] = None):
    """panels: [(persona_label, diff (L,T), alignment)]. Layer 0 at bottom, final at top."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    n = len(panels)
    T_max = max(d.shape[1] for _, d, _ in panels)
    fig, axes = plt.subplots(n, 1, figsize=(max(10, 0.32 * T_max + 3), 3.6 * n + 1.2), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    im = None
    for k, (label, diff, al) in enumerate(panels):
        ax = axes[k][0]
        L, T = diff.shape
        drop = set((exclude_cols or {}).get(label, []))
        keep = [t for t in range(T) if t not in drop]
        d = diff[:, keep]
        toks = [al.persona_tokens[t] for t in keep]
        im = ax.imshow(d, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm, interpolation="nearest")
        # hatch the persona-word columns (unmatched -> NaN)
        for j, t in enumerate(keep):
            if np.all(np.isnan(diff[:, t])):
                ax.add_patch(Rectangle((j - 0.5, -0.5), 1, L, fill=False, hatch="////",
                                       edgecolor="#8a8a86", linewidth=0))
        # boundary markers
        pos = {t: j for j, t in enumerate(keep)}
        for name, style in (("user_turn_start", ":"), ("user_turn_end", ":"), ("assistant_start", "-")):
            if name in al.boundaries and al.boundaries[name] in pos:
                x = pos[al.boundaries[name]]
                ax.axvline(x - 0.5, color="#0b0b0b", lw=1.6 if name == "assistant_start" else 0.9,
                           ls=style, alpha=0.9 if name == "assistant_start" else 0.6)
                if name == "assistant_start":
                    ax.text(x - 0.5, L - 0.5, " assistant start", ha="left", va="top", fontsize=8,
                            color="#0b0b0b", fontweight="bold")
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels([t.replace("▁", "␣").replace("\n", "\\n") for t in toks],
                           rotation=90, fontsize=6.5, family="monospace")
        ax.set_yticks(range(0, L, max(1, L // 6)))
        ax.set_ylabel("layer", fontsize=9)
        ax.set_title(f"{label.upper()} vs ASSISTANT", loc="left", fontsize=11, color="#0b0b0b")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    cbar = fig.colorbar(im, ax=[a[0] for a in axes], fraction=0.02, pad=0.01)
    persona_names = " / ".join(p[0] for p in panels)
    cbar.set_label(f"←  more {persona_names}   |   more assistant  →", fontsize=9)
    fig.suptitle(f"{title}\nLayer 0 (bottom) to {panels[0][1].shape[0] - 1} (top)  ·  {scale_note}",
                 fontsize=12, x=0.02, ha="left")
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_start_lines(stats: Dict[str, Dict], out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor("#fcfcfb")
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100"]
    for k, (name, s) in enumerate(stats.items()):
        if "contrast_by_layer_at_start" not in s:
            continue
        y = s["contrast_by_layer_at_start"]
        ax.plot(range(len(y)), y, lw=2, color=colors[k % len(colors)], label=name)
    ax.axhline(0, color="#d9d8d3", lw=0.8)
    L = len(next(iter(stats.values())).get("contrast_by_layer_at_start", [0]))
    ax.axvspan(int(round(L * 0.75)), L - 1, color="#d9d8d3", alpha=0.25, lw=0, label="final 25% of layers")
    ax.set_xlabel("layer"); ax.set_ylabel("persona − control contrast at assistant-start token\n(negative = more persona-like)")
    ax.set_title("Assistant-axis contrast at the assistant-start token, by layer", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(out_path, dpi=170, facecolor=fig.get_facecolor()); plt.close(fig)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else float(o))
