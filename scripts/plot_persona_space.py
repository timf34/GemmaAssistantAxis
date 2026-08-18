#!/usr/bin/env python3
"""Persona-space scatter: PC1 (y) vs PC2 (x) of the role vectors at one layer, one panel per
model, each persona coloured by a hand-assigned category. Axis labels carry the variance explained.

PC1 is oriented so the assistant end points UP (its sign is arbitrary in PCA); PC2's sign is
oriented consistently across panels by aligning it to the first panel where roles overlap.

Usage:
    python scripts/plot_persona_space.py \\
        --release "Gemma 3 27B=/path/to/gemma-3-27b/release/gemma-3-27b" \\
        --release "Gemma 4 31B=/path/to/gemma-4-31b/release/gemma-4-31b" \\
        --out persona_space.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

# ---- persona categories (all 275 roles + default) -----------------------------------------------
CATEGORIES = {
    "assistant-like": """assistant tutor teacher instructor mentor coach guide counselor therapist translator
        interpreter editor proofreader summarizer secretary consultant facilitator mediator moderator
        librarian curator archivist grader evaluator examiner reviewer validator screener scheduler
        coordinator organizer planner dispatcher trainer navigator advocate caregiver healer synthesizer
        generalist specialist collaborator supervisor recruiter presenter""",
    "professional": """accountant analyst auditor economist statistician mathematician physicist chemist biologist
        scientist researcher scholar historian anthropologist archaeologist sociologist psychologist
        linguist geographer cartographer engineer architect builder mechanic programmer debugger
        technologist designer doctor paramedic pharmacist nutritionist veterinarian lawyer judge
        detective journalist reporter marketer merchant auctioneer entrepreneur producer publisher
        blogger podcaster photographer musician composer novelist writer playwright poet bard actor
        comedian chef bartender sommelier pilot soldier veteran ambassador emissary diplomat negotiator
        strategist forecaster futurist theorist philosopher critic interviewer influencer celebrity
        activist revolutionary politician""",
    "human (other)": """adolescent teenager toddler infant student graduate parent grandparent elder retiree
        newlywed divorcee widow orphan patient addict amnesiac prisoner criminal smuggler pirate rogue
        spy hacker vigilante warrior competitor gamer surfer daredevil traveler nomad wanderer flaneur
        pilgrim hermit exile expatriate immigrant refugee provincial cosmopolitan bohemian hedonist
        ascetic minimalist hoarder collector workaholic procrastinator perfectionist purist
        traditionalist luddite rebel anarchist maverick contrarian cynic skeptic stoic pragmatist
        realist idealist optimist romantic dreamer altruist martyr zealot evangelist prophet guru
        sage mystic shaman witch fool jester trickster gossip narcissist loner survivor caveman
        ancient absurdist dilettante amateur prodigy virtuoso polymath improviser observer witness
        peacekeeper pacifist guardian protector saboteur destroyer provocateur devils_advocate
        fixer networker""",
    "non-human": """demon angel ghost wraith revenant spirit vampire golem chimera leviathan whale predator
        prey parasite virus tree coral_reef mycorrhizal ecosystem wind void alien eldritch aberration
        familiar genie oracle hive swarm symbiont hybrid crystalline homunculus""",
    "AI / abstract": """robot cyborg simulacrum echo tulpa egregore avatar zeitgeist chameleon shapeshifter
        default""",
}
ROLE2CAT = {r: c for c, roles in CATEGORIES.items() for r in roles.split()}

# validated 5-slot categorical palette (dataviz reference instance; CVD ΔE 9.2)
COLORS = {
    "assistant-like": "#2a78d6",
    "professional":   "#eb6834",
    "human (other)":  "#eda100",
    "non-human":      "#1baf7a",
    "AI / abstract":  "#4a3aa7",
}


def load_release(d):
    d = Path(d)
    axis = torch.load(d / "assistant_axis.pt", weights_only=False).float().numpy()
    default = torch.load(d / "default_vector.pt", weights_only=False).float().numpy()
    roles = {p.stem: torch.load(p, weights_only=False).float().numpy() for p in (d / "role_vectors").glob("*.pt")}
    return axis, default, roles


def pca_layer(axis, default, roles, layer, standardize=True):
    from sklearn.decomposition import PCA
    names = sorted(roles)
    X = np.stack([roles[n][layer] for n in names])
    mu = X.mean(0)
    # Standardize each hidden dimension across roles (z-score). Gemma 2/3 have "massive activation"
    # dimensions (one coordinate spanning thousands while the rest span tens); without this, PC1 is
    # just that coordinate and personas at its extremes (poet, toddler, pirate) masquerade as the
    # most assistant-like. sklearn StandardScaler-equivalent, applied to the default too.
    sd = X.std(0) + 1e-8 if standardize else np.ones_like(mu)
    Xs = (X - mu) / sd
    pca = PCA(n_components=2).fit(Xs)
    Z = Xs @ pca.components_.T
    zd = ((default[layer] - mu) / sd) @ pca.components_.T
    # Orient PC1 so "more assistant-like" points UP. Use the DEFAULT's position (not the axis
    # dot-product): the default sits at the assistant end by construction, and the raw axis dot
    # product mis-oriented Gemma 3 whose projections are large-magnitude negative.
    if zd[0] < np.median(Z[:, 0]):
        Z[:, 0] *= -1; zd[0] *= -1
        pca.components_[0] *= -1
    return names, Z, zd, pca.explained_variance_ratio_, pca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", action="append", required=True, help='"Label=/path/to/release" (repeatable)')
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=None, help="layer index (default: each model's middle layer)")
    ap.add_argument("--label-n", type=int, default=6, help="direct-label the N most extreme personas per end of PC1")
    ap.add_argument("--no-standardize", action="store_true", help="raw PCA (no per-dimension z-scoring)")
    ap.add_argument("--flip-axes", action="store_true", help="PC1 on x, PC2 on y (default: PC1 on y)")
    ap.add_argument("--clip-pc1", type=float, default=None,
                    help="clip PC1 to the central quantile range, e.g. 0.02 keeps [2%%,98%%]; outliers are pinned "
                         "at the edge and drawn hollow so they are visibly present, not silently dropped")
    ap.add_argument("--clip-pc2", type=float, default=None, help="same, for PC2")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = []
    for spec in args.release:
        label, path = spec.split("=", 1)
        axis, default, roles = load_release(path)
        L = args.layer if args.layer is not None else axis.shape[0] // 2
        names, Z, zd, var, pca = pca_layer(axis, default, roles, L, standardize=not args.no_standardize)
        panels.append(dict(label=label, layer=L, names=names, Z=Z, zd=zd, var=var, n_layers=axis.shape[0]))

    # orient PC2 consistently: flip each panel's PC2 to agree with the first panel over shared roles
    ref = panels[0]
    ref_pos = {n: ref["Z"][i, 1] for i, n in enumerate(ref["names"])}
    for p in panels[1:]:
        common = [n for n in p["names"] if n in ref_pos]
        if len(common) >= 10:
            a = np.array([ref_pos[n] for n in common])
            b = np.array([p["Z"][p["names"].index(n), 1] for n in common])
            if np.corrcoef(a, b)[0, 1] < 0:
                p["Z"][:, 1] *= -1; p["zd"][1] *= -1

    n = len(panels)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    pw, ph = (10.5, 6.2) if (args.flip_axes and n == 1) else (7.2, 6.4)
    fig, axes = plt.subplots(rows, cols, figsize=(pw * cols, ph * rows), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")

    for k, p in enumerate(panels):
        ax = axes[k // cols][k % cols]
        ax.set_facecolor("#fcfcfb")
        Z, names, var = p["Z"].copy(), p["names"], p["var"]
        zd = p["zd"].copy()
        cats = [ROLE2CAT.get(nm, "human (other)") for nm in names]

        # Clip PC1 to the central quantile range so the main cluster fills the panel. Points outside
        # are pinned at the edge and drawn hollow — visibly present, never silently dropped.
        clipped = np.zeros(len(names), dtype=bool)
        for comp, q in ((0, args.clip_pc1), (1, args.clip_pc2)):
            if q:
                lo, hi = np.quantile(Z[:, comp], [q, 1 - q])
                pad = 0.03 * (hi - lo)
                clipped |= (Z[:, comp] < lo) | (Z[:, comp] > hi)
                Z[:, comp] = np.clip(Z[:, comp], lo - pad, hi + pad)
                zd[comp] = np.clip(zd[comp], lo - pad, hi + pad)

        # axis assignment: default PC1 on y; --flip-axes puts PC1 on x
        def xy(row):
            return (row[0], row[1]) if args.flip_axes else (row[1], row[0])
        for cat, col in COLORS.items():
            idx_in = [i for i, c in enumerate(cats) if c == cat and not clipped[i]]
            idx_out = [i for i, c in enumerate(cats) if c == cat and clipped[i]]
            if idx_in:
                xs, ys = zip(*[xy(Z[i]) for i in idx_in])
                ax.scatter(xs, ys, s=26, c=col, alpha=0.85, edgecolors="#fcfcfb", linewidths=0.6, zorder=3)
            if idx_out:
                xs, ys = zip(*[xy(Z[i]) for i in idx_out])
                ax.scatter(xs, ys, s=30, facecolors="none", edgecolors=col, linewidths=1.2,
                           marker="D", zorder=3)
        # default assistant: distinct marker
        dx, dy = xy(zd)
        ax.scatter([dx], [dy], s=140, marker="*", c=COLORS["AI / abstract"],
                   edgecolors="#0b0b0b", linewidths=0.9, zorder=5)
        ax.annotate("default", (dx, dy), xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color="#0b0b0b", fontweight="bold", zorder=6)
        # direct-label the extremes of PC1 (selective, not every point) — use unclipped order
        order = np.argsort(p["Z"][:, 0])
        low_end, high_end = list(order[:args.label_n]), list(order[-args.label_n:])
        for group, dense in ((low_end, False), (high_end, True)):
            for j, i in enumerate(group):
                lx, ly = xy(Z[i])
                # dense end (assistant cluster): fan labels out vertically so they don't overprint
                dy = (j - (len(group) - 1) / 2) * 9 if dense else 2
                ax.annotate(names[i], (lx, ly), xytext=(6, dy), textcoords="offset points",
                            fontsize=7, color="#52514e", zorder=6,
                            arrowprops=dict(arrowstyle="-", color="#d9d8d3", lw=0.5) if dense else None)
        ax.axhline(0, color="#d9d8d3", lw=0.8, zorder=1); ax.axvline(0, color="#d9d8d3", lw=0.8, zorder=1)
        pc1_lab = f"PC1  ({var[0]:.1%} of variance)  →  more assistant-like"
        pc2_lab = f"PC2  ({var[1]:.1%} of variance)"
        if args.clip_pc1:
            pc1_lab += f"   [clipped to central {1 - 2 * args.clip_pc1:.0%}; ◇ = pinned outliers]"
        if args.clip_pc2:
            pc2_lab += f"   [clipped to central {1 - 2 * args.clip_pc2:.0%}]"
        if args.flip_axes:
            ax.set_xlabel(pc1_lab, fontsize=10, color="#0b0b0b"); ax.set_ylabel(pc2_lab, fontsize=10, color="#0b0b0b")
        else:
            ax.set_xlabel(pc2_lab, fontsize=10, color="#0b0b0b"); ax.set_ylabel(pc1_lab, fontsize=10, color="#0b0b0b")
        ax.set_title(f"{p['label']}   —   layer {p['layer']}/{p['n_layers']}, {len(names)} personas",
                     fontsize=11, color="#0b0b0b", loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d9d8d3")
        ax.tick_params(colors="#52514e", labelsize=8)

    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=cat)
               for cat, c in COLORS.items()]
    handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor=COLORS["AI / abstract"],
                          markeredgecolor="#0b0b0b", markersize=13, label="default assistant"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Persona space: PCA of " + ("per-dimension standardized " if not args.no_standardize else "") + "role vectors (mid layer), coloured by persona category",
                 fontsize=13, color="#0b0b0b", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(args.out, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", args.out)
    # coverage report so category gaps are visible rather than silent
    all_roles = set().union(*(set(p["names"]) for p in panels))
    uncat = sorted(r for r in all_roles if r not in ROLE2CAT)
    if uncat:
        print(f"note: {len(uncat)} roles fell back to 'human (other)': {', '.join(uncat)}")


if __name__ == "__main__":
    main()
