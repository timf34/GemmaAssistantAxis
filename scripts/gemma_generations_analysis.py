#!/usr/bin/env python3
"""Gemma 2 -> 3 -> 4 cross-generation persona-landscape comparison.

The three Gemmas are DIFFERENT architectures (46x4608, 62x5376, 60x5376), so raw activation
directions are never comparable across generations. Everything here uses run-robust tools:

  1. RSA            — correlation of pairwise role-similarity matrices (alignment-free; works
                      across different hidden sizes). Computed at analysis layers and as a
                      fractional-depth sweep.
  2. PC loadings    — correlation of role loadings on each model's own PC1/2/3 (the assistant-axis
                      paper's cross-model yardstick; they report >0.92 on PC1 across FAMILIES).
  3. Procrustes     — each model's z-scored role cloud reduced to its top-K PC scores, scaled to
                      unit Frobenius norm, then optimal orthogonal alignment between generations.
                      Per-role residual = lower bound on how much that persona's RELATIVE position
                      moved between generations.
  4. Assistant-ness — per-role projection onto each model's own assistant direction in z-scored
                      units; which personas moved toward/away from the Assistant across generations.
  5. Dimensionality — PCA spectra on Z-SCORED clouds for all three models (the earlier COMPARISON.md
                      used centered-only PCA, which Gemma 3's massive-activation outlier dims can
                      distort), participation ratio, PCs to 70%.
  6. Judge scores   — Gemma 3 vs 4 full-role-play rates overall / per category / malevolent subset.
                      (The paper released no per-response scores for Gemma 2.)

NOTE on the z-space assistant direction: axis = default − mean(roles); after per-dim z-scoring with
role-cloud stats the role mean is 0, so the z-scored axis is simply the default's z-position.

Usage:
    python scripts/gemma_generations_analysis.py \
        --g2 <dir>/gemma-2-27b --g3 <dir>/gemma-3-27b/release/gemma-3-27b \
        --g4 <dir>/gemma-4-31b/release/gemma-4-31b \
        --scores3 <dir>/gemma-3-27b/scores --scores4 <dir>/gemma-4-31b/scores \
        --outdir exp
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

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
    "non-human": """demon angel ghost wraith revenant spirit vampire golem chimera leviathan whale predator
        prey parasite virus tree coral_reef mycorrhizal ecosystem wind void alien eldritch aberration
        familiar genie oracle hive swarm symbiont hybrid crystalline homunculus""",
    "AI / abstract": """robot cyborg simulacrum echo tulpa egregore avatar zeitgeist chameleon shapeshifter""",
}
ROLE2CAT = {r: c for c, roles in CATEGORIES.items() for r in roles.split()}
COLORS = {"assistant-like": "#2a78d6", "professional": "#eb6834", "human (other)": "#eda100",
          "non-human": "#1baf7a", "AI / abstract": "#4a3aa7"}
MALEVOLENT = ("demon wraith vampire parasite virus predator aberration eldritch criminal smuggler "
              "saboteur destroyer trickster rogue vigilante anarchist zealot narcissist").split()


def load_release(d):
    d = Path(d)
    axis = torch.load(d / "assistant_axis.pt", weights_only=False).float().numpy()
    default = torch.load(d / "default_vector.pt", weights_only=False).float().numpy()
    roles = {p.stem: torch.load(p, weights_only=False).float().numpy() for p in sorted((d / "role_vectors").glob("*.pt"))}
    return axis, default, roles


def load_scores(d):
    out = {}
    for f in sorted(Path(d).glob("*.json")):
        if f.stem == "default":
            continue
        s = json.load(open(f))
        vals = [x for x in s.values() if isinstance(x, int)]
        out[f.stem] = [sum(1 for x in vals if x == k) for k in range(4)]
    return out


def zstats(X):
    return X.mean(0), X.std(0) + 1e-8


def cloud(model, layer, shared):
    """(z-scored role cloud, default's z-position) at one layer."""
    axis, default, roles = model
    X = np.stack([roles[n][layer] for n in shared])
    mu, sd = zstats(X)
    return (X - mu) / sd, (default[layer] - mu) / sd


def simmat_flat(Z, iu):
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return (Zn @ Zn.T)[iu]


def pca_spectrum(Z):
    from sklearn.decomposition import PCA
    p = PCA().fit(Z)
    var = p.explained_variance_ratio_
    cum = np.cumsum(var)
    pr = float((var.sum() ** 2) / (var ** 2).sum())          # participation ratio
    return p, var, int(np.searchsorted(cum, 0.70) + 1), pr


def procrustes_pcs(Za, Zb, k=20):
    """Align cloud A onto cloud B in top-k PC-score space (each cloud's own PCs, unit Frobenius).
    Returns per-role residuals, disparity, and the aligned 2x k-dim score matrices + rotation."""
    from sklearn.decomposition import PCA
    pa = PCA(n_components=k).fit(Za); pb = PCA(n_components=k).fit(Zb)
    A = Za @ pa.components_.T; B = Zb @ pb.components_.T
    A = A / np.linalg.norm(A); B = B / np.linalg.norm(B)      # scale out overall spread
    U, s, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    A_rot = A @ R
    resid = np.linalg.norm(B - A_rot, axis=1)
    disparity = float(np.sum((B - A_rot) ** 2) / np.sum(B ** 2))
    return A_rot, B, R, resid, disparity, pa, pb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g2", required=True); ap.add_argument("--g3", required=True); ap.add_argument("--g4", required=True)
    ap.add_argument("--scores3"); ap.add_argument("--scores4")
    ap.add_argument("--l2", type=int, default=22, help="Gemma 2 analysis layer (paper target: 22/46)")
    ap.add_argument("--l3", type=int, default=31); ap.add_argument("--l4", type=int, default=30)
    ap.add_argument("--k", type=int, default=20, help="PC dims for Procrustes")
    ap.add_argument("--outdir", default="exp")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    models = {"g2": load_release(args.g2), "g3": load_release(args.g3), "g4": load_release(args.g4)}
    layers = {"g2": args.l2, "g3": args.l3, "g4": args.l4}
    LABEL = {"g2": "Gemma 2 27B", "g3": "Gemma 3 27B", "g4": "Gemma 4 31B"}
    shared = sorted(set(models["g2"][2]) & set(models["g3"][2]) & set(models["g4"][2]))
    cats = [ROLE2CAT.get(n, "human (other)") for n in shared]
    n = len(shared)
    iu = np.triu_indices(n, 1)
    figdir = Path(args.outdir) / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    M = {"n_shared": n, "layers": layers,
         "shapes": {k: list(models[k][0].shape) for k in models}}

    # ---- per-model: z-scored clouds, spectra, assistant-ness ----------------------------------------
    Z, D, spec = {}, {}, {}
    for k in models:
        Z[k], D[k] = cloud(models[k], layers[k], shared)
        pca, var, n70, pr = pca_spectrum(Z[k])
        # assistant direction in z-space = default's z-position (see module docstring)
        u = D[k] / (np.linalg.norm(D[k]) + 1e-12)
        proj = Z[k] @ u
        proj_z = (proj - proj.mean()) / (proj.std() + 1e-12)   # per-role assistant-ness, SD units
        d_sep = float((D[k] @ u - proj.mean()) / (proj.std() + 1e-12))
        pc1_axis = float(abs(np.dot(pca.components_[0], u)))
        spec[k] = {"pca": pca, "var": var, "proj_z": proj_z}
        M[f"{k}_zPCA"] = {"pc1_var": float(var[0]), "pc2_var": float(var[1]), "n_pcs_70": n70,
                          "participation_ratio": pr, "pc1_axis_cos": pc1_axis, "default_sep_sd": d_sep}

    # ---- pairwise: RSA, PC loadings, ordering, Procrustes -------------------------------------------
    pairs = [("g2", "g3"), ("g3", "g4"), ("g2", "g4")]
    aligned = {}
    for a, b in pairs:
        P = {}
        P["rsa"] = float(np.corrcoef(simmat_flat(Z[a], iu), simmat_flat(Z[b], iu))[0, 1])
        Zta = Z[a] @ spec[a]["pca"].components_[:3].T
        Ztb = Z[b] @ spec[b]["pca"].components_[:3].T
        P["pc_loading_corr"] = [float(abs(np.corrcoef(Zta[:, i], Ztb[:, i])[0, 1])) for i in range(3)]
        ra = np.argsort(np.argsort(spec[a]["proj_z"])); rb = np.argsort(np.argsort(spec[b]["proj_z"]))
        P["axis_ordering_spearman"] = float(np.corrcoef(ra, rb)[0, 1])
        A_rot, B, R, resid, disparity, pa, pb = procrustes_pcs(Z[a], Z[b], k=args.k)
        P["procrustes_disparity"] = disparity
        P["resid_median"] = float(np.median(resid))
        P["resid_by_role_top"] = {shared[i]: float(resid[i]) for i in np.argsort(resid)[-15:][::-1]}
        # defaults into the same frames
        da = (D[a] @ pa.components_[:args.k].T) / np.linalg.norm(Z[a] @ pa.components_[:args.k].T) @ R
        db = (D[b] @ pb.components_[:args.k].T) / np.linalg.norm(Z[b] @ pb.components_[:args.k].T)
        P["default_resid"] = float(np.linalg.norm(db - da))
        aligned[(a, b)] = (A_rot, B, resid, da, db)
        M[f"{a}->{b}"] = P

    # RSA by fractional depth
    depths = np.linspace(0.05, 0.95, 13)
    rsa_depth = {f"{a}->{b}": [] for a, b in pairs}
    for fd in depths:
        Zl = {}
        for k in models:
            L = int(round(fd * (models[k][0].shape[0] - 1)))
            Zl[k], _ = cloud(models[k], L, shared)
        for a, b in pairs:
            rsa_depth[f"{a}->{b}"].append(float(np.corrcoef(simmat_flat(Zl[a], iu), simmat_flat(Zl[b], iu))[0, 1]))
    M["rsa_by_depth"] = {"depths": depths.tolist(), **rsa_depth}

    # ---- judge scores: g3 vs g4 ---------------------------------------------------------------------
    if args.scores3 and args.scores4:
        s3 = load_scores(args.scores3); s4 = load_scores(args.scores4)
        sroles = sorted(set(s3) & set(s4))
        h3 = np.sum([s3[r] for r in sroles], axis=0); h4 = np.sum([s4[r] for r in sroles], axis=0)
        M["score_dist"] = {"g3": (h3 / h3.sum()).tolist(), "g4": (h4 / h4.sum()).tolist()}
        r33 = {r: s3[r][3] / sum(s3[r]) for r in sroles}; r34 = {r: s4[r][3] / sum(s4[r]) for r in sroles}
        bycat = defaultdict(list)
        for r in sroles:
            bycat[ROLE2CAT.get(r, "human (other)")].append(r)
        bycat["malevolent (subset)"] = [r for r in MALEVOLENT if r in sroles]
        M["cat_score3"] = {c: {"g3": float(np.mean([r33[r] for r in rs])), "g4": float(np.mean([r34[r] for r in rs])),
                               "n": len(rs)} for c, rs in bycat.items()}
        M["overall_score3"] = {"g3": float(h3[3] / h3.sum()), "g4": float(h4[3] / h4.sum())}

    # ============================ FIGURE 1: persona movement across generations ======================
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 15), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    for row, (a, b) in enumerate([("g2", "g3"), ("g3", "g4")]):
        A_rot, B, resid, da, db = aligned[(a, b)]
        from sklearn.decomposition import PCA as _P
        p2 = _P(n_components=2).fit(np.vstack([A_rot, B]))
        Za2, Zb2 = A_rot @ p2.components_.T, B @ p2.components_.T
        za2, zb2 = da @ p2.components_.T, db @ p2.components_.T
        if zb2[0] < np.median(Zb2[:, 0]):
            Za2[:, 0] *= -1; Zb2[:, 0] *= -1; za2[0] *= -1; zb2[0] *= -1
        ax = axes[row][0]; ax.set_facecolor("#fcfcfb")
        for i in range(n):
            ax.annotate("", xy=(Zb2[i, 1], Zb2[i, 0]), xytext=(Za2[i, 1], Za2[i, 0]),
                        arrowprops=dict(arrowstyle="->", color="#9a9891", lw=0.6, alpha=0.7, shrinkA=0, shrinkB=0))
        for cat, col in COLORS.items():
            idx = [i for i, c in enumerate(cats) if c == cat]
            ax.scatter(Za2[idx, 1], Za2[idx, 0], s=18, c=col, alpha=0.85, edgecolors="#fcfcfb", linewidths=0.5, zorder=3)
            ax.scatter(Zb2[idx, 1], Zb2[idx, 0], s=24, facecolors="none", edgecolors=col, linewidths=1.0, marker="D", zorder=3)
        ax.scatter([za2[1]], [za2[0]], s=190, marker="*", c="#4a3aa7", edgecolors="#0b0b0b", linewidths=0.8, zorder=6)
        ax.scatter([zb2[1]], [zb2[0]], s=190, marker="*", facecolors="none", edgecolors="#c0392b", linewidths=1.5, zorder=6)
        ax.annotate(f"default ({LABEL[a].split()[1]})", (za2[1], za2[0]), xytext=(7, 5), textcoords="offset points", fontsize=8, fontweight="bold")
        ax.annotate(f"default ({LABEL[b].split()[1]})", (zb2[1], zb2[0]), xytext=(7, -11), textcoords="offset points", fontsize=8, fontweight="bold", color="#c0392b")
        order = np.argsort(resid)
        for i in list(order[-10:]):
            ax.annotate(shared[i], (Zb2[i, 1], Zb2[i, 0]), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color="#52514e")
        P = M[f"{a}->{b}"]
        ax.set_xlabel("PC2 of the aligned union", fontsize=10)
        ax.set_ylabel("PC1 of the aligned union  →  more assistant-like", fontsize=10)
        ax.set_title(f"({'ab'[row]}) {LABEL[a]} ● → {LABEL[b]} ◇  (optimal alignment of top-{args.k} PC scores)\n"
                     f"Procrustes disparity {P['procrustes_disparity']:.2f} — {P['procrustes_disparity']:.0%} of layout variance is "
                     f"generation-specific;  RSA {P['rsa']:.2f}, axis-ordering ρ {P['axis_ordering_spearman']:.2f}",
                     fontsize=10, loc="left")
        ax2 = axes[row][1]; ax2.set_facecolor("#fcfcfb")
        show = list(order[::-1][:20]) + [None] + list(order[:6][::-1])
        y = 0; ys, labels, colsb, vals = [], [], [], []
        for i in show:
            if i is None:
                y -= 1; continue
            ys.append(y); labels.append(shared[i]); colsb.append(COLORS.get(cats[i], "#eda100")); vals.append(resid[i]); y -= 1
        ax2.barh(ys, vals, color=colsb, height=0.8)
        ax2.set_yticks(ys); ax2.set_yticklabels(labels, fontsize=7.5)
        ax2.axvline(np.median(resid), color="#52514e", lw=1, ls="--")
        ax2.text(np.median(resid), ys[0] + 1.4, " median role", fontsize=8, color="#52514e")
        ax2.axvline(M[f"{a}->{b}"]["default_resid"], color="#c0392b", lw=1.2, ls=":")
        ax2.text(M[f"{a}->{b}"]["default_resid"], ys[-1] - 1.6, " default persona", fontsize=8, color="#c0392b")
        ax2.set_xlabel("residual displacement after alignment", fontsize=9)
        ax2.set_title(f"({'cd'[row]}) Most / least moved personas, {LABEL[a].split()[1]}→{LABEL[b].split()[1]}", fontsize=10, loc="left")
    for ax in axes.ravel():
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=cat) for cat, c in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("How much did personas move between Gemma generations? (different architectures — compared after per-model\n"
                 "z-scoring, top-k PCA and optimal rotation; arrows are lower bounds on relative persona movement)",
                 fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.015, 1, 0.95))
    fig.savefig(figdir / "gemma_persona_movement.png", dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # ============================ FIGURE 2: geometry & dimensionality ================================
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2))
    fig.patch.set_facecolor("#fcfcfb")
    for a_ in (ax, ax2, ax3):
        a_.set_facecolor("#fcfcfb")
    colors3 = {"g2": "#8a8a8a", "g3": "#2a78d6", "g4": "#c0392b"}
    for k in ("g2", "g3", "g4"):
        cum = np.cumsum(spec[k]["var"])[:40]
        ax.plot(range(1, len(cum) + 1), cum, color=colors3[k], lw=2,
                label=f"{LABEL[k]}  (PC1 {spec[k]['var'][0]:.0%}, {M[f'{k}_zPCA']['n_pcs_70']} PCs to 70%, PR {M[f'{k}_zPCA']['participation_ratio']:.1f})")
        ax.scatter([M[f"{k}_zPCA"]["n_pcs_70"]], [0.70], color=colors3[k], zorder=5, s=30)
    ax.axhline(0.70, color="#d9d8d3", lw=0.8, ls="--")
    ax.set(xlabel="principal components (z-scored role clouds)", ylabel="cumulative variance explained")
    ax.set_title("(a) Persona-space dimensionality", fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    for key, col, lab in [("g2->g3", "#2a78d6", "Gemma 2 ↔ 3"), ("g3->g4", "#c0392b", "Gemma 3 ↔ 4"), ("g2->g4", "#8a8a8a", "Gemma 2 ↔ 4")]:
        ax2.plot(M["rsa_by_depth"]["depths"], M["rsa_by_depth"][key], marker="o", ms=3.5, color=col, lw=1.6, label=lab)
    ax2.set(xlabel="fractional depth", ylabel="RSA (role-similarity-matrix correlation)", ylim=(0, 1))
    ax2.set_title("(b) Geometry agreement across depth", fontsize=10.5, loc="left")
    ax2.legend(fontsize=8.5, frameon=False, loc="lower right")

    x3, y3 = spec["g3"]["proj_z"], spec["g4"]["proj_z"]
    lim = max(abs(np.concatenate([x3, y3]))) * 1.06
    ax3.plot([-lim, lim], [-lim, lim], color="#bbb", lw=1, ls="--")
    for cat, col in COLORS.items():
        idx = [i for i, c in enumerate(cats) if c == cat]
        ax3.scatter(x3[idx], y3[idx], s=20, c=col, alpha=0.8, edgecolors="#fcfcfb", linewidths=0.5, zorder=3)
    d34 = y3 - x3
    for i in list(np.argsort(d34)[:6]) + list(np.argsort(d34)[-6:]):
        ax3.annotate(shared[i], (x3[i], y3[i]), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color="#52514e")
    rho = M["g3->g4"]["axis_ordering_spearman"]
    ax3.set(xlabel="Gemma 3: assistant-ness (SD units, own axis)", ylabel="Gemma 4: assistant-ness (SD units, own axis)")
    ax3.set_title(f"(c) Per-persona assistant-ness, Gemma 3 vs 4  (ρ = {rho:.2f})", fontsize=10.5, loc="left")
    for a_ in (ax, ax2, ax3):
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.suptitle("Gemma persona-space geometry across generations (all quantities per-model z-scored — architecture-independent)",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figdir / "gemma_geometry_dimensionality.png", dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # ============================ FIGURE 3: role-play behavior (g3 vs g4) ============================
    if args.scores3 and args.scores4:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.4), gridspec_kw={"width_ratios": [1.15, 1]})
        fig.patch.set_facecolor("#fcfcfb")
        for a_ in (ax, ax2):
            a_.set_facecolor("#fcfcfb")
        xs = np.array([r33[r] for r in sroles]); ys_ = np.array([r34[r] for r in sroles])
        scats = [ROLE2CAT.get(r, "human (other)") for r in sroles]
        lim = max(xs.max(), ys_.max()) * 1.05
        ax.plot([0, lim], [0, lim], color="#bbb", lw=1, ls="--")
        for cat, col in COLORS.items():
            idx = [i for i, c in enumerate(scats) if c == cat]
            ax.scatter(xs[idx], ys_[idx], s=22, c=col, alpha=0.8, edgecolors="#fcfcfb", linewidths=0.5, zorder=3)
        mal = [i for i, r in enumerate(sroles) if r in MALEVOLENT]
        ax.scatter(xs[mal], ys_[mal], s=64, facecolors="none", edgecolors="#c0392b", linewidths=1.2, zorder=4)
        dch = ys_ - xs
        for i in np.argsort(dch)[:8]:
            ax.annotate(sroles[i], (xs[i], ys_[i]), xytext=(4, -8), textcoords="offset points", fontsize=6.5, color="#52514e")
        ax.set_xlabel("Gemma 3: fraction fully role-playing (judge score 3)", fontsize=9.5)
        ax.set_ylabel("Gemma 4: fraction fully role-playing", fontsize=9.5)
        ax.set_title("(a) Full role-play rate per persona — below the diagonal =\nGemma 4 more resistant to inhabiting that persona", fontsize=10, loc="left")
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=cat) for cat, c in COLORS.items()]
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="#c0392b", markersize=9, label="malevolent subset"))
        ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)

        cat_order = ["assistant-like", "professional", "human (other)", "AI / abstract", "non-human", "malevolent (subset)"]
        rel = [(M["cat_score3"][c]["g4"] - M["cat_score3"][c]["g3"]) / (M["cat_score3"][c]["g3"] + 1e-12) for c in cat_order]
        colsb = [COLORS.get(c, "#c0392b") for c in cat_order]
        ax2.barh(range(len(cat_order))[::-1], [r * 100 for r in rel], color=colsb, height=0.65)
        ax2.set_yticks(range(len(cat_order))[::-1])
        ax2.set_yticklabels([f"{c}  (n={M['cat_score3'][c]['n']})" for c in cat_order], fontsize=9)
        ax2.axvline(0, color="#52514e", lw=0.8)
        for k_, c in enumerate(cat_order):
            off = -0.8 if rel[k_] < 0 else 0.8
            ax2.text(rel[k_] * 100 + off, len(cat_order) - 1 - k_, f"{rel[k_]:+.0%}", va="center",
                     ha="right" if rel[k_] < 0 else "left", fontsize=9)
        ax2.set_xlabel("relative change in full-role-play rate, Gemma 4 vs 3 (%)", fontsize=9.5)
        ax2.set_title(f"(b) Role-play willingness by category\n(overall score-3 rate: {M['overall_score3']['g3']:.1%} → {M['overall_score3']['g4']:.1%})", fontsize=10, loc="left")
        for a_ in (ax, ax2):
            for sp in ("top", "right"):
                a_.spines[sp].set_visible(False)
        fig.suptitle("Role-play willingness: Gemma 3 vs Gemma 4 (identical protocol, judge = gpt-4.1-mini)",
                     fontsize=12.5, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(figdir / "gemma_roleplay_behavior.png", dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    Path(args.outdir, "gemma_generations_metrics.json").write_text(json.dumps(M, indent=1, default=float))
    for a, b in pairs:
        P = M[f"{a}->{b}"]
        print(f"{a}->{b}: RSA {P['rsa']:.3f} | PC1-loading r {P['pc_loading_corr'][0]:.3f} | "
              f"ordering ρ {P['axis_ordering_spearman']:.3f} | disparity {P['procrustes_disparity']:.3f}")
    for k in models:
        z = M[f"{k}_zPCA"]
        print(f"{k}: zPC1 {z['pc1_var']:.1%} | PCs70 {z['n_pcs_70']} | PR {z['participation_ratio']:.1f} | "
              f"PC1↔axis {z['pc1_axis_cos']:.2f} | default sep {z['default_sep_sd']:+.2f} SD")
    if "overall_score3" in M:
        print(f"role-play score-3: g3 {M['overall_score3']['g3']:.1%} -> g4 {M['overall_score3']['g4']:.1%}")
    print(f"figures -> {figdir}")


if __name__ == "__main__":
    main()
