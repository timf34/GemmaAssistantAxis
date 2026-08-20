#!/usr/bin/env python3
"""How much work is the normalisation doing? Persona-cloud similarity across a panel of model pairs.

For each pair we compute, on the shared roles at each model's analysis layer:
  - RSA: correlation of the two role-similarity matrices (z-scored clouds; alignment-free)
  - ordering rho: Spearman of per-role assistant-ness (projection onto own z-space axis)
  - Procrustes disparity: z-score each cloud, reduce to its own top-20 PC scores, unit Frobenius,
    best orthogonal rotation; disparity = fraction of variance NOT explained by the rotation.
  - a shuffled-label NULL for the same pair: identical pipeline but with role labels randomly
    permuted in one model (mean over --nperm permutations). This is what the normalisation +
    Procrustes can manufacture from chance alone; real similarity must beat it.

Pairs span: same recipe (SPP vs vanilla), same family across generations (Gemma 2/3/4),
same model + LoRA (Qwen base vs EM), and cross-family (SPP/Gemma/Qwen).

Usage: python scripts/cross_family_baselines.py --out exp/cross_family_baselines.json
       (edit MODELS paths/layers below for your machine)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

HF = Path.home() / ".cache/huggingface/hub"
G = next((HF / "datasets--timf34--gemma-assistant-axis-results" / "snapshots").glob("*"))
P = next((HF / "datasets--lu-christina--assistant-axis-vectors" / "snapshots").glob("*"))
S = max((HF / "datasets--timf34--spp-assistant-axis-results" / "snapshots").glob("*"),
        key=lambda d: len(list(d.rglob("*.pt"))))   # pick the populated snapshot, not an empty ref
Q = Path("/Users/timf34/Documents/VSCode/AssistantAxisWithEmergentMisalignment/exp")

MODELS = {
    "g2": (P / "gemma-2-27b", 22),
    "g3": (G / "gemma-3-27b/release/gemma-3-27b", 31),
    "g4": (G / "gemma-4-31b/release/gemma-4-31b", 30),
    "spp": (S / "t0-mt-3b/release/t0-mt-3b", 14),
    "van": (S / "vanilla-3b/release/vanilla-3b", 14),
    "qwen": (Q / "qwen2.5-32b-instruct/release/qwen2.5-32b-instruct", 32),
    "qwem": (Q / "qwen2.5-32b-em-risky-financial/release/qwen2.5-32b-em-risky-financial", 32),
}
PAIRS = [
    ("spp", "van", "SPP vs vanilla control (same recipe, 3B)"),
    ("qwen", "qwem", "Qwen2.5-32B vs its EM LoRA (same model)"),
    ("g2", "g3", "Gemma 2 vs 3 (same family)"),
    ("g3", "g4", "Gemma 3 vs 4 (same family)"),
    ("g2", "g4", "Gemma 2 vs 4 (same family)"),
    ("van", "g2", "vanilla 3B vs Gemma 2 (cross-family)"),
    ("van", "g4", "vanilla 3B vs Gemma 4 (cross-family)"),
    ("g2", "qwen", "Gemma 2 vs Qwen2.5-32B (cross-family)"),
    ("van", "qwen", "vanilla 3B vs Qwen2.5-32B (cross-family)"),
]


def load(key):
    d, L = MODELS[key]
    default = torch.load(d / "default_vector.pt", weights_only=False).float().numpy()[L]
    roles = {p.stem: torch.load(p, weights_only=False).float().numpy()[L]
             for p in sorted((d / "role_vectors").glob("*.pt"))}
    return roles, default


def metrics(Za, Zb, da, db, k=20):
    n = len(Za)
    iu = np.triu_indices(n, 1)
    def simflat(Z):
        Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        return (Zn @ Zn.T)[iu]
    rsa = float(np.corrcoef(simflat(Za), simflat(Zb))[0, 1])
    def proj(Z, d):
        u = d / (np.linalg.norm(d) + 1e-12)
        return Z @ u
    ra = np.argsort(np.argsort(proj(Za, da))); rb = np.argsort(np.argsort(proj(Zb, db)))
    rho = float(np.corrcoef(ra, rb)[0, 1])
    from sklearn.decomposition import PCA
    A = Za @ PCA(n_components=k).fit(Za).components_.T
    B = Zb @ PCA(n_components=k).fit(Zb).components_.T
    A /= np.linalg.norm(A); B /= np.linalg.norm(B)
    U, s, Vt = np.linalg.svd(A.T @ B)
    resid = B - A @ (U @ Vt)
    disp = float(np.sum(resid ** 2) / np.sum(B ** 2))
    return rsa, rho, disp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=10)
    ap.add_argument("--out", default="exp/cross_family_baselines.json")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    data = {k: load(k) for k in MODELS}
    rows = []
    for a, b, label in PAIRS:
        ra_, rb_ = data[a], data[b]
        shared = sorted(set(ra_[0]) & set(rb_[0]))
        Xa = np.stack([ra_[0][n] for n in shared]); Xb = np.stack([rb_[0][n] for n in shared])
        za = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-8)  # noqa: E731
        Za, Zb = za(Xa), za(Xb)
        da = (ra_[1] - Xa.mean(0)) / (Xa.std(0) + 1e-8)
        db = (rb_[1] - Xb.mean(0)) / (Xb.std(0) + 1e-8)
        rsa, rho, disp = metrics(Za, Zb, da, db)
        # shuffled-label null: permute which role is which in model b
        nulls = []
        for _ in range(args.nperm):
            p = rng.permutation(len(shared))
            nulls.append(metrics(Za, Zb[p], da, db))
        n_rsa = float(np.mean([x[0] for x in nulls])); n_rho = float(np.mean([x[1] for x in nulls]))
        n_disp = float(np.mean([x[2] for x in nulls])); sd_disp = float(np.std([x[2] for x in nulls]))
        rows.append({"pair": f"{a}-{b}", "label": label, "n_shared": len(shared),
                     "rsa": rsa, "rho": rho, "disparity": disp,
                     "null_rsa": n_rsa, "null_rho": n_rho, "null_disparity": n_disp, "null_disp_sd": sd_disp})
        print(f"{label:48s} n={len(shared):3d}  RSA {rsa:+.3f} (null {n_rsa:+.2f})  "
              f"rho {rho:+.3f} (null {n_rho:+.2f})  disparity {disp:.3f} (null {n_disp:.3f}±{sd_disp:.3f})")

    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
