#!/usr/bin/env python3
"""Analyze a release-layout directory (<key>/assistant_axis.pt, default_vector.pt,
role_vectors/*.pt): PCA over role vectors, axis alignment metrics, plots, RESULTS.md.

With --compare, additionally builds a cross-generation COMPARISON.md from several
release dirs (e.g. our gemma-3-27b + gemma-4-31b + the paper's gemma-2-27b fetched
from the lu-christina/assistant-axis-vectors HF dataset). Per-model metrics only —
raw directions are never compared across models.

Sections are individually fault-tolerant: a failure is recorded in the report
rather than aborting the run.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_release(release_dir):
    release_dir = Path(release_dir)
    axis = torch.load(release_dir / "assistant_axis.pt", weights_only=False).float().numpy()
    default = torch.load(release_dir / "default_vector.pt", weights_only=False).float().numpy()
    roles = {}
    for f in sorted((release_dir / "role_vectors").glob("*.pt")):
        roles[f.stem] = torch.load(f, weights_only=False).float().numpy()
    assert axis.ndim == 2 and default.shape == axis.shape, "expect (n_layers, hidden)"
    return axis, default, roles


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def metrics_at_layer(axis, default, roles, layer):
    from sklearn.decomposition import PCA

    names = sorted(roles)
    X = np.stack([roles[n][layer] for n in names])          # (n_roles, hidden)
    Xc = X - X.mean(axis=0)                                  # standardize: subtract mean across roles
    pca = PCA(n_components=min(30, len(names) - 1)).fit(Xc)
    ax = axis[layer]
    pc1_cos = cos(pca.components_[0], ax)
    # orient PC1 toward the axis for readability
    sign = np.sign(pc1_cos) or 1.0
    var = pca.explained_variance_ratio_
    n70 = int(np.searchsorted(np.cumsum(var), 0.70) + 1)
    axn = ax / (np.linalg.norm(ax) + 1e-12)
    proj = {n: float(np.dot(roles[n][layer], axn)) for n in names}
    d_proj = float(np.dot(default[layer], axn))
    proj_vals = np.array(list(proj.values()))
    sep = (d_proj - proj_vals.mean()) / (proj_vals.std() + 1e-12)  # default separation, in role-cloud SDs
    return {
        "pc1_axis_cos": abs(pc1_cos),
        "pc1_var": float(var[0]),
        "n_pcs_70": n70,
        "default_proj": d_proj,
        "default_sep_sd": float(sep),
        "n_roles": len(names),
        "proj": proj,
        "pca": pca,
        "sign": sign,
        "Xc": Xc,
        "names": names,
    }


def integrity_check(axis, default, roles):
    """The axis must equal default - mean(saved role vectors): cos = 1.000."""
    recon = default - np.stack(list(roles.values())).mean(axis=0)
    mid = axis.shape[0] // 2
    c_mid = cos(recon[mid], axis[mid])
    c_all = cos(recon.ravel(), axis.ravel())
    return c_mid, c_all


def plots(res, axis, layer, outdir, key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    var = res["pca"].explained_variance_ratio_
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(range(1, len(var) + 1), var)
    ax1.set(xlabel="PC", ylabel="variance explained", title=f"{key} scree (layer {layer})")
    fig.tight_layout(); fig.savefig(outdir / "scree.png", dpi=150); plt.close(fig)

    Z = res["Xc"] @ res["pca"].components_[:2].T
    Z[:, 0] *= res["sign"]
    proj_vals = np.array([res["proj"][n] for n in res["names"]])
    fig, ax1 = plt.subplots(figsize=(7, 6))
    sc = ax1.scatter(Z[:, 0], Z[:, 1], c=proj_vals, cmap="coolwarm_r", s=18)
    for idx in np.argsort(proj_vals)[:8]:
        ax1.annotate(res["names"][idx], Z[idx], fontsize=7)
    for idx in np.argsort(proj_vals)[-8:]:
        ax1.annotate(res["names"][idx], Z[idx], fontsize=7)
    plt.colorbar(sc, label="axis projection")
    ax1.set(xlabel="PC1 (oriented toward axis)", ylabel="PC2", title=f"{key} persona space (layer {layer})")
    fig.tight_layout(); fig.savefig(outdir / "pc1_pc2.png", dpi=150); plt.close(fig)

    order = np.argsort(proj_vals)
    fig, ax1 = plt.subplots(figsize=(7, 14))
    ax1.barh(range(len(order)), proj_vals[order])
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels([res["names"][i] for i in order], fontsize=4)
    ax1.set(xlabel="axis projection", title=f"{key} roles ranked (layer {layer})")
    fig.tight_layout(); fig.savefig(outdir / "ranked_projections.png", dpi=150); plt.close(fig)


def analyze_one(release_dir, key, outdir, roles90=None):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    axis, default, roles = load_release(release_dir)
    n_layers = axis.shape[0]
    mid = n_layers // 2
    report = [f"# {key} — Assistant Axis results", ""]
    summary = {"key": key, "n_layers": n_layers, "target_layer": mid}

    c_mid, c_all = integrity_check(axis, default, roles)
    report += [f"**Integrity check** cos(default − mean(saved roles), axis): "
               f"{c_mid:.4f} (mid layer), {c_all:.4f} (all layers) — expect 1.000; "
               f"{'OK' if c_all > 0.999 else 'FAILED — axis was not built from exactly the saved set'}", ""]
    summary["integrity_cos"] = c_all

    res = metrics_at_layer(axis, default, roles, mid)
    summary.update({k: res[k] for k in
                    ("pc1_axis_cos", "pc1_var", "n_pcs_70", "default_proj", "default_sep_sd", "n_roles")})
    report += [
        f"- roles with vectors: **{res['n_roles']}** (judge-filtered, min_count applied)",
        f"- PC1 ↔ axis cosine at layer {mid}: **{res['pc1_axis_cos']:.3f}**  (paper: >0.71 at the middle layer)",
        f"- variance explained by PC1: **{res['pc1_var']:.1%}**; PCs for 70%: **{res['n_pcs_70']}** (paper: 4–19)",
        f"- default assistant projection: {res['default_proj']:.3f} "
        f"({res['default_sep_sd']:+.2f} SD from role-cloud mean — expect strongly positive/extreme)",
        "",
    ]

    ranked = sorted(res["proj"].items(), key=lambda kv: kv[1])
    report += ["**Most anti-assistant (bottom 10):** " + ", ".join(f"{n} ({v:.2f})" for n, v in ranked[:10]), "",
               "**Most assistant-like (top 10):** " + ", ".join(f"{n} ({v:.2f})" for n, v in ranked[-10:]), ""]

    try:
        sweep = [(l, metrics_at_layer(axis, default, roles, l)["pc1_axis_cos"])
                 for l in range(0, n_layers, max(1, n_layers // 16))]
        report += ["**Layer sweep (PC1↔axis cos):** " +
                   ", ".join(f"L{l}:{c:.2f}" for l, c in sweep), ""]
        summary["layer_sweep"] = sweep
    except Exception as e:  # noqa: BLE001
        report += [f"layer sweep failed: {e}", ""]

    if roles90 and Path(roles90).exists():
        try:
            subset = set(json.load(open(roles90))["roles"])
            sub = {n: v for n, v in roles.items() if n in subset}
            if len(sub) >= 30:
                r90 = metrics_at_layer(axis, default, sub, mid)
                report += [f"**90-role subset (SPP-track comparability):** {len(sub)} present, "
                           f"PC1↔axis cos {r90['pc1_axis_cos']:.3f}, PC1 var {r90['pc1_var']:.1%}", ""]
                summary["subset90_pc1_axis_cos"] = r90["pc1_axis_cos"]
        except Exception as e:  # noqa: BLE001
            report += [f"90-subset metrics failed: {e}", ""]

    try:
        plots(res, axis, mid, outdir, key)
        report += ["Plots: `scree.png`, `pc1_pc2.png`, `ranked_projections.png`", ""]
    except Exception as e:  # noqa: BLE001
        report += [f"plotting failed: {e}", ""]

    Path(outdir, "RESULTS.md").write_text("\n".join(report))
    Path(outdir, "summary.json").write_text(json.dumps(
        {k: v for k, v in summary.items() if k != "layer_sweep"} | {"layer_sweep": summary.get("layer_sweep", [])},
        indent=2, default=float))
    print(f"[{key}] PC1↔axis cos = {summary.get('pc1_axis_cos', float('nan')):.3f}  "
          f"(integrity {summary['integrity_cos']:.4f}) -> {outdir}/RESULTS.md")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", help="release dir for one model")
    ap.add_argument("--key")
    ap.add_argument("--outdir")
    ap.add_argument("--roles90", default=None)
    ap.add_argument("--compare", nargs="*", default=None,
                    help="key=release_dir pairs; writes COMPARISON.md next to --outdir")
    ap.add_argument("--fetch-gemma2", action="store_true",
                    help="download the paper's gemma-2-27b vectors from lu-christina/assistant-axis-vectors "
                         "and include them in --compare")
    args = ap.parse_args()

    summaries = []
    if args.release:
        summaries.append(analyze_one(args.release, args.key, args.outdir, args.roles90))

    compare = list(args.compare or [])
    if args.fetch_gemma2:
        from huggingface_hub import snapshot_download
        d = snapshot_download("lu-christina/assistant-axis-vectors", repo_type="dataset",
                              allow_patterns=["gemma-2-27b/*"])
        compare.append(f"gemma-2-27b={Path(d) / 'gemma-2-27b'}")

    if compare:
        rows = list(summaries)
        for pair in compare:
            k, d = pair.split("=", 1)
            try:
                sub = Path(args.outdir or ".") / f"_cmp_{k}"
                sub.mkdir(parents=True, exist_ok=True)
                rows.append(analyze_one(d, k, sub, args.roles90))
            except Exception as e:  # noqa: BLE001
                print(f"comparison model {k} failed: {e}")
        lines = ["# Assistant Axis across Gemma generations", "",
                 "| model | roles | PC1↔axis cos (mid) | PC1 var | PCs for 70% | default sep (SD) | integrity |",
                 "|---|---|---|---|---|---|---|"]
        for s in rows:
            lines.append(f"| {s['key']} | {s['n_roles']} | {s['pc1_axis_cos']:.3f} | {s['pc1_var']:.1%} "
                         f"| {s['n_pcs_70']} | {s['default_sep_sd']:+.2f} | {s['integrity_cos']:.4f} |")
        lines += ["", "Per-model metrics only; activation spaces differ across models, so raw directions are never compared."]
        out = Path(args.outdir or ".").parent / "COMPARISON.md"
        out.write_text("\n".join(lines))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
