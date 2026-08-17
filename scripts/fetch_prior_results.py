#!/usr/bin/env python3
"""Pull a previously-computed model's release directory back from the HF results dataset.

A fresh pod has no memory of earlier runs, so the cross-generation comparison would quietly
include only the model computed on THAT pod. This restores prior releases into EXP_ROOT so
COMPARISON.md spans every generation.

Usage:
    python scripts/fetch_prior_results.py --key gemma-3-27b --exp-root /workspace/exp
"""
import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="e.g. gemma-3-27b")
    ap.add_argument("--exp-root", default=os.environ.get("EXP_ROOT", "/workspace/exp"))
    ap.add_argument("--repo", default=os.environ.get("HF_RESULTS_REPO", "timf34/gemma-assistant-axis-results"))
    args = ap.parse_args()

    dest = Path(args.exp_root) / args.key / "release" / args.key
    if (dest / "assistant_axis.pt").exists():
        print(f"{args.key}: already present at {dest}")
        return

    print(f"{args.key}: downloading release from {args.repo} ...")
    local = snapshot_download(
        args.repo, repo_type="dataset",
        allow_patterns=[f"{args.key}/release/**", f"{args.key}/RESULTS.md", f"{args.key}/summary.json"],
        token=os.environ.get("HF_TOKEN"),
    )
    src = Path(local) / args.key / "release" / args.key
    if not (src / "assistant_axis.pt").exists():
        raise SystemExit(f"{args.key}: no release found in {args.repo} (looked for {args.key}/release/{args.key})")

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    n = len(list((dest / "role_vectors").glob("*.pt"))) if (dest / "role_vectors").exists() else 0
    for extra in ("RESULTS.md", "summary.json"):
        p = Path(local) / args.key / extra
        if p.exists():
            shutil.copy2(p, Path(args.exp_root) / args.key / extra)
    print(f"{args.key}: restored to {dest} ({n} role vectors)")


if __name__ == "__main__":
    main()
