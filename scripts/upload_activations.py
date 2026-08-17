#!/usr/bin/env python3
"""Upload a model's raw activations to a SEPARATE public HF dataset.

Activations are 57-220GB per model, so they get their own dataset rather than sitting next to
100MB of reports. Uses the resumable large-folder uploader: safe to re-run after an interruption.

Why keep them at all: the pipeline's vectors are per-ROLE means. Anything at the individual-
response level — projecting the score-2 "half-declined" responses onto the axis, within-role
variance, a different score threshold — needs the raw per-response activations, which the
vectors cannot give you. They ARE reproducible from the transcripts (~25 min of GPU, no judge
cost), but keeping them avoids renting a GPU just to re-derive something you already had.

Usage:
    python scripts/upload_activations.py --dir /workspace/exp/gemma-4-31b/activations \
        --key gemma-4-31b --repo timf34/gemma-assistant-axis-activations
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()

    d = Path(args.dir)
    files = sorted(d.glob("*.pt"))
    if not files:
        raise SystemExit(f"no .pt files in {d}")
    size_gb = sum(f.stat().st_size for f in files) / 1e9
    print(f"{args.key}: {len(files)} files, {size_gb:.1f} GB -> {args.repo}/{args.key}")

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    private = os.environ.get("HF_PRIVATE", "0") == "1"
    api.create_repo(args.repo, repo_type="dataset", private=private, exist_ok=True)
    try:
        api.update_repo_settings(repo_id=args.repo, repo_type="dataset", private=private)
    except Exception:  # noqa: BLE001
        pass

    # upload_large_folder: parallel, resumable (keeps a .cache/huggingface state dir in --dir),
    # and designed for exactly this many-GB, many-files case.
    api.upload_large_folder(
        folder_path=str(d), repo_id=args.repo, repo_type="dataset",
        path_in_repo=args.key, allow_patterns=["*.pt"], print_report=True,
    )
    print(f"{args.key}: activations uploaded ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
