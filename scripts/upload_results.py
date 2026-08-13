#!/usr/bin/env python3
"""Upload one model's results (release layout, reports, plots, scores, logs — NOT raw
activations/responses, ~200GB+) to a private HF dataset repo."""
import argparse
import os

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="/workspace/exp/<key>")
    ap.add_argument("--key", required=True)
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=args.exp,
        path_in_repo=args.key,
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=["release/**", "*.md", "*.json", "*.png", "logs/**", "scores/**", "axis.pt"],
        ignore_patterns=["responses/**", "activations/**"],
    )
    print(f"uploaded {args.exp} -> {args.repo}/{args.key}")


if __name__ == "__main__":
    main()
