#!/usr/bin/env python3
"""Upload one model's results (release layout, reports, plots, scores, logs — NOT raw
activations/responses, ~200GB+) to an HF dataset repo.

PUBLIC by default: private repos share a small LFS storage quota, and a 403 from that quota is
what stopped an earlier run's .pt tensors from ever being backed up. Set HF_PRIVATE=1 to override."""
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
    private = os.environ.get("HF_PRIVATE", "0") == "1"
    api.create_repo(args.repo, repo_type="dataset", private=private, exist_ok=True)
    # exist_ok=True does NOT change visibility of an existing repo, so set it explicitly.
    try:
        api.update_repo_settings(repo_id=args.repo, repo_type="dataset", private=private)
    except Exception as e:  # noqa: BLE001
        print(f"(could not set visibility: {e})")
    print(f"repo {args.repo} visibility: {'private' if private else 'PUBLIC'}")
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
