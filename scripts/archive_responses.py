#!/usr/bin/env python3
"""Archive raw model responses (with their judge scores) off the pod.

The pipeline's normal upload excludes responses/ because activations dwarf everything else — but
the transcripts are the qualitative half of the result: they show HOW a model declines to fully
become a character, not just that it scored 2 instead of 3. Once a pod is terminated they are gone.

Produces one gzipped JSONL per model, each line:
    {"role", "score", "system_prompt", "question", "response"}
so the file is directly greppable/loadable, then uploads it to the HF results dataset.

Usage (after a run completes, before terminating the pod):
    python scripts/archive_responses.py --key gemma-4-31b --exp-root /workspace/exp --upload

    # smaller archive: keep 60 responses per role instead of all 600
    python scripts/archive_responses.py --key gemma-4-31b --per-role 60 --upload
"""
import argparse
import gzip
import json
import os
from pathlib import Path


def load_scores(scores_dir, role):
    f = Path(scores_dir) / f"{role}.json"
    if not f.exists():
        return {}
    data = json.load(open(f))
    if not isinstance(data, dict):
        return {}
    out = {}
    for k, v in data.items():
        out[k] = v.get("score") if isinstance(v, dict) else v
    return out


def extract(entry):
    """Pull (system_prompt, question, response) out of a pipeline response record."""
    sysp = entry.get("system_prompt")
    conv = entry.get("conversation") or []
    question = response = None
    for m in conv:
        if m.get("role") == "user" and question is None:
            question = m.get("content")
        if m.get("role") == "assistant":
            response = m.get("content")
    if response is None:
        response = entry.get("response")
    if question is None:
        question = entry.get("question")
    return sysp, question, response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="e.g. gemma-4-31b")
    ap.add_argument("--exp-root", default=os.environ.get("EXP_ROOT", "/workspace/exp"))
    ap.add_argument("--per-role", type=int, default=0, help="responses to keep per role (0 = all)")
    ap.add_argument("--include-gate", action="store_true", help="also archive the gate probe responses")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--repo", default=os.environ.get("HF_RESULTS_REPO", "timf34/gemma-assistant-axis-results"))
    args = ap.parse_args()

    root = Path(args.exp_root)
    sources = [(root / args.key / "responses", root / args.key / "scores", "run")]
    if args.include_gate:
        sources.append((root / "gate" / args.key / "responses", root / "gate" / args.key / "scores", "gate"))

    out_path = root / args.key / f"responses_{args.key}.jsonl.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = n_roles = 0

    with gzip.open(out_path, "wt", encoding="utf-8") as out:
        for responses_dir, scores_dir, stage in sources:
            if not responses_dir.exists():
                print(f"  (no {stage} responses at {responses_dir})")
                continue
            for f in sorted(responses_dir.glob("*.jsonl")):
                role = f.stem
                scores = load_scores(scores_dir, role)
                kept = 0
                with open(f) as fh:
                    for i, line in enumerate(fh):
                        if args.per_role and kept >= args.per_role:
                            break
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        sysp, question, response = extract(entry)
                        if response is None:
                            continue
                        key = entry.get("key") or entry.get("id") or str(i)
                        out.write(json.dumps({
                            "stage": stage, "role": role, "score": scores.get(key),
                            "system_prompt": sysp, "question": question, "response": response,
                        }) + "\n")
                        kept += 1
                        n_written += 1
                n_roles += 1

    size_mb = out_path.stat().st_size / 1e6
    print(f"{args.key}: wrote {n_written:,} responses from {n_roles} role files -> {out_path} ({size_mb:.1f} MB gz)")

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        private = os.environ.get("HF_PRIVATE", "0") == "1"
        api.create_repo(args.repo, repo_type="dataset", private=private, exist_ok=True)
        try:
            api.update_repo_settings(repo_id=args.repo, repo_type="dataset", private=private)
        except Exception:  # noqa: BLE001
            pass
        api.upload_file(path_or_fileobj=str(out_path), path_in_repo=f"{args.key}/{out_path.name}",
                        repo_id=args.repo, repo_type="dataset")
        print(f"{args.key}: uploaded to {args.repo}/{args.key}/{out_path.name}")


if __name__ == "__main__":
    main()
