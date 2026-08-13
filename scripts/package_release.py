#!/usr/bin/env python3
"""Package pipeline outputs into the release layout used by
lu-christina/assistant-axis-vectors:

    <key>/assistant_axis.pt        raw tensor (n_layers, hidden)
    <key>/default_vector.pt        raw tensor (n_layers, hidden)
    <key>/role_vectors/<role>.pt   raw tensor (n_layers, hidden)

Pipeline vector files are dicts {"vector": tensor, "type": ..., "role": ...};
this unwraps them to raw tensors.
"""
import argparse
from pathlib import Path

import torch


def unwrap(obj):
    if isinstance(obj, dict):
        for k in ("vector", "axis", "tensor"):
            if k in obj:
                return obj[k]
        raise ValueError(f"dict without a known tensor key: {list(obj.keys())}")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors_dir", required=True)
    ap.add_argument("--axis", required=True)
    ap.add_argument("--out", required=True, help="release dir, e.g. .../release/gemma-3-27b")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "role_vectors").mkdir(parents=True, exist_ok=True)

    axis = unwrap(torch.load(args.axis, weights_only=False))
    torch.save(axis, out / "assistant_axis.pt")

    n_roles = 0
    default_saved = False
    for f in sorted(Path(args.vectors_dir).glob("*.pt")):
        data = torch.load(f, weights_only=False)
        vec = unwrap(data)
        role = data.get("role", f.stem) if isinstance(data, dict) else f.stem
        vtype = data.get("type", "") if isinstance(data, dict) else ""
        if "default" in role or vtype == "mean":
            torch.save(vec, out / "default_vector.pt")
            default_saved = True
        else:
            torch.save(vec, out / "role_vectors" / f"{role}.pt")
            n_roles += 1

    assert default_saved, "no default vector found in vectors_dir"
    assert n_roles > 0, "no role vectors found"
    print(f"packaged: axis {tuple(axis.shape)}, default, {n_roles} role vectors -> {out}")


if __name__ == "__main__":
    main()
