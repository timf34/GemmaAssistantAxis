#!/usr/bin/env python3
"""Verify how persona instructions reach an SPP model.

These models were trained WITHOUT a system role. If a chat template silently renders an
unsupported system turn, the pipeline's auto-detection takes the system path and the persona
instruction is effectively ignored — which looks identical to "the model can't role-play".

Prints the rendered templates and the path that will be taken. Exits non-zero if the
configuration is unsafe.
"""
import argparse
import os
import sys

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    forced = os.environ.get("ASSISTANT_AXIS_FORCE_USER_CONCAT") == "1"

    sys_msgs = [{"role": "system", "content": "__SYSTEM_TEST__"}, {"role": "user", "content": "hello"}]
    user_only = [{"role": "user", "content": "You are a pirate.\n\nhello"}]

    try:
        with_sys = tok.apply_chat_template(sys_msgs, tokenize=False, add_generation_prompt=False)
        renders_system = "__SYSTEM_TEST__" in with_sys
    except Exception as e:  # noqa: BLE001
        with_sys, renders_system = f"<raised {type(e).__name__}: {e}>", False

    without = tok.apply_chat_template(user_only, tokenize=False, add_generation_prompt=True)

    print("--- with system turn ---");  print(with_sys)
    print("--- user-only (persona concatenated) ---"); print(without)
    print(f"\ntemplate renders a system turn: {renders_system}")
    print(f"ASSISTANT_AXIS_FORCE_USER_CONCAT: {'1 (forcing user-turn concatenation)' if forced else 'unset'}")

    if forced:
        print("\nVERDICT: OK — persona instructions go into the user turn, which these models understand.")
        return 0
    if renders_system:
        print("\nVERDICT: UNSAFE — the template renders a system turn, so the pipeline would use the "
              "system path. These models were trained without a system role, so persona instructions "
              "would likely be ignored. Set ASSISTANT_AXIS_FORCE_USER_CONCAT=1.", file=sys.stderr)
        return 1
    print("\nVERDICT: OK — no system support detected, pipeline will concatenate into the user turn.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
