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

    # The question this script exists to answer is "does the persona text actually reach the model?"
    # It used to answer a DIFFERENT question -- "does the template render a system turn?" -- and
    # treat yes as UNSAFE. That inverted the meaning of its own signal and failed both models:
    #   * gemma-3 has no system role, but its template MERGES system content into the user turn, so
    #     the probe string appears -> flagged UNSAFE, when merging is exactly the safe behaviour.
    #   * gemma-4 has a genuine system role (<|turn>system ... <turn|>) -> flagged UNSAFE, when it is
    #     the model's native persona channel. Verified by generation on 2026-08-17: on the pipeline's
    #     own path a "speak like a pirate" system turn yields "Ahoy there, matey! ... the capital be
    #     Paris" -- persona obeyed and question answered. Nothing was being dropped.
    # The old verdict also recommended ASSISTANT_AXIS_FORCE_USER_CONCAT=1, which NO pipeline code
    # reads -- it only flipped this script's own output, so it silenced the alarm without changing
    # what the models were sent.
    #
    # So: ask the pipeline itself. format_conversation() is what builds every real conversation, and
    # it takes the system path ONLY when the probe text survives rendering, else it concatenates into
    # the user turn. Render what it produces and confirm the persona text is present in the result.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assistant-axis"))
    from assistant_axis.generation import format_conversation

    persona = "__PERSONA_TEST__"
    conv = format_conversation(instruction=persona, question="hello", tokenizer=tok)
    path = "system turn" if any(m["role"] == "system" for m in conv) else "user-turn concatenation"
    rendered = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    delivered = persona in rendered

    print(f"\npipeline format_conversation() path: {path}")
    print("--- what the pipeline will actually send ---"); print(rendered)

    if forced:
        print("\nVERDICT: OK — ASSISTANT_AXIS_FORCE_USER_CONCAT=1 set. NOTE: no pipeline code reads "
              "this variable; it changes this script's verdict only, not what the model receives.")
        return 0
    if not delivered:
        print(f"\nVERDICT: UNSAFE — the persona text does NOT survive into the prompt the pipeline "
              f"sends (path: {path}). Persona instructions would be silently dropped and every "
              f"role vector would be meaningless.", file=sys.stderr)
        return 1
    print(f"\nVERDICT: OK — persona text reaches the model via {path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
