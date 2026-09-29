#!/usr/bin/env python3
"""Acceptance test: does the PrismML wheel actually run Ternary Bonsai 2 27B?

Loads the PQ2_0 GGUF, renders a chat prompt with thinking disabled, and runs a
real completion. Verifies the answer is parseable (not just that the model
loaded), which is the thing the ABI patch actually protects.

Usage:
    python3 scripts/acceptance_bonsai.py <model.gguf> [--n-ctx 4096]
"""
from __future__ import annotations

import argparse
import re
import sys

MODEL_DEFAULT = ".spike/models/Ternary-Bonsai-2-27B-PQ2_0.gguf"

PROMPT = (
    "Extract the names from this text as a JSON array of strings. "
    "Text: 'Ada met Grace in Paris, and Grace met Alan in London.' "
    "Reply with only the JSON array."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=MODEL_DEFAULT)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    print("=" * 70)
    print("1. Loading model (this is the ABI-critical step: struct layout)")
    print("=" * 70)
    llm = Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        flash_attn=True,
        verbose=True,
    )
    print(f"\nmodel n_ctx_train = {llm._model.n_ctx_train()}")
    print(f"active n_ctx      = {llm.n_ctx()}")
    print(f"n_vocab           = {llm.n_vocab()}")

    print("\n" + "=" * 70)
    print("2. Resolving EOS/BOS token text")
    print("=" * 70)
    # Gotcha: this GGUF has tokenizer.ggml.eos_token_id but NOT
    # tokenizer.ggml.eos_token, so llm.metadata["tokenizer.ggml.eos_token"]
    # raises KeyError.
    meta = llm.metadata
    eos_id = meta.get("tokenizer.ggml.eos_token_id")
    eos_token = (
        llm._model.token_get_text(int(eos_id)) if eos_id is not None else "<|im_end|>"
    )
    bos_id = meta.get("tokenizer.ggml.bos_token_id")
    bos_token = (
        llm._model.token_get_text(int(bos_id)) if bos_id is not None else None
    )
    print(f"eos_token_id={eos_id} -> {eos_token!r}")
    print(f"bos_token_id={bos_id} -> {bos_token!r}")
    assert "tokenizer.ggml.eos_token" not in meta, (
        "metadata now HAS eos_token; update this test and the README gotcha"
    )

    print("\n" + "=" * 70)
    print("3. Rendering prompt with enable_thinking=False")
    print("=" * 70)
    template = meta["tokenizer.chat_template"]
    # create_chat_completion has no **kwargs, so it cannot forward
    # enable_thinking; Jinja2ChatFormatter can.
    handler = Jinja2ChatFormatter(
        template=template, eos_token=eos_token, bos_token=bos_token
    )
    formatted = handler(
        messages=[{"role": "user", "content": PROMPT}],
        enable_thinking=False,
    )
    prompt = formatted.prompt
    print(f"--- rendered prompt ({len(prompt)} chars) ---")
    print(prompt[-600:])
    assert "<think>" not in prompt, "enable_thinking=False should suppress <think>"
    assert "</think>" in prompt, "expected the closed empty think block"

    print("\n" + "=" * 70)
    print("4. Running completion")
    print("=" * 70)
    out = llm.create_completion(
        prompt=prompt, max_tokens=args.max_tokens, temperature=0.0
    )
    choice = out["choices"][0]
    text = choice["text"]
    print(f"\n--- RAW COMPLETION ({out['usage']['completion_tokens']} tokens) ---")
    print(repr(text))
    print("\n--- COMPLETION (rendered) ---")
    print(text)
    print("\n--- finish_reason:", choice.get("finish_reason"))

    print("\n" + "=" * 70)
    print("5. Assertions")
    print("=" * 70)
    # Strip a residual think block if the model emitted one anyway.
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\[.*?\]", body, flags=re.S)
    ok = False
    if m:
        import json

        try:
            names = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            print(f"FAIL: extracted JSON did not parse: {exc}")
            names = None
        if isinstance(names, list):
            lowered = [str(n).lower() for n in names]
            expected = {"ada", "grace", "alan"}
            print(f"extracted names: {names}")
            missing = expected - set(lowered)
            if missing:
                print(f"FAIL: missing expected names: {sorted(missing)}")
            else:
                ok = True
                print("PASS: all three expected names present")
    if not ok:
        print("FAIL: could not extract a valid JSON name array from the completion")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
