#!/usr/bin/env python3
"""Generic acceptance test: does the PrismML wheel run *any* GGUF correctly?

Model-agnostic on purpose. The ABI patch in patches/0001-prism-ctypes-abi.patch
reshapes llama_model_params and llama_context_params, so a wrong field offset
breaks *every* model, not just the fork's exotic quantizations. This test is the
regression guard for that: run it against a stock llama.cpp quantization (e.g.
Gemma 4 Q4_0) and confirm the wheel still behaves.

Unlike scripts/acceptance_bonsai.py it makes no assumption about the chat
template, the special tokens or the architecture -- it uses whatever the GGUF
declares. The only semantic assertion is a deterministic arithmetic answer.

Usage:
    python3 scripts/smoke_test.py <model.gguf> [--n-ctx 4096] [--max-tokens 200]
"""
from __future__ import annotations

import argparse
import re
import sys

# Deterministic enough to assert on at temperature 0: 2+2 can only be 4.
PROMPT = "What is 2 + 2? Reply with only the number and nothing else."


def resolve_special_token(llm, name: str) -> str | None:
    """Resolve a special token's text from its id.

    Gotcha: most GGUFs (Bonsai, Gemma 4) carry `tokenizer.ggml.eos_token_id` but
    NOT the `tokenizer.ggml.eos_token` string, so a metadata lookup of the text
    raises KeyError. Read the id and ask the model instead. The `Llama`
    constructor does exactly this internally (llama.py: token_get_text).
    """
    token_id = llm.metadata.get(f"tokenizer.ggml.{name}_token_id")
    if token_id is not None:
        return llm._model.token_get_text(int(token_id))
    # Fall back to the model API for GGUFs that omit the *_token_id key.
    getter = getattr(llm._model, f"token_{name}", None)
    if getter is None:
        return None
    tid = getter()
    return llm._model.token_get_text(tid) if tid != -1 else None


def longest_repeated_run(text: str, window: int = 40) -> tuple[str, int] | None:
    """Find the most-repeated fixed-size window -- catches degenerate loops."""
    best: tuple[str, int] | None = None
    seen: dict[str, int] = {}
    for i in range(0, max(0, len(text) - window)):
        chunk = text[i : i + window]
        seen[chunk] = seen.get(chunk, 0) + 1
        if best is None or seen[chunk] > best[1]:
            best = (chunk, seen[chunk])
    if best and best[1] >= 3:
        return best
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument(
        "--thinking",
        action="store_true",
        help="render the generation prompt with thinking enabled (default: off)",
    )
    ap.add_argument("--prompt", default=PROMPT, help="override the test prompt")
    args = ap.parse_args()

    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    failures: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}: {label}" + (f" -- {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    print("=" * 70)
    print("1. Loading model (ABI-critical step: struct layout must match the fork)")
    print("=" * 70)
    llm = Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        flash_attn=True,
        verbose=True,
    )
    meta = llm.metadata
    print(f"\narchitecture   = {meta.get('general.architecture')}")
    print(f"file_type      = {meta.get('general.file_type')}")
    print(f"n_ctx_train    = {llm._model.n_ctx_train()}")
    print(f"active n_ctx   = {llm.n_ctx()}")
    print(f"n_vocab        = {llm.n_vocab()}")
    print(f"block_count    = {meta.get(meta.get('general.architecture', '') + '.block_count')}")

    print("\n" + "=" * 70)
    print("2. Resolving special tokens")
    print("=" * 70)
    eos_token = resolve_special_token(llm, "eos")
    bos_token = resolve_special_token(llm, "bos")
    print(f"eos  id={meta.get('tokenizer.ggml.eos_token_id')} -> {eos_token!r}")
    print(f"bos  id={meta.get('tokenizer.ggml.bos_token_id')} -> {bos_token!r}")
    check(eos_token is not None, "eos token text resolved")
    check(bos_token is not None, "bos token text resolved")

    print("\n" + "=" * 70)
    print("3. Chat template auto-selection")
    print("=" * 70)
    print(f"chat_format = {llm.chat_format!r}")
    check(
        llm.chat_format is not None,
        "a chat format was selected",
        "if this is 'llama-2' the GGUF had no usable template",
    )
    template = meta.get("tokenizer.chat_template")
    check(template is not None, "GGUF carries an embedded chat template")

    print("\n" + "=" * 70)
    print("4. Rendering prompt with the GGUF's own Jinja template")
    print("=" * 70)
    # create_chat_completion has no **kwargs, so it cannot forward
    # enable_thinking; Jinja2ChatFormatter can (it splats **kwargs into
    # environment.render()).
    handler = Jinja2ChatFormatter(
        template=template, eos_token=eos_token, bos_token=bos_token
    )
    formatted = handler(
        messages=[{"role": "user", "content": args.prompt}],
        enable_thinking=args.thinking,
    )
    prompt = formatted.prompt
    print(f"--- rendered prompt ({len(prompt)} chars), tail: ---")
    print(prompt[-400:])
    check(bool(prompt.strip()), "template rendered a non-empty prompt")
    if not args.thinking:
        # With thinking disabled these templates open and immediately close the
        # thought channel rather than omitting it, so assert it is empty.
        m = re.search(r"thought>(.*?)<channel\|>", prompt, flags=re.S)
        if m:
            check(
                not m.group(1).strip(),
                "thought channel is empty (thinking disabled)",
                repr(m.group(1)),
            )

    print("\n" + "=" * 70)
    print("5. Running completion")
    print("=" * 70)
    out = llm.create_completion(
        prompt=prompt, max_tokens=args.max_tokens, temperature=0.0
    )
    choice = out["choices"][0]
    text = choice["text"]
    finish_reason = choice.get("finish_reason")
    print(f"\n--- RAW ({out['usage']['completion_tokens']} tokens) ---")
    print(repr(text))
    print(f"\n--- finish_reason: {finish_reason} ---")

    print("\n" + "=" * 70)
    print("6. Assertions")
    print("=" * 70)
    body = text.strip()
    check(bool(body), "completion is non-empty")
    check("�" not in body, "no U+FFFD replacement characters (tokenizer drift)")
    check(
        finish_reason in ("stop", "length"),
        "finish_reason is sane",
        str(finish_reason),
    )
    rep = longest_repeated_run(body)
    check(rep is None, "no degenerate repetition loop", repr(rep) if rep else "")
    # The one semantic assertion. Only meaningful for the default prompt; with
    # --prompt there is no known-correct answer, so we stop at the structural
    # checks above plus a length floor.
    if args.prompt == PROMPT:
        # Scan the whole answer: a reasoning model may restate it, and 2+2 has
        # no alternative spelling other than "four".
        check(
            bool(re.search(r"\b(4|four)\b", body, re.IGNORECASE)),
            "arithmetic answer 2+2=4 present",
        )
    else:
        check(len(body) >= 8, "answer is substantive, not a stub", f"{len(body)} chars")
        print(f"\n--- COMPLETION (rendered) ---\n{body}")

    print("=" * 70)
    if failures:
        print(f"RESULT: FAIL ({len(failures)}): {failures}")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
