#!/usr/bin/env python3
"""Regenerate PEP 503 index.html files for a local GitHub Pages directory tree.

Usage:
    python3 scripts/gen_index.py <pages_root>

The directory structure should be:
    <pages_root>/
        whl/
            prism-metal/
                llama_cpp_python-*.whl
            prism-cpu/
                ...
            prism-linux-cpu/
                ...

This script is append-only: it never deletes existing wheels or index entries.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path


def gen_index(root: Path) -> None:
    backends = [
        "prism-metal",
        "prism-cpu",
        "prism-linux-cpu",
    ]
    for backend in backends:
        dir = root / "whl" / backend
        if not dir.exists():
            continue
        wheels = sorted(dir.glob("*.whl"))
        if not wheels:
            continue
        lines = [
            "<!DOCTYPE html>",
            "<html><head><meta charset=\"utf-8\"><title>Index of /whl/{}</title></head><body>".format(backend),
            "<h1>Index of /whl/{}</h1>".format(backend),
        ]
        for whl in wheels:
            lines.append(
                '<a href="{}">{}</a><br/>'.format(
                    html.escape(whl.name), html.escape(whl.name)
                )
            )
        lines.append("</body></html>")
        (dir / "index.html").write_text("\n".join(lines))
        print(f"  {backend}: {len(wheels)} wheel(s) -> index.html")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <pages_root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        return 1
    gen_index(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())