#!/usr/bin/env python3
"""Build the PEP 503 package index served from GitHub Pages.

Usage:
    python3 scripts/gen_index.py <pages_root>

Input layout (what the publish job stages):
    <pages_root>/whl/<backend>/*.whl

Output layout:
    <pages_root>/whl/<backend>/index.html
        A *flat* listing whose links point into the project directory. Usable
        with `pip install --find-links` / `uv pip install --find-links`.
    <pages_root>/whl/<backend>/llama-cpp-python/index.html
        A real PEP 503 "simple" project page. Required for
        `pip install --index-url .../whl/<backend>/`, because pip and uv
        normalise the project name and then request
        `<index>/llama-cpp-python/`. A flat index at the backend root 404s on
        that path and resolution fails with "no matching distribution".

The wheels themselves live only in the project directory, so nothing is
duplicated on disk and there is exactly one URL per wheel.

This script is additive: it only writes index.html files, never removes wheels.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

BACKENDS = ["prism-metal", "prism-cpu", "prism-linux-cpu"]
# PEP 503 normalised project name.
PROJECT = "llama-cpp-python"


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _page(title: str, links: list[str]) -> str:
    out = [
        "<!DOCTYPE html>",
        '<html><head><meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        "</head><body>",
        f"<h1>{html.escape(title)}</h1>",
    ]
    out += links
    out += ["</body></html>", ""]
    return "\n".join(out)


def gen_backend(backend_dir: Path) -> None:
    project_dir = backend_dir / PROJECT
    project_dir.mkdir(parents=True, exist_ok=True)

    # Collect from both locations: the publish job stages fresh wheels at the
    # backend root, while the append-only carry-over step downloads older wheels
    # straight into the project directory.
    wheels = {w.name: w for w in project_dir.glob("*.whl")}
    for whl in backend_dir.glob("*.whl"):
        wheels[whl.name] = whl
    if not wheels:
        print(f"  {backend_dir.name}: no wheels, skipping")
        return
    wheels = [wheels[k] for k in sorted(wheels)]

    # Normalise: every wheel ends up in the PEP 503 project directory.
    for whl in wheels:
        dest = project_dir / whl.name
        if dest.resolve() == whl.resolve():
            continue
        if not dest.exists() or dest.stat().st_size != whl.stat().st_size:
            dest.write_bytes(whl.read_bytes())
        whl.unlink()

    # PEP 503 project page: links are relative to the project directory.
    (project_dir / "index.html").write_text(
        _page(
            f"Links for {PROJECT} ({backend_dir.name})",
            [f'<a href="{html.escape(w.name)}">{html.escape(w.name)}</a><br/>' for w in wheels],
        )
    )

    # Flat listing at the backend root, pointing into the project directory.
    (backend_dir / "index.html").write_text(
        _page(
            f"Flat index: {backend_dir.name}",
            [
                f'<a href="{PROJECT}/{html.escape(w.name)}">{html.escape(w.name)}</a><br/>'
                for w in wheels
            ],
        )
    )
    print(f"  {backend_dir.name}: {len(wheels)} wheel(s) indexed")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        return 1
    for backend in BACKENDS:
        backend_dir = root / "whl" / backend
        if not backend_dir.is_dir():
            continue
        gen_backend(backend_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
