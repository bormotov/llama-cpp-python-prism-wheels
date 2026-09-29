#!/usr/bin/env python3
"""Prepare a llama-cpp-python source tree for building against the PrismML fork.

Two things happen here, both of which used to be fragile inline shell/regex in
the CI workflow:

1. The ctypes ABI patch (``patches/0001-prism-ctypes-abi.patch``) is applied
   with ``git apply``. This replaces a pile of ``re.sub`` calls that silently
   no-op'd when the upstream text drifted.
2. ``llama_cpp.__version__`` is rewritten to a PEP 440 local version
   ``<upstream>+prism.<tag>`` (dashes in the tag become dots). Done in Python
   rather than ``sed -i''`` because that flag is BSD-only and broke on Linux.

Usage:
    python3 scripts/prepare_llama_cpp_python.py <src_dir> <upstream_version> <prism_tag>

Example:
    python3 scripts/prepare_llama_cpp_python.py llama-cpp-python 0.3.35 prism-b10743-adfffbe
    # -> llama_cpp.__version__ == "0.3.35+prism.b10743.adfffbe"
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

PATCH_NAME = "0001-prism-ctypes-abi.patch"


def find_patch() -> Path:
    """Locate the ABI patch, searching upward from this file.

    CI checks this repo out into a subdirectory (``ci/``), a local dev may have
    it at the workspace root, so don't assume a fixed depth. Do not search
    arbitrarily far up: a patch from an unrelated checkout would be worse than
    a clear failure.
    """
    for parent in Path(__file__).resolve().parents[:4]:
        candidate = parent / "patches" / PATCH_NAME
        if candidate.is_file():
            return candidate
    fail(
        f"could not find patches/{PATCH_NAME} in any parent of {__file__}. "
        "Run this script from a full checkout of this repository."
    )

# Field order of the fork headers, asserted after patching. A ctypes Structure
# whose `_fields_` order diverges from the C struct silently corrupts every
# field after the divergence, so we pin the neighbours of each new field.
# (Reference sizeof(): llama_model_params == 80, llama_context_params == 168.)
EXPECTED_FIELDS = {
    "llama_model_params": {
        # new field is the very first member in the C struct
        "__around__": ("dspark_head_source", None, "devices"),
    },
    "llama_context_params": {
        "__around__": ("path_kv_mean_center", "type_v", "abort_callback"),
    },
    "llama_opt_params": {
        "__around__": ("optimizer_type", "get_opt_pars_ud", None),
    },
}
EXPECTED_GGML_TYPE_COUNT = 144


def fail(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def apply_patch(src: Path) -> None:
    patch = find_patch()
    print(f"Applying {patch}")
    proc = subprocess.run(
        ["git", "apply", "--check", "-v", str(patch)],
        cwd=src,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        fail(
            "patch does not apply cleanly. The upstream llama_cpp/llama_cpp.py "
            "layout has changed; re-derive the patch with "
            "scripts/check_binding_compat.py and regenerate "
            "patches/0001-prism-ctypes-abi.patch."
        )
    subprocess.run(["git", "apply", str(patch)], cwd=src, check=True)


def set_version(src: Path, upstream: str, prism_tag: str) -> str:
    # Fork tags look like "prism-b10743-adfffbe"; the local segment is
    # "prism.b10743.adfffbe" so the wheel reads
    #   0.3.35+prism.b10743.adfffbe
    # Strip a leading "prism-" so passing either the raw tag or an already
    # normalised one cannot produce "prism.prism.".
    bare = prism_tag[len("prism-"):] if prism_tag.startswith("prism-") else prism_tag
    local = "prism." + bare.replace("-", ".")
    version = f"{upstream}+{local}"
    init = src / "llama_cpp" / "__init__.py"
    if not init.is_file():
        fail(f"{init} not found")
    text = init.read_text()
    new_text, n = re.subn(
        r'^__version__ = ".*"$', f'__version__ = "{version}"', text, count=1, flags=re.M
    )
    if n != 1:
        fail(f"could not find __version__ line in {init}")
    init.write_text(new_text)
    return version


def _struct_field_names(node: ast.ClassDef) -> list[str]:
    """Extract field names from a ctypes.Structure class's `_fields_` list."""
    for stmt in node.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not any(getattr(t, "id", None) == "_fields_" for t in stmt.targets):
            continue
        if not isinstance(stmt.value, (ast.List, ast.Tuple)):
            continue
        names = []
        for elt in stmt.value.elts:
            if isinstance(elt, ast.Tuple) and elt.elts:
                first = elt.elts[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    names.append(first.value)
        return names
    return []


def verify_layout(src: Path) -> None:
    """Statically assert the patched binding matches the fork's C ABI.

    Done with `ast` rather than by importing the module: llama_cpp.py pulls in
    numpy/diskcache, which are not necessarily installed on a bare build
    runner, and we want this check to run before `uv build` installs anything.
    """
    binding = src / "llama_cpp" / "llama_cpp.py"
    tree = ast.parse(binding.read_text(), filename=str(binding))
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}

    for struct, spec in EXPECTED_FIELDS.items():
        new, before, after = spec["__around__"]
        node = classes.get(struct)
        if node is None:
            fail(f"class {struct} not found in {binding}")
        names = _struct_field_names(node)
        if not names:
            fail(f"could not read _fields_ of {struct}")
        if new not in names:
            fail(f"{struct} is missing the [prism] field {new!r}; patch did not apply?")
        i = names.index(new)
        if before is not None and (i == 0 or names[i - 1] != before):
            fail(
                f"{struct}.{new} must directly follow {before!r} in the C layout; "
                f"got {names[max(0, i - 1):i]!r}"
            )
        if after is not None and (i + 1 >= len(names) or names[i + 1] != after):
            fail(
                f"{struct}.{new} must be directly followed by {after!r} in the C layout; "
                f"got {names[i + 1:i + 2]!r}"
            )
        print(f"  {struct}: {new!r} between {before!r} and {after!r} (ok)")

    text = binding.read_text()
    m = re.search(r"^GGML_TYPE_COUNT = (\d+)", text, flags=re.M)
    if not m:
        fail("GGML_TYPE_COUNT not found in patched binding")
    if int(m.group(1)) != EXPECTED_GGML_TYPE_COUNT:
        fail(
            f"GGML_TYPE_COUNT is {m.group(1)}, expected {EXPECTED_GGML_TYPE_COUNT} "
            "(the fork added ggml types)"
        )
    print(f"  GGML_TYPE_COUNT = {m.group(1)} (ok)")


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    src = Path(argv[1]).resolve()
    upstream, prism_tag = argv[2], argv[3]
    if not (src / "llama_cpp" / "llama_cpp.py").is_file():
        fail(f"{src} does not look like a llama-cpp-python checkout "
             "(no llama_cpp/llama_cpp.py)")

    apply_patch(src)
    version = set_version(src, upstream, prism_tag)
    print(f"  __version__ = {version}")
    verify_layout(src)
    print("prepare_llama_cpp_python: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
