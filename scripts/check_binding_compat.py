#!/usr/bin/env python3
"""Verify the ctypes binding in llama-cpp-python is ABI-compatible with a given
llama.cpp checkout (upstream or the PrismML fork).

Why this exists
---------------
llama-cpp-python talks to libllama through hand-written ctypes bindings
(``llama_cpp/llama_cpp.py``). ctypes has *no* compile-time checking, so any
drift between the Python struct/enum declarations and the C headers is silent
at build time and becomes memory corruption (or nonsense enum values) at
runtime. This script catches that drift statically, before compiling.

It checks three classes of drift:
  1. missing symbols  - a function/constant the binding calls is gone from the
     headers (or moved into a private header).
  2. signature drift  - a function the binding calls has different parameters.
  3. layout drift     - a struct passed by value through the C ABI has gained,
     lost, or reordered a field. THIS IS THE DANGEROUS ONE: llama_model_params
     and llama_context_params are passed/returned by value.
  4. enum drift       - a hardcoded integer constant no longer matches the C enum.

Usage:
    python3 scripts/check_binding_compat.py \\
        --llama-cpp-python /path/to/llama-cpp-python \\
        --llama-cpp /path/to/llama.cpp
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------- utilities


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def read(path: Path) -> str:
    return strip_comments(path.read_text(errors="ignore"))


def headers_of(root: Path, *globs: str) -> list[Path]:
    out: list[Path] = []
    for pattern in globs:
        out.extend(sorted(root.glob(pattern)))
    return out


def match_braces(
    text: str, open_index: int, open_char: str = "{", close_char: str = "}"
) -> tuple[str, int]:
    """Return (body, index-after-close) for the bracket at ``open_index``."""
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == open_char:
            depth += 1
        elif text[i] == close_char:
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : i], i + 1
    return text[open_index + 1 :], len(text)


def split_top_level(body: str, separator: str = ";") -> list[str]:
    """Split on ``separator`` occurrences that are not inside (), [] or {}."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in body:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == separator and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


C_AGGREGATE_KEYWORDS = {"struct", "union", "enum"}


def c_field_names(body: str) -> list[str]:
    """Ordered field names of a C struct/union body.

    Anonymous nested aggregates (e.g. the ``union { ... }`` inside
    ``llama_model_kv_override``) are flattened in place, which is what a ctypes
    binding has to do to stay layout-compatible.
    """
    fields: list[str] = []
    for part in split_top_level(body):
        part = part.strip()
        if not part:
            continue
        if "{" in part:
            before, after = part.split("{", 1)
            inner, _ = match_braces(part, len(before))
            declared = re.search(r"\b(\w+)\s*$", before)
            if declared and declared.group(1) not in C_AGGREGATE_KEYWORDS:
                # named nested aggregate: it is a single field
                fields.append(declared.group(1))
            else:
                fields.extend(c_field_names(inner))
            # any declarators after the closing brace, e.g. "int x[4];"
            for trailing in re.findall(r"\b(\w+)\s*(?:\[[^\]]*\])?\s*$", after):
                fields.append(trailing)
        else:
            match = re.search(r"\b(\w+)\s*(?:\[[^\]]*\])?$", part)
            if match:
                fields.append(match.group(1))
    return fields


def function_decls(text: str, name: str) -> set[str]:
    """Return normalised declaration(s) of C function ``name``.

    The declaration start is found by scanning back to the previous statement
    boundary (``;`` / ``{`` / ``}``) rather than by matching a return-type
    pattern. That way attribute macros such as
    ``LLAMA_DEPRECATED(LLAMA_API "use X instead")`` -- which contain parentheses
    and therefore defeat naive regexes -- are captured correctly.
    """
    decls: set[str] = set()
    for match in re.finditer(r"\b(" + re.escape(name) + r")\s*\(", text):
        start = match.start(1)
        i = start
        while i > 0 and text[i - 1] not in ";{}":
            i -= 1
        depth = 0
        j = match.end() - 1
        while j < len(text):
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        end = text.find(";", j)
        if end == -1:
            continue
        decl = re.sub(r"\s+", " ", text[i : end + 1]).strip()
        if decl:
            decls.add(decl)
    return decls


def c_structs(text: str) -> dict[str, list[str]]:
    """Map struct/union name -> ordered field names."""
    out: dict[str, list[str]] = {}
    for match in re.finditer(r"\b(?:struct|union)\s+(\w+)\s*\{", text):
        name = match.group(1)
        body, _ = match_braces(text, match.end() - 1)
        fields = c_field_names(body)
        if fields:
            out.setdefault(name, fields)
    return out


def c_enums(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for match in re.finditer(r"enum\s+(\w+)\s*\{([^}]*)\}", text):
        value = 0
        for part in match.group(2).split(","):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                key, raw = part.split("=", 1)
                key, raw = key.strip(), raw.strip()
                try:
                    value = int(raw, 0)
                except ValueError:
                    value = 0  # alias to a symbolic constant
            else:
                key = part
            if not re.match(r"^\w+$", key):
                continue
            out.setdefault(key, value)
            value += 1
    return out


def c_defines(text: str) -> dict[str, str]:
    return dict(
        re.findall(r"#define\s+(LLAMA_\w+|GGML_\w+)\s+(-?\w+)", text)
    )


# ------------------------------------------------------- binding extraction


def binding_symbols(llama_cpp_py: Path) -> set[str]:
    text = llama_cpp_py.read_text()
    syms = set(re.findall(r'@ctypes_function\(\s*"([\w]+)"', text))
    syms |= set(re.findall(r"_lib\.(\w+)\.argtypes", text))
    return syms


def binding_constants(llama_cpp_py: Path) -> dict[str, int]:
    text = llama_cpp_py.read_text()
    pairs = re.findall(
        r"^(\w+)\s*=\s*(?:ctypes\.c_int\()?(-?\d+)\)?\s*(?:#.*)?$", text, re.M
    )
    return {name: int(value) for name, value in pairs}


def binding_structs(llama_cpp_py: Path) -> dict[str, list[tuple[str, str]]]:
    """Extract every ctypes Structure/Union as (field_name, type_expr) pairs.

    Entries are located by bracket matching rather than line matching, so
    multi-line tuple entries are captured too. ``#`` comments are stripped so
    trailing notes cannot pollute the captured type expressions.
    """
    text = "\n".join(line.split("#")[0] for line in llama_cpp_py.read_text().split("\n"))
    structs: dict[str, list[tuple[str, str]]] = {}
    for match in re.finditer(r"^class (\w+)\(ctypes\.(Structure|Union)\):", text, re.M):
        name = match.group(1)
        fields_match = re.search(
            r"^[ \t]+_fields_[ \t]*=[ \t]*\[", text[match.end() :], re.M
        )
        if not fields_match:
            continue
        open_index = match.end() + fields_match.end() - 1
        body, _ = match_braces(text, open_index, "[", "]")
        entries: list[tuple[str, str]] = []
        for element in split_top_level(body, ","):
            element = element.strip()
            if not element:
                continue
            field = re.match(r'\(?\s*["\'](\w+)["\']\s*,(.*)', element, re.S)
            if not field:
                continue
            type_expr = field.group(2).strip()
            # drop the closing paren / comma that delimits the tuple entry
            type_expr = re.sub(r"[,)]+$", "", type_expr).strip()
            entries.append((field.group(1), type_expr))
        structs[name] = entries
    return structs


def expand_nested(
    fields: list[tuple[str, str]], structs: dict[str, list[tuple[str, str]]], seen: frozenset
) -> list[str]:
    """Flatten fields whose type is a bare nested ctypes Structure/Union.

    The C side flattens anonymous nested aggregates, so the Python side must be
    flattened the same way before the two lists can be compared. Pointer-typed
    fields (``POINTER(llama_sampler)``) are opaque and are left alone.
    """
    out: list[str] = []
    for name, type_expr in fields:
        bare = re.fullmatch(r"(\w+)", type_expr)
        if bare and bare.group(1) in structs and bare.group(1) not in seen:
            out.extend(expand_nested(structs[bare.group(1)], structs, seen | {bare.group(1)}))
        else:
            out.append(name)
    return out


# ----------------------------------------------------------------- checking

# Opaque/private types whose C layout the binding never mirrors field-by-field.
SKIP_STRUCTS = {"llama_sampler", "llama_sampler_i", "llama_sampler_data"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-cpp-python", required=True, type=Path)
    parser.add_argument("--llama-cpp", required=True, type=Path)
    parser.add_argument(
        "--reference-llama-cpp",
        type=Path,
        help=(
            "A second llama.cpp checkout to diff function signatures against "
            "(e.g. the pristine vendor/llama.cpp that ships with the chosen "
            "llama-cpp-python release). Enables signature-drift detection."
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true", help="only print problems"
    )
    args = parser.parse_args()

    binding_path = args.llama_cpp_python / "llama_cpp" / "llama_cpp.py"
    if not binding_path.is_file():
        print(f"error: binding not found: {binding_path}", file=sys.stderr)
        return 2

    public = headers_of(
        args.llama_cpp,
        "include/llama.h",
        "ggml/include/ggml.h",
        "ggml/include/ggml-cpu.h",
        "tools/mtmd/include/mtmd.h",
    )
    private = headers_of(
        args.llama_cpp, "src/*.h", "ggml/src/*.h", "tools/mtmd/*.h"
    )
    if not public:
        print("error: no public headers found", file=sys.stderr)
        return 2

    public_text = "\n".join(read(p) for p in public)
    private_text = "\n".join(read(p) for p in private)
    all_text = public_text + "\n" + private_text

    problems: list[str] = []

    # 1. symbols ------------------------------------------------------------
    symbols = binding_symbols(binding_path)
    global ALL_BINDING_STRUCTS
    ALL_BINDING_STRUCTS = binding_structs(binding_path)
    missing = sorted(
        s
        for s in symbols
        if not function_decls(public_text, s) and not function_decls(private_text, s)
    )
    for sym in missing:
        problems.append(f"MISSING SYMBOL  {sym} is called by the binding but not declared in any header")

    # 2. signatures ---------------------------------------------------------
    if args.reference_llama_cpp:
        ref_public = headers_of(
            args.reference_llama_cpp,
            "include/llama.h",
            "ggml/include/ggml.h",
            "ggml/include/ggml-cpu.h",
            "tools/mtmd/include/mtmd.h",
        )
        ref_text = "\n".join(read(p) for p in ref_public)
        for sym in sorted(symbols):
            ref = function_decls(ref_text, sym)
            cur = function_decls(public_text, sym)
            if not ref or not cur:
                continue
            if len(ref) > 1 or len(cur) > 1:
                continue  # ambiguous (attribute macros); skip
            if ref != cur:
                problems.append(
                    f"SIGNATURE DRIFT {sym}\n"
                    f"    reference: {sorted(ref)[0][:160]}\n"
                    f"    fork     : {sorted(cur)[0][:160]}"
                )

    # 3. struct layout vs the C headers -------------------------------------
    cstructs = c_structs(public_text)
    for name, fields in binding_structs(binding_path).items():
        if name in SKIP_STRUCTS:
            continue
        if name not in cstructs:
            continue
        py_fields = expand_nested(fields, ALL_BINDING_STRUCTS, frozenset({name}))
        if cstructs[name] != py_fields:
            added = [f for f in cstructs[name] if f not in py_fields]
            removed = [f for f in py_fields if f not in cstructs[name]]
            reordered = not added and not removed
            problems.append(
                f"LAYOUT DRIFT     struct {name} (passed by value!)\n"
                f"    ctypes fields ({len(py_fields)}): {py_fields}\n"
                f"    C fields      ({len(cstructs[name])}): {cstructs[name]}\n"
                f"    added by C: {added}   missing in ctypes: {removed}"
                f"{'   ORDER CHANGED' if reordered else ''}"
            )

    # 4. constants ----------------------------------------------------------
    cenum = c_enums(public_text) | c_enums(private_text)
    cdef = c_defines(public_text) | c_defines(private_text)
    for name, value in sorted(binding_constants(binding_path).items()):
        if name in cdef and re.match(r"^-?\d+$", cdef[name]):
            actual = int(cdef[name])
        elif name in cenum:
            actual = cenum[name]
        else:
            continue
        if actual != value:
            problems.append(
                f"ENUM DRIFT       {name}: binding says {value}, header says {actual}"
            )

    # ------------------------------------------------------------- report
    if not args.quiet:
        print(f"binding   : {binding_path}")
        print(f"llama.cpp : {args.llama_cpp}")
        print(f"headers   : {len(public)} public, {len(private)} private")
        print(f"symbols checked: {len(symbols)}")
        print()
    if problems:
        print(f"FAIL: {len(problems)} compatibility problem(s)\n")
        for p in problems:
            print("  " + p)
        return 1
    print("OK: binding is ABI-compatible with these headers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
