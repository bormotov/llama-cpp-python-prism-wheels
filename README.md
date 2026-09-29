# llama-cpp-python-prism-wheels

Pre-built wheels of `llama-cpp-python` compiled against the **PrismML fork of llama.cpp** (`PrismML-Eng/llama.cpp@prism`), enabling:

- **Ternary Bonsai 2 27B** (`PQ2_0` / `PTQ1_0` quantization) on Apple Silicon (Metal)
- Full backward compatibility with stock models (Gemma, Llama, Qwen, etc.)

## Why this exists

The PrismML fork adds ternary/1-bit quantization formats (`PQ2_0`, `PTQ1_0`) required by models like **Ternary-Bonsai-2-27B**. Stock `llama-cpp-python` (PyPI) is built against upstream llama.cpp and **cannot load these models** — they either error with "unknown quantization" or silently produce garbage.

The PrismML org provides only CLI binaries (`llama-server`, `llama-cli`). **No Python wheels exist** — this project fills that gap.

## Status

| Phase | Status |
|-------|--------|
| Local spike build (Metal) | ✅ `0.3.35+prism.b10743.adfffbe` |
| ABI compatibility verified | ✅ 3 struct layout fixes + enum bump |
| Basic import / struct test | ✅ |
| Model load test (Bonsai 27B PQ2_0) | ✅ 6.8 GB on Metal, 65/65 layers offloaded |
| **Chat completion on Bonsai** | ✅ `["Ada", "Grace", "Alan"]`, `finish_reason: stop` |
| GitHub Actions CI (Metal + CPU + Linux) | ✅ all three green |
| GitHub Pages index | ✅ PEP 503, installable via pip/uv |
| Back-compat test (Gemma 12B QAT) | ⏳ pending |

Run the Bonsai acceptance test yourself:

```bash
python3 scripts/acceptance_bonsai.py path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf
```

It asserts the model loads, that `enable_thinking=False` yields an empty think
block, and that the completion parses to the expected names — i.e. it checks
correctness, not just that the wheel imports.

## Quick start (consumer)

The index is a standard **PEP 503 "simple" index**, so the normal tools work:

```bash
# pip — keep PyPI as the default so llama-cpp-python's own deps resolve there
pip install --extra-index-url \
  https://bormotov.github.io/llama-cpp-python-prism-wheels/whl/prism-metal/ \
  "llama-cpp-python==0.3.35+prism.b10743.adfffbe"
```

```bash
# uv
uv pip install --extra-index-url \
  https://bormotov.github.io/llama-cpp-python-prism-wheels/whl/prism-metal/ \
  "llama-cpp-python==0.3.35+prism.b10743.adfffbe"
```

In `pyproject.toml`, pin the index only for macOS so other platforms keep using
the stock PyPI wheel:

```toml
[[tool.uv.index]]
name = "llama-prism-metal"
url = "https://bormotov.github.io/llama-cpp-python-prism-wheels/whl/prism-metal/"
explicit = true

[tool.uv.sources]
llama-cpp-python = { index = "llama-prism-metal", marker = "sys_platform == 'darwin'" }
```

Pick the index that matches your platform:

| Index | Platform |
|-------|----------|
| `whl/prism-metal` | macOS ARM64 with Metal — **use this for Bonsai** |
| `whl/prism-cpu` | macOS ARM64, CPU only |
| `whl/prism-linux-cpu` | Linux x86_64, CPU + OpenBLAS |

Then in Python:

```python
from llama_cpp import Llama

# Bonsai 27B (requires enable_thinking=False to disable reasoning trace)
llm = Llama(
    model_path="Ternary-Bonsai-2-27B-PQ2_0.gguf",
    n_gpu_layers=-1,          # Metal offload
    n_ctx=4096,               # see "Context size" below; native 262144 OOMs on M5
    flash_attn=True,
    chat_format="chatml",     # or use the GGUF-embedded template
    verbose=True,
)

# IMPORTANT: pass enable_thinking=False via a custom chat handler
# (create_chat_completion doesn't forward **kwargs to the template)
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

handler = Jinja2ChatFormatter(
    template=llm.metadata["tokenizer.chat_template"],
    eos_token=llm.metadata["tokenizer.ggml.eos_token"],
    bos_token=llm.metadata["tokenizer.ggml.bos_token"],
)

response = handler(
    messages=[{"role": "user", "content": "Extract names: ..."}],
    enable_thinking=False,      # <-- disables the <thinking> block
    reasoning_effort="low",
)
prompt = response.prompt
# then feed to llm.create_completion(prompt=prompt, ...)
```

### Gotchas when using Bonsai

- **`tokenizer.ggml.eos_token` is missing.** The GGUF only carries
  `tokenizer.ggml.eos_token_id`. Read the token text via
  `llm.token_get_text(eos_id)` or hardcode `<|im_end|>` / `<|endoftext|>`, or the
  `Jinja2ChatFormatter` constructor above raises `KeyError`.
- **The GGUF embeds its own chat template.** llama-cpp-python selects
  `chat_template.default` automatically, so `chat_format="chatml"` is not
  required — pass the template explicitly only when you need `**kwargs` like
  `enable_thinking`.
- **`create_chat_completion` has no `**kwargs`**, so `enable_thinking` cannot be
  forwarded through it; use `Jinja2ChatFormatter` directly as shown.
- **`enable_thinking=False` emits an empty think block, it does not remove the
  tags.** The rendered generation prompt ends with
  `<think>\n\n</think>\n\n`. Don't assert `"<think>" not in prompt` — assert the
  block is *empty*, otherwise every run looks like a failure.
- **Context size.** The model declares `n_ctx_train = 262144`, but allocating
  that fails on an M5 (Metal `kIOGPUCommandBufferCallbackErrorOutOfMemory`,
  `llama_decode returned -3`) at roughly 18 GB. `n_ctx=4096` works and is the
  default to use.

## Build recipe (spike)

```bash
# 1. Clone llama-cpp-python at the exact release tag
git clone --branch v0.3.35 --recurse-submodules --shallow-submodules \
  https://github.com/abetlen/llama-cpp-python

# 2. Repoint the submodule to PrismML fork at the chosen tag
cd llama-cpp-python
git submodule set-url vendor/llama.cpp https://github.com/PrismML-Eng/llama.cpp
git submodule sync
rm -rf vendor/llama.cpp
git submodule update --init --recursive --depth 1
cd vendor/llama.cpp
git fetch --depth 1 origin refs/tags/prism-b10743-adfffbe:refs/tags/prism-b10743-adfffbe
git checkout prism-b10743-adfffbe
git submodule update --init --recursive --depth 1
cd ../..

# 3. Patch the ctypes binding for struct layout drift and set the version.
#    This applies patches/0001-prism-ctypes-abi.patch and verifies the result.
python3 scripts/prepare_llama_cpp_python.py . 0.3.35 prism-b10743-adfffbe
#    -> __version__ = "0.3.35+prism.b10743.adfffbe"

# 4. Build with Metal
CMAKE_ARGS="-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_CURL=OFF" \
uv build --wheel --out-dir dist .
```

## ABI compatibility notes (critical)

The PrismML fork (`b10743` ≈ 7k commits ahead of the `v0.3.35` vendor) introduces three **struct layout changes** in types passed **by value** across the C ABI:

| Struct | Change | Risk if unpatched |
|--------|--------|-------------------|
| `llama_model_params` | New first field `dspark_head_source` (pointer) | All subsequent fields misaligned → silent corruption |
| `llama_context_params` | New field `path_kv_mean_center` inserted before `abort_callback` | Callback pointer corrupted → crashes |
| `llama_opt_params` | New trailing field `optimizer_type` (enum) | Benign if unused, but struct size wrong |

Plus one **enum drift**:
- `GGML_TYPE_COUNT`: 43 → 144 (new quantization types added)

The script `scripts/check_binding_compat.py` automates detection of these drifts.

In CI the fix is applied as a real patch file, `patches/0001-prism-ctypes-abi.patch`,
by `scripts/prepare_llama_cpp_python.py`, which also:

- sets `__version__` to the PEP 440 local version, and
- **re-verifies the result** by parsing the patched `_fields_` lists with `ast`
  and asserting each new field sits between the expected neighbours, plus
  `GGML_TYPE_COUNT == 144`.

The verification is static rather than an import, because `llama_cpp.py` pulls
in numpy/diskcache, which are not installed yet at that point in the job. If
upstream ever reorders those structs, `git apply` fails loudly and the job stops
with a pointer to `check_binding_compat.py` — the previous inline `re.sub`
version silently no-op'd and produced a wheel that crashes at inference time.

To re-derive the patch after a fork bump:

```bash
git clone --branch v0.3.35 --depth 1 https://github.com/abetlen/llama-cpp-python /tmp/lcp
# edit /tmp/lcp/llama_cpp/llama_cpp.py to match the fork headers
python3 scripts/check_binding_compat.py \
  --llama-cpp-python /tmp/lcp \
  --llama-cpp /path/to/llama.cpp-fork
cd /tmp/lcp && git diff -- llama_cpp/llama_cpp.py > patches/0001-prism-ctypes-abi.patch
```

## CI design (GitHub Actions → GitHub Pages)

### Matrix

| Job | OS | Runner | CMake flags | Output |
|-----|-----|--------|-------------|--------|
| `metal` | macOS-latest | `macos-latest` (ARM64) | `-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON` | `whl/prism-metal/` |
| `cpu` | macOS-latest | `macos-latest` | (no Metal) | `whl/prism-cpu/` |
| `linux-cpu` | ubuntu-latest | `ubuntu-latest` | `-DGGML_BLAS=ON` | `whl/prism-linux-cpu/` |

*CUDA jobs omitted (no GPU runners on public GitHub). Add self-hosted if needed.*

### Versioning

```
Wheel name: llama_cpp_python-0.3.35+prism.b10743.adfffbe-<platform>.whl
            │               │                │
            │               │                └─ PrismML fork tag (dots for PEP 440)
            │               └─ "prism" local version label
            └─ upstream llama-cpp-python release
```

### Index structure (PEP 503)

```
https://<user>.github.io/llama-cpp-python-prism-wheels/
└── whl/
    ├── prism-metal/                    # macOS ARM64 + Metal
    │   ├── index.html                  # flat listing -> --find-links
    │   └── llama-cpp-python/           # PEP 503 "simple" project page
    │       ├── index.html
    │       └── llama_cpp_python-0.3.35+prism.b10743.adfffbe-py3-none-macosx_*.whl
    ├── prism-cpu/                      # macOS ARM64 CPU-only
    │   └── ... same shape
    └── prism-linux-cpu/                # Linux x86_64 CPU
        └── ... same shape
```

The nested `llama-cpp-python/` directory is **required**, not cosmetic. Given
`--index-url .../whl/prism-metal/`, both pip and uv normalise the project name
and then request `.../whl/prism-metal/llama-cpp-python/`. A flat index at the
backend root 404s there and resolution fails with
`No matching distribution found` — even though the wheel URL is perfectly
reachable. Use `--index-url` (not `--extra-index-url`) only if you also host the
runtime dependencies; otherwise prefer `--extra-index-url` so PyPI still serves
numpy/jinja2/diskcache.

Append-only: **old wheels are never deleted**. Consumers pin exact versions
(`==0.3.35+prism.b10743.adfffbe`); the publish job re-downloads whatever is
already live and republishes it alongside the new build.

### Release workflow

1. Choose a PrismML fork tag (e.g. `prism-b10743-adfffbe`).
2. Push a release tag of **this** repo matching `prism-b*` (e.g. `prism-b10743-adfffbe-v1`).
3. GitHub Actions builds all three matrix jobs, then deploys to Pages.
4. Consumers install from `/whl/<backend>/`.

Via `workflow_dispatch` you can pass `prism_tag` and `llama_cpp_python_version`
explicitly instead of pushing a tag.

Note the two tag namespaces are different things: the **fork** tag
(`prism-b10743-adfffbe`) identifies the llama.cpp commit, while the **release**
tag of this repo (`prism-b10743-adfffbe-v1`) is just a trigger and a build
counter. The wheel version is derived from the fork tag, so re-running the same
fork tag rebuilds the same version string.

### CI gotchas hit while building this

Recording these because each one cost a full debug cycle:

- **`actions/checkout` without `path:`** checks out into the workspace root. A
  second checkout with `path: llama-cpp-python/vendor/llama.cpp` then creates a
  *brand new empty* `llama-cpp-python/` tree next to the real one, and the patch
  step fails with `llama_cpp.py not found` while every preceding step reports
  success. Always set `path:` on the first checkout.
- **Multi-line `python3 -c "…"` inside `run: |`** is invalid YAML. The script
  body sits at column 0, and block scalar content must be indented deeper than
  its key — the run then fails in 0s with "workflow file issue" and no logs. Use
  a heredoc with a quoted delimiter, or (what we do) a checked-in script.
- **`sed -i '' …` is BSD-only.** It works on macOS and fails on
  `ubuntu-latest`, which needs `sed -i …`. The version bump is done in Python.
- **Pages `build_type` is `workflow`**, so the deploy must go through
  `actions/upload-pages-artifact` + `actions/deploy-pages`. Pushing to a
  `gh-pages` branch with `peaceiris/actions-gh-pages` publishes a commit the
  Pages site will never serve.
- **`sparse-checkout` is not free.** `sparse-checkout: scripts` checked out
  `scripts/` but not `patches/`, so the patch step failed with "patch not
  found" on every runner. Only reach for it if the checkout is genuinely large.
- **Bash parameter expansion does not work inside `${{ }}`.**
  `${VAR//-/.}` is evaluated by Actions as a literal string, not by bash. Move it
  into a `run:` block.

## License

- llama.cpp (PrismML fork): MIT
- llama-cpp-python: MIT
- Ternary-Bonsai-2-27B model: Apache-2.0

All compatible with redistribution.# force rebuild Tue Sep 29 05:07:17 +04 2026
