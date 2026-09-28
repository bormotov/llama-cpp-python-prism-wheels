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
| Local spike build (Metal) | ✅ Complete (`0.3.35+prism.b10743.adfffbe`) |
| ABI compatibility verified | ✅ (3 struct layout fixes, enum bump) |
| Basic import / struct test | ✅ |
| Model load test (Bonsai 27B PQ2_0) | ⏳ Download in progress |
| Back-compat test (Gemma 12B) | ⏳ Pending |
| GitHub Actions CI (Metal + CPU) | 📋 Designed below |
| GitHub Pages index | 📋 Designed below |

## Quick start (consumer)

```toml
# pyproject.toml or uv.toml
[[tool.uv.index]]
name = "llama-prism-metal"
url = "https://<YOUR_GH_USER>.github.io/llama-cpp-python-prism-wheels/whl/prism-metal"
explicit = true

[tool.uv.sources]
llama-cpp-python = { index = "llama-prism-metal", marker = "sys_platform == 'darwin'" }
```

```bash
uv sync  # installs the PrismML wheel on macOS, stock wheel elsewhere
```

Then in Python:

```python
from llama_cpp import Llama

# Bonsai 27B (requires enable_thinking=False to disable reasoning trace)
llm = Llama(
    model_path="Ternary-Bonsai-2-27B-PQ2_0.gguf",
    n_gpu_layers=-1,          # Metal offload
    n_ctx=262144,             # model's native context
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

# 3. Patch the ctypes binding for struct layout drift (see scripts/check_binding_compat.py)
#    - llama_model_params: prepend `dspark_head_source` (void*)
#    - llama_context_params: insert `path_kv_mean_center` (char*) before abort_callback
#    - llama_opt_params: append `optimizer_type` (int)
#    - GGML_TYPE_COUNT: 43 → 144

# 4. Bump version to PEP 440 local version
#    __version__ = "0.3.35+prism.b10743.adfffbe"

# 5. Build with Metal
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
├── whl/
│   ├── prism-metal/           # macOS ARM64 + Metal
│   │   ├── index.html
│   │   └── llama_cpp_python-0.3.35+prism.b10743.adfffbe-cp311-*.whl
│   ├── prism-cpu/             # macOS ARM64 CPU-only
│   │   ├── index.html
│   │   └── ...
│   └── prism-linux-cpu/       # Linux x86_64 CPU
│       ├── index.html
│       └── ...
```

Append-only: **old wheels are never deleted**. Consumers pin exact versions (`==0.3.35+prism.b10743.adfffbe`).

### Release workflow

1. Tag the fork commit: `git tag prism-bXXXXXX-<shortsha>` (or use existing PrismML tag)
2. Bump `__version__` in `llama_cpp/__init__.py`
3. Push tag → GitHub Actions builds all matrix jobs
4. On success, `gh-pages` branch updated with new wheels + regenerated `index.html`

## License

- llama.cpp (PrismML fork): MIT
- llama-cpp-python: MIT
- Ternary-Bonsai-2-27B model: Apache-2.0

All compatible with redistribution.