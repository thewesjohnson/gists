# Laguna-XS.2 on Apple Silicon — Reproduction Scripts

Reproduction scripts for testing [Poolside Laguna-XS.2](https://huggingface.co/poolside/Laguna-XS.2) on Apple Silicon Macs.

## `transformers_test.py`

Tests whether HuggingFace Transformers can load Laguna-XS.2 on MPS (Apple Silicon).

Tries three strategies in order:
1. Local GGUF via transformers' GGUF loader (auto-detects common paths, skips if not found)
2. Upstream `poolside/Laguna-XS.2` with `load_in_4bit` (bitsandbytes)
3. Upstream `poolside/Laguna-XS.2` in float16 on MPS

### Usage

```bash
pip install transformers torch
python transformers_test.py
```

### Results (2026-05-26)

Tested on MacBook Pro M1 Max, 32 GB, macOS 26.4.1, transformers 5.9.0, torch 2.11.0.

| Attempt | Result | Error |
|---------|--------|-------|
| GGUF | FAIL | `GGUF model with architecture laguna is not supported yet` |
| load_in_4bit | FAIL | `LagunaForCausalLM.__init__() got an unexpected keyword argument 'load_in_4bit'` (bitsandbytes is CUDA-only) |
| float16 MPS | FAIL | `Invalid buffer size: 62.29 GiB` (exceeds 32 GB) |

Transformers 5.9.0 recognizes the architecture (`LagunaForCausalLM` exists), but there is no viable quantized path for Apple Silicon MPS.

### Related issues

- [ggml-org/llama.cpp#23249](https://github.com/ggml-org/llama.cpp/issues/23249) — llama.cpp architecture support
- [lmstudio-ai/lmstudio-bug-tracker#1968](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1968) — LM Studio
- [ollama/ollama#15892](https://github.com/ollama/ollama/issues/15892) — Ollama output artefacts
