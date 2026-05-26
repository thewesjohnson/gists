# Laguna-XS.2 on Apple Silicon

Testing [Poolside Laguna-XS.2](https://huggingface.co/poolside/Laguna-XS.2) (33B total / 3B active MoE, 256 experts + 1 shared, Apache 2.0) across every local inference runtime on Apple Silicon.

## Full runtime status (2026-05-26)

Tested on MacBook Pro M1 Max, 32 GB unified memory, macOS 26.4.1.

| Runtime | Version | Result |
|---------|---------|--------|
| llama.cpp | build 9330 | `unknown model architecture: 'laguna'` |
| LM Studio | 0.4.14+4 | Fails to load (both GGUF and MLX variants) |
| mlx-lm | 0.31.3 | Architecture not recognized |
| Ollama | 0.24.0 | Loads but garbled output ([#15892](https://github.com/ollama/ollama/issues/15892)) |
| vLLM Metal | 0.2.0 (Apple Silicon plugin) | Not in supported models list |
| HF Transformers | 5.9.0 | `LagunaForCausalLM` exists but no quantized MPS path (see below) |
| vLLM (mainline, GPU) | 0.21.0 | Works ([vllm-project/vllm#41129](https://github.com/vllm-project/vllm/pull/41129)) |

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

### Results

| Attempt | Result | Error |
|---------|--------|-------|
| GGUF | FAIL | `GGUF model with architecture laguna is not supported yet` |
| load_in_4bit | FAIL | `LagunaForCausalLM.__init__() got an unexpected keyword argument 'load_in_4bit'` (bitsandbytes is CUDA-only) |
| float16 MPS | FAIL | `Invalid buffer size: 62.29 GiB` (exceeds 32 GB) |

Transformers 5.9.0 recognizes the architecture (`LagunaForCausalLM` exists), but there is no viable quantized path for Apple Silicon MPS.

## Models tested

- GGUF Q4_K_M: [`Lucebox/Laguna-XS.2-GGUF`](https://huggingface.co/Lucebox/Laguna-XS.2-GGUF) (`d0be991`)
- MLX 4-bit: [`mlx-community/Laguna-XS.2-4bit`](https://huggingface.co/mlx-community/Laguna-XS.2-4bit) (`799182f`)
- Upstream: [`poolside/Laguna-XS.2`](https://huggingface.co/poolside/Laguna-XS.2)

## Related issues

- [ggml-org/llama.cpp#23249](https://github.com/ggml-org/llama.cpp/issues/23249) — llama.cpp architecture support (primary blocker)
- [lmstudio-ai/lmstudio-bug-tracker#1968](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1968) — LM Studio
- [ollama/ollama#15892](https://github.com/ollama/ollama/issues/15892) — Ollama output artefacts
