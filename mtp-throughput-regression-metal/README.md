# MTP Throughput Regression on Apple Silicon (Metal)

Qwen MTP models load and produce correct output but generate slower than their non-MTP counterparts on Apple Silicon (Metal backend). Draft tokens are proposed and accepted, but speculative decoding provides no net speedup — it actively degrades throughput at every MTP configuration. Higher draft ceilings make it progressively worse.

## System

| Component | Detail |
|-----------|--------|
| Machine | MacBook Pro 14-inch (2021, MacBookPro18,4) |
| Chip | Apple M1 Max |
| CPU | 10-core (8 Performance + 2 Efficiency) |
| GPU | 24-core, Metal 4 |
| Memory | 32 GB unified |
| OS | macOS 26.4.1 (Build 25E253) |

## Software

| Component | Version |
|-----------|---------|
| llama.cpp (homebrew) | b9330 (commit 328874d05), AppleClang 21.0.0, Metal backend |
| LM Studio | 0.4.14 (build 4) |

## Models

| Model | Format | Quant | Size | Source |
|-------|--------|-------|-----:|--------|
| Qwen3.5-9B | GGUF | Q4_K_M | 5.23 GiB | Qwen/Qwen3.5-9B-GGUF |
| Qwen3.5-9B-MTP | GGUF | Q4_K_M | 5.47 GiB | unsloth/Qwen3.5-9B-MTP-GGUF |
| Qwen3.6-35B-A3B | GGUF | Q4_K_M | 20.56 GiB | Qwen/Qwen3.6-35B-A3B-GGUF |
| Qwen3.6-35B-A3B-MTP | GGUF | Q4_K_XL | 24.64 GiB | unsloth/Qwen3.6-35B-A3B-MTP-GGUF |

## Test 1: llama-server Draft Ceiling Sweep (Qwen3.5-9B-MTP, 2048 tokens)

Prompt: "Write a detailed explanation of how neural network backpropagation works, including the chain rule, gradient descent, vanishing gradients, and practical techniques like batch normalization and skip connections. Include mathematical notation where appropriate."

Each configuration launched as a fresh `llama-server` process. `--spec-type draft-mtp` with `--spec-draft-n-min` / `--spec-draft-n-max` to control floor/ceiling. temperature=0, max_tokens=2048.

| Config | Think ON tok/s | Think OFF tok/s | Draft Accept (ON/OFF) |
|--------|------:|-------:|:---:|
| **Baseline (non-MTP)** | **25.3** | **25.1** | **—** |
| MTP 0/0 | 22.4 | 22.1 | 100% / 100% |
| MTP 0/2 | 21.9 | 21.3 | 76% / 73% |
| MTP 0/6 | 19.3 | 18.3 | 44% / 41% |
| MTP 2/2 | 22.6 | 22.1 | 76% / 73% |
| MTP 2/6 | 19.5 | 18.4 | 44% / 41% |
| MTP 6/6 | 19.1 | 18.2 | 44% / 41% |

### Findings

1. **MTP is a net loss at every configuration.** Even MTP 0/0 with 100% draft acceptance is 11% slower than the non-MTP baseline. Draft evaluation overhead on Metal exceeds any speculative gain.
2. **Ceiling drives the regression, floor has no effect.** 0/6, 2/6, and 6/6 all produce ~19 tok/s. 0/2 and 2/2 both produce ~22 tok/s. Only `--spec-draft-n-max` matters.
3. **More draft tokens = worse acceptance AND more overhead.** max=0: 100% accept, max=2: ~75%, max=6: ~43%. Each step costs throughput.
4. **No MTP configuration beats the non-MTP baseline.**

## Test 2: LM Studio MTP Sweep (Qwen3.5-9B-MTP, short prompt)

MTP min/max set in LM Studio UI. Model reloaded between changes. Prompt: "What is 17 multiplied by 23? Give just the number." temperature=0, max_tokens=1024.

| Model | MTP min/max | Think | Tokens | Time | tok/s |
|-------|:-----------:|-------|-------:|-----:|------:|
| **qwen3.5-9b (base)** | **n/a** | **ON** | **537 ct** | **18.8s** | **28.6** |
| **qwen3.5-9b (base)** | **n/a** | **OFF** | **4 ct** | **0.3s** | **12.0** |
| qwen3.5-9b-mtp | 0/0 | ON | 287 ct | 9.6s | 29.9 |
| qwen3.5-9b-mtp | 0/0 | OFF | 4 ct | 0.5s | 8.5 |
| qwen3.5-9b-mtp | 0/2 | ON | 287 ct | 10.8s | 26.6 |
| qwen3.5-9b-mtp | 0/2 | OFF | 4 ct | 0.4s | 9.3 |
| qwen3.5-9b-mtp | 0/6 | ON | 287 ct | 12.7s | 22.6 |
| qwen3.5-9b-mtp | 0/6 | OFF | 4 ct | 0.6s | 6.6 |
| qwen3.5-9b-mtp | 2/2 | ON | 287 ct | 10.8s | 26.6 |
| qwen3.5-9b-mtp | 2/2 | OFF | 4 ct | 0.4s | 9.1 |
| qwen3.5-9b-mtp | 2/6 | ON | 287 ct | 12.6s | 22.8 |
| qwen3.5-9b-mtp | 2/6 | OFF | 4 ct | 0.6s | 6.9 |
| qwen3.5-9b-mtp | 6/6 | ON | 287 ct | 12.6s | 22.8 |
| qwen3.5-9b-mtp | 6/6 | OFF | 4 ct | 0.6s | 6.8 |

Same ceiling-drives-regression pattern as llama-server. LM Studio compounds the problem further — think OFF throughput drops to 6-9 tok/s vs llama-server's 18-22 tok/s.

## Test 3: LM Studio 27-Model Automated Probe (default MTP settings)

### Qwen3.5-9B: MTP vs non-MTP

| Test | MTP tok/s | Base tok/s | Slowdown |
|------|----------:|----------:|---------:|
| text_baseline (thinking) | 21.5 | 27.2 | 1.3x |
| think_off | 6.3 | 10.0 | 1.6x |
| tool_use | 18.5 | 19.6 | 1.1x |
| json_schema | 22.6 | 23.6 | 1.0x |
| fim | 19.6 | 25.2 | 1.3x |

### Qwen3.6-35B-A3B: MTP vs non-MTP

| Test | MTP tok/s | Base tok/s | Slowdown |
|------|----------:|----------:|---------:|
| text_baseline (thinking) | 5.5 | 25.0 | **5x** |
| think_on | 5.6 | 36.0 | **6x** |
| think_off | 1.0 | 13.8 | **14x** |
| tool_use | 3.4 | 37.2 | **11x** |
| tool_use_think_off | 8.0 | 36.9 | **5x** |
| json_schema | 5.6 | 33.6 | **6x** |
| fim | 5.7 | 39.4 | **7x** |

The 35B MoE MTP variant is **5–14x slower** across all probe types via LM Studio.

## llama-server Logs

### MTP Model Load + Speculative Init

```
llama_model_loader: - kv  32: qwen35.nextn_predict_layers u32 = 1
create_tensor: loading tensor blk.32.nextn.eh_proj.weight
create_tensor: loading tensor blk.32.nextn.enorm.weight
create_tensor: loading tensor blk.32.nextn.hnorm.weight
create_tensor: loading tensor blk.32.nextn.shared_head_norm.weight
srv    load_model: [spec] adding 1808.02 MiB to fit_params_target for device MTL0
srv    load_model: [spec] estimated memory usage of MTP context is 1808.02 MiB
srv    load_model: creating MTP draft context against the target model
common_speculative_impl_draft_mtp: adding speculative implementation 'draft-mtp'
common_speculative_impl_draft_mtp: - n_max=3, n_min=0, p_min=0.00, n_embd=4096, backend_sampling=1
common_speculative_impl_draft_mtp: - gpu_layers=-1, cache_k=f16, cache_v=f16, ctx_tgt=yes, ctx_dft=yes, devices=[default]
srv    load_model: speculative decoding context initialized
srv    load_model: context checkpoints enabled, max = 32, min spacing = 256
```

### Sample Inference Timings (271 tokens, `--spec-type draft-mtp`)

```
prompt eval time =     518.24 ms /    26 tokens (   19.93 ms per token,    50.17 tokens per second)
       eval time =    7551.42 ms /   271 tokens (   27.87 ms per token,    35.89 tokens per second)
      total time =    8069.66 ms /   297 tokens
   graphs reused =         79
draft acceptance = 0.80000 (  192 accepted /   240 generated)
```

## Reproduction

```bash
# llama-server b9330+ required

# 1. Baseline (non-MTP model)
llama-server -m Qwen3.5-9B-Q4_K_M.gguf -ngl 99 --port 8080
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"Write a detailed explanation of how neural network backpropagation works, including the chain rule, gradient descent, vanishing gradients, and practical techniques like batch normalization and skip connections."}],
    "max_tokens":2048, "temperature":0
  }'
# Note tok/s from response timings

# 2. MTP model with speculative decoding enabled
llama-server -m Qwen3.5-9B-MTP-Q4_K_M.gguf -ngl 99 --port 8080 \
  --spec-type draft-mtp --spec-draft-n-max 6
# Same curl — compare tok/s and draft acceptance

# 3. Vary --spec-draft-n-max: 0, 2, 6
# Observe: higher ceiling = lower tok/s, lower acceptance
```

## Related Issues

- ggml-org/llama.cpp: [#23533](https://github.com/ggml-org/llama.cpp/issues/23533) (SYCL no speedup), [#23203](https://github.com/ggml-org/llama.cpp/issues/23203) (SYCL slowdown) — different backends, same symptom class
- lmstudio-ai/lmstudio-bug-tracker: [#1941](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1941), [#1948](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1948), [#1951](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1951) — MTP load failures, different symptom

No existing report of MTP throughput regression on Metal/Apple Silicon.
