# MTP Throughput Regression on Apple Silicon (Metal)

Qwen MTP models load and produce correct output but generate slower than their non-MTP counterparts on Apple Silicon. Draft tokens are detected but speculative decoding provides no speedup — it actively degrades throughput. Higher MTP draft ceilings make it progressively worse.

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

## Test Conditions

- Each model loaded individually (full unload between models)
- Memory settled (swap < 500MB) before each load
- Prompt: "What is 17 multiplied by 23? Give just the number."
- temperature=0, max_tokens=1024
- Thinking ON = default; Thinking OFF = assistant prefill `<think>\n\n</think>\n`
- MTP min/max configured via LM Studio UI, model reloaded between changes

## Results: MTP Draft Ceiling Sweep (Qwen3.5-9B-MTP vs baseline)

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

### Findings

1. **Ceiling drives the regression, floor has no effect.** 0/6, 2/6, and 6/6 all produce ~22.8 tok/s (think ON). 0/2 and 2/2 both produce 26.6 tok/s. The min parameter is irrelevant.
2. **More draft tokens = slower.** max=0: 29.9 → max=2: 26.6 → max=6: 22.6 tok/s (think ON). Each increase in ceiling degrades throughput.
3. **MTP 0/0 matches base on think ON** (29.9 vs 28.6) but is **30% slower on think OFF** (8.5 vs 12.0). Even with drafting effectively disabled, the MTP model has overhead.
4. **No MTP configuration beats the non-MTP baseline** for think OFF throughput.

## Results: 27-Model Automated Probe (default MTP settings)

### Qwen3.5-9B: MTP vs non-MTP

| Test | MTP tok/s | Base tok/s | Slowdown |
|------|----------:|----------:|---------:|
| text_baseline (thinking) | 21.5 | 27.2 | 1.3x |
| think_off | 6.3 | 10.0 | 1.6x |
| tool_use | 18.5 | 19.6 | 1.1x |
| json_schema | 22.6 | 23.6 | 1.0x |
| fim | 19.6 | 25.2 | 1.3x |

Mild regression (~1.3x average).

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

Severe regression. The 35B MoE MTP variant is **5–14x slower** across all probe types.

## Reproduction

```bash
# Requires llama-server (homebrew llama.cpp b9330+) or LM Studio 0.4.14+

# 1. Load non-MTP baseline
llama-server -m Qwen3.5-9B-Q4_K_M.gguf -ngl 99

# Test with thinking ON (default)
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"What is 17 multiplied by 23? Give just the number."}],
    "max_tokens":1024,
    "temperature":0
  }'

# Test with thinking OFF (assistant prefill)
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[
      {"role":"user","content":"What is 17 multiplied by 23? Give just the number."},
      {"role":"assistant","content":"<think>\n\n</think>\n","prefix":true}
    ],
    "max_tokens":1024,
    "temperature":0
  }'

# 2. Load MTP variant (with draft model support)
llama-server -m Qwen3.5-9B-MTP-Q4_K_M.gguf -ngl 99

# Same curl commands — compare tok/s in response usage field
```

## Related Issues

- ggml-org/llama.cpp: [#23533](https://github.com/ggml-org/llama.cpp/issues/23533) (SYCL no speedup), [#23203](https://github.com/ggml-org/llama.cpp/issues/23203) (SYCL slowdown)
- lmstudio-ai/lmstudio-bug-tracker: [#1941](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1941), [#1948](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1948), [#1951](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1951) (MTP load failures — different symptom)

No existing report of MTP throughput regression on Metal/Apple Silicon.
