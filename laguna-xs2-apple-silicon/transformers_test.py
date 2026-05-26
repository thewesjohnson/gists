"""Test: Can HuggingFace Transformers load Laguna-XS.2 on Apple Silicon?

Attempts three loading strategies in order:
  1. Local GGUF Q4_K_M via transformers' GGUF support (skips if not found)
  2. Upstream repo (poolside/Laguna-XS.2) with load_in_4bit
  3. Upstream repo in float16 on MPS (expect OOM on <64 GB)

Usage:
    pip install transformers torch
    python laguna_transformers_test.py

Requires: Python 3.12+, transformers >=4.45, torch >=2.4 with MPS support.
"""

import sys
import time
import traceback

UPSTREAM_REPO = "poolside/Laguna-XS.2"
GGUF_SEARCH_PATHS = [
    "~/.lmstudio/models/Lucebox/Laguna-XS.2-GGUF/laguna-xs2-Q4_K_M.gguf",
    "~/.ollama/models/blobs",
    "~/laguna-xs2-Q4_K_M.gguf",
]


def find_gguf():
    """Search common paths for the Laguna GGUF file."""
    import glob
    import os

    for p in GGUF_SEARCH_PATHS:
        expanded = os.path.expanduser(p)
        if os.path.isfile(expanded):
            return expanded
        for match in glob.glob(os.path.join(expanded, "*laguna*Q4_K_M*.gguf")):
            return match
    return None


def test_gguf():
    """Attempt 1: Load local GGUF via transformers."""
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gguf_path = find_gguf()
    print(f"\n{'='*60}")
    print(f"ATTEMPT 1: GGUF via transformers")
    if gguf_path is None:
        print("No local GGUF found — skipping. Download from:")
        print("  huggingface.co/Lucebox/Laguna-XS.2-GGUF")
        print(f"{'='*60}\n")
        print("RESULT: SKIP")
        return None
    print(f"Path: {gguf_path}")
    print(f"Exists: {os.path.exists(gguf_path)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            os.path.dirname(gguf_path),
            gguf_file=os.path.basename(gguf_path),
            device_map="mps",
        )
        tok = AutoTokenizer.from_pretrained(
            os.path.dirname(gguf_path),
            gguf_file=os.path.basename(gguf_path),
        )
        elapsed = time.time() - t0
        print(f"LOADED in {elapsed:.1f}s")
        print(f"Model type: {type(model).__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        ids = tok("def hello():", return_tensors="pt").input_ids.to("mps")
        out = model.generate(ids, max_new_tokens=50)
        print(f"Output: {tok.decode(out[0], skip_special_tokens=True)}")
        print("\nRESULT: PASS")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAILED after {elapsed:.1f}s")
        print(f"Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\nRESULT: FAIL")
        return False


def test_upstream_4bit():
    """Attempt 2: Load upstream model with bitsandbytes 4-bit."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"ATTEMPT 2: Upstream repo with load_in_4bit")
    print(f"Repo: {UPSTREAM_REPO}")
    print(f"{'='*60}\n")

    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(UPSTREAM_REPO)
        model = AutoModelForCausalLM.from_pretrained(
            UPSTREAM_REPO,
            load_in_4bit=True,
            device_map="auto",
        )
        elapsed = time.time() - t0
        print(f"LOADED in {elapsed:.1f}s")
        print(f"Model type: {type(model).__name__}")

        ids = tok("def hello():", return_tensors="pt").input_ids
        out = model.generate(ids, max_new_tokens=50)
        print(f"Output: {tok.decode(out[0], skip_special_tokens=True)}")
        print("\nRESULT: PASS")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAILED after {elapsed:.1f}s")
        print(f"Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\nRESULT: FAIL")
        return False


def test_upstream_mps_float16():
    """Attempt 3: Load upstream model in float16 on MPS (will likely OOM)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"ATTEMPT 3: Upstream repo float16 on MPS (expect OOM)")
    print(f"Repo: {UPSTREAM_REPO}")
    print(f"{'='*60}\n")

    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(UPSTREAM_REPO)
        model = AutoModelForCausalLM.from_pretrained(
            UPSTREAM_REPO,
            torch_dtype="float16",
            device_map="mps",
        )
        elapsed = time.time() - t0
        print(f"LOADED in {elapsed:.1f}s")
        print(f"Model type: {type(model).__name__}")

        ids = tok("def hello():", return_tensors="pt").input_ids.to("mps")
        out = model.generate(ids, max_new_tokens=50)
        print(f"Output: {tok.decode(out[0], skip_special_tokens=True)}")
        print("\nRESULT: PASS")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAILED after {elapsed:.1f}s")
        print(f"Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\nRESULT: FAIL")
        return False


if __name__ == "__main__":
    import torch

    print("Laguna-XS.2 Transformers Load Test")
    print(f"torch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    import transformers
    print(f"transformers: {transformers.__version__}")

    results = {}

    # Attempt 1: Local GGUF
    results["gguf"] = test_gguf()

    if results["gguf"] is not True:
        # Attempt 2: upstream + 4bit quantization
        results["4bit"] = test_upstream_4bit()

    if not results.get("4bit", False) and results["gguf"] is not True:
        # Attempt 3: upstream + float16 (expect OOM on <64GB)
        results["float16_mps"] = test_upstream_mps_float16()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for k, v in results.items():
        status = "PASS" if v is True else ("SKIP" if v is None else "FAIL")
        print(f"  {k}: {status}")
    sys.exit(0 if any(v is True for v in results.values()) else 1)
