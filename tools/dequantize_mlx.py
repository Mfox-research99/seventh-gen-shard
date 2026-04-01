#!/usr/bin/env python3
"""
dequantize_mlx.py — Dequantize a fused MLX model to clean bf16 SafeTensors

After mlx_lm.fuse with a quantized base model, the output contains MLX artifact
tensors (*.scales, *.biases, uint32 packed weights) that convert_hf_to_gguf.py
cannot handle. This script dequantizes the fused model to standard bf16, removing
the artifacts, so it can be converted to GGUF.

Usage:
    python dequantize_mlx.py --fused-path <fused_mlx_dir> --output-path <dequant_dir>

Developed in Phase 7 (Anubis GGUF conversion) and Phase 8 (Humanist LoRA GGUF).
See: Sessions/2026-03-28-federated-village-phase8-anubis-gguf-conversion.md
"""

import argparse
import sys
from pathlib import Path


def dequantize(fused_path: str, output_path: str) -> None:
    try:
        import mlx_lm.utils as u
    except ImportError:
        print("ERROR: mlx_lm not available. Run in seventh_gen conda env.", file=sys.stderr)
        sys.exit(1)

    fused_path  = Path(fused_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[dequantize] Loading fused model from: {fused_path}")

    # Monkey-patch load_adapters to bypass adapter check during load
    # (the fused model dir doesn't have a separate adapter dir)
    orig_load_adapters = u.load_adapters
    u.load_adapters = lambda model, path, **kw: model

    try:
        result = u.load(str(fused_path), return_config=True)
    finally:
        u.load_adapters = orig_load_adapters

    if len(result) == 3:
        model, tokenizer, config = result
    else:
        model, tokenizer = result
        config = {}

    print(f"[dequantize] Dequantizing model...")
    from mlx_lm.utils import dequantize_model
    model = dequantize_model(model)

    # Remove quantization config entries that would confuse convert_hf_to_gguf.py
    for key in ("quantization", "quantization_config", "_quantization"):
        config.pop(key, None)

    print(f"[dequantize] Saving dequantized model to: {output_path}")
    u.save(str(output_path), str(fused_path), model, tokenizer, config)

    print(f"[dequantize] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dequantize fused MLX model to bf16 SafeTensors")
    parser.add_argument("--fused-path",  required=True, help="Path to fused MLX model dir")
    parser.add_argument("--output-path", required=True, help="Path to save dequantized model")
    args = parser.parse_args()
    dequantize(args.fused_path, args.output_path)
