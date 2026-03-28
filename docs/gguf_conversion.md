# GGUF Conversion — Seventh Shard LoRA Fused Models

Convert a seventh_shard MLX-fused LoRA model to a llama.cpp-compatible Q4_K_M GGUF
for use in federated_village.

---

## Environment requirements

All conversion steps run in the `seventh_gen` conda env.

Required packages (all installed 2026-03-28):

```
mlx-lm         # fuse + dequantize
transformers   # convert_hf_to_gguf.py
torch          # convert_hf_to_gguf.py
numpy          # convert_hf_to_gguf.py
gguf==0.18.0   # convert_hf_to_gguf.py (must match converter version)
mistral_common # convert_hf_to_gguf.py (tokenizer support)
```

Install missing deps:
```bash
/opt/anaconda3/envs/seventh_gen/bin/pip install torch gguf mistral-common
```

**Converter script:** use the one bundled with `llama-cpp-python` in the `village` env:
```
/opt/anaconda3/envs/village/lib/python3.11/site-packages/bin/convert_hf_to_gguf.py
```
Do NOT use the homebrew llama.cpp converter — it targets a newer unreleased `gguf` version
that is incompatible with PyPI `gguf` 0.18.0.

**Quantize binary:** homebrew llama.cpp:
```
/opt/homebrew/bin/llama-quantize
```

---

## Step 1 — Fuse LoRA adapter and dequantize to bfloat16 safetensors

```bash
/opt/anaconda3/envs/seventh_gen/bin/mlx_lm fuse \
  --model ~/models/<base-model-mlx-4bit> \
  --adapter-path ~/seventh_shard/adapters_<model> \
  --dequantize \
  --save-path /tmp/<model>_fused_bf16
```

This produces ~14GB of bfloat16 safetensors in `/tmp/<model>_fused_bf16/`.
Requires ~14GB free in /tmp and enough RAM to hold the dequantized weights.
Do not run while any Village inference session is active.

Copy tokenizer files from the base model if they are missing from the output:
```bash
for f in tokenizer.json tokenizer_config.json special_tokens_map.json merges.txt vocab.json; do
  cp ~/models/<base-model-mlx-4bit>/$f /tmp/<model>_fused_bf16/ 2>/dev/null || true
done
```

---

## Step 2 — Convert bfloat16 safetensors to float16 GGUF

```bash
/opt/anaconda3/envs/seventh_gen/bin/python \
  /opt/anaconda3/envs/village/lib/python3.11/site-packages/bin/convert_hf_to_gguf.py \
  /tmp/<model>_fused_bf16 \
  --outtype f16 \
  --outfile /tmp/<model>_fused_f16.gguf
```

Produces ~14–15GB f16 GGUF. Verify with:
```bash
ls -lh /tmp/<model>_fused_f16.gguf
```

---

## Step 3 — Quantize to Q4_K_M

```bash
llama-quantize \
  /tmp/<model>_fused_f16.gguf \
  ~/models/<model-dir>/<model>-Q4_K_M.gguf \
  Q4_K_M
```

Target size: ~4.4GB for a 7B model. Verify:
```bash
ls -lh ~/models/<model-dir>/<model>-Q4_K_M.gguf
```

---

## Step 4 — Clean up

```bash
rm -rf /tmp/<model>_fused_bf16 /tmp/<model>_fused_f16.gguf
```

Recovers ~28GB of tmp space.

---

## Step 5 — Test in federated_village

```bash
cd ~/federated_village
VILLAGE_MODEL=~/models/<model-dir>/<model>-Q4_K_M.gguf \
VILLAGE_MODEL_NAME=<model>-Q4_K_M \
/opt/anaconda3/envs/village/bin/python run_session.py --scenario scenarios/scenario_06.md
```

---

## Completed conversions

| Model | GGUF path | Date | Notes |
|---|---|---|---|
| Qwen2.5-7B-seventh-gen | `~/models/Qwen2.5-7B-seventh-gen-fused/Qwen2.5-7B-seventh-gen-Q4_K_M.gguf` | 2026-03-28 | Pre-SC06-fix adapter; escalates on SC06 in shard context |

---

## Known issues / gotchas

- `mlx_lm fuse --export-gguf` does not support Qwen2 architecture — always use the two-step path
- The homebrew `convert_hf_to_gguf.py` targets unreleased `gguf` internals — use the village env's copy
- MLX fused models store weights as 4-bit quantized safetensors — `--dequantize` is mandatory
- Do not run conversion while Village inference is active — dequantize step requires ~14GB RAM
