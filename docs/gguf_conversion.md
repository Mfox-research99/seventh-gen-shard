# GGUF Conversion — Seventh Shard LoRA Fused Models

Convert a seventh_shard MLX-fused LoRA model to a llama.cpp-compatible Q4_K_M GGUF
for use in federated_village.

**Last validated:** 2026-04-01 (Humanist LoRA — Anubis-Mini-8B-humanist-Q4_K_M.gguf)

---

## Summary

The core problem: `mlx_lm.fuse` with a **quantized base model** (e.g., Anubis-Mini-8B-mlx-4bit)
produces output SafeTensors containing MLX artifact tensors (`*.biases`, `*.scales`, uint32 packed
weights). These crash `convert_hf_to_gguf.py` with errors like:

```
ValueError: Can not map tensor 'lm_head.biases'
```

The fix is a **4-step pipeline** that separates fuse, dequantize, convert, and quantize:

```
mlx_lm.fuse → dequantize_mlx.py → convert_hf_to_gguf.py → llama-quantize
```

The `ensure_humanist_gguf()` function in `federated_village/benchmark_suite.py` is the
canonical automatable version of this pipeline.

---

## Environment requirements

| Step | Environment | Binary/Script |
|---|---|---|
| Fuse | `seventh_gen` conda env | `/opt/anaconda3/envs/seventh_gen/bin/mlx_lm.fuse` |
| Dequantize | `seventh_gen` conda env | `/opt/anaconda3/envs/seventh_gen/bin/python` + `seventh_shard/tools/dequantize_mlx.py` |
| Convert | **`village` conda env** | `/opt/anaconda3/envs/village/bin/python` + `/opt/homebrew/bin/convert_hf_to_gguf.py` |
| Quantize | system | `/opt/homebrew/bin/llama-quantize` |

**Critical env note:**
- Use `village` Python for `convert_hf_to_gguf.py` — it has a compatible `gguf` module
- Do NOT use `seventh_gen` Python for convert — it throws `AttributeError: MISTRAL4` (gguf version mismatch)
- The Homebrew `/opt/homebrew/bin/convert_hf_to_gguf.py` IS correct — no need to clone llama.cpp source

**Do NOT run conversion while Village inference is active** — dequantize step requires ~14GB RAM.

---

## Step 1 — Fuse LoRA adapter

```bash
/opt/anaconda3/envs/seventh_gen/bin/mlx_lm.fuse \
  --model ~/models/<base-model-mlx-4bit> \
  --adapter-path ~/seventh_shard/adapters/<adapter-dir> \
  --save-path ~/models/<model-name>-fused
```

**No `--dequantize` flag. No `--export-gguf` flag.**

- `--dequantize` is broken when the base is quantized (produces unusable output)
- `--export-gguf` throws `NotImplementedError: Conversion of quantized models is not yet supported`
- The output is a fused MLX SafeTensor dir with artifact tensors — Step 2 cleans them

Output: `~/models/<model-name>-fused/` (~same size as base model)
Time: ~10–15s

---

## Step 2 — Dequantize to clean bf16 SafeTensors

```bash
/opt/anaconda3/envs/seventh_gen/bin/python \
  ~/seventh_shard/tools/dequantize_mlx.py \
  --fused-path ~/models/<model-name>-fused \
  --output-path ~/models/<model-name>-dequant
```

This removes the MLX artifact tensors (`*.biases`, `*.scales`, uint32 packed weights) by
loading the fused model via the Python API, calling `dequantize_model()`, and saving clean
bf16 SafeTensors in standard HuggingFace format.

Output: `~/models/<model-name>-dequant/` (~14–16GB bf16 SafeTensors)
Time: ~60–90s

**See `tools/dequantize_mlx.py` for implementation details.** The script uses a
`load_adapters` monkey-patch to bypass the adapter-dir check during load of an already-fused
model (where no separate adapter directory exists).

---

## Step 3 — Convert bf16 SafeTensors → f16 GGUF

```bash
/opt/anaconda3/envs/village/bin/python \
  /opt/homebrew/bin/convert_hf_to_gguf.py \
  ~/models/<model-name>-dequant \
  --outfile ~/models/<model-name>-gguf/<model-name>-f16.gguf \
  --outtype f16
```

Output: `<model-name>-f16.gguf` (~15–16GB)
Time: ~40–60s

---

## Step 4 — Quantize to Q4_K_M

```bash
mkdir -p ~/models/<model-name>-gguf
/opt/homebrew/bin/llama-quantize \
  ~/models/<model-name>-gguf/<model-name>-f16.gguf \
  ~/models/<model-name>-gguf/<model-name>-Q4_K_M.gguf \
  Q4_K_M
```

Output: `<model-name>-Q4_K_M.gguf` (~4.6GB for 8B model)
Time: ~90–120s

---

## Step 5 — Clean up intermediates

```bash
rm -rf ~/models/<model-name>-fused
rm -rf ~/models/<model-name>-dequant
rm ~/models/<model-name>-gguf/<model-name>-f16.gguf
```

Recovers ~30GB of space.

---

## Step 6 — Test in federated_village

```bash
cd ~/federated_village
VILLAGE_MODEL=~/models/<model-name>-gguf/<model-name>-Q4_K_M.gguf \
VILLAGE_MODEL_NAME=<model-name> \
/opt/anaconda3/envs/village/bin/python run_session.py --scenario scenarios/scenario_06.md
```

---

## Automated version

`federated_village/benchmark_suite.py` → `ensure_humanist_gguf()` runs this full pipeline
automatically, with resume support (skips steps whose output already exists). Use as a
reference implementation when automating future conversions.

---

## Completed conversions

| Model | GGUF path | Adapter | Date | Notes |
|---|---|---|---|---|
| Anubis-Mini-8B-seventh-gen | `~/models/Anubis-Mini-8B-seventh-gen-gguf/Anubis-Mini-8B-seventh-gen-Q4_K_M.gguf` | `adapters_v2/` (grief/Elder) | 2026-03-28 | Phase 7. Validated SC02/SC04/SC06. Active 4th Village model. |
| Anubis-Mini-8B-humanist | `~/models/Anubis-Mini-8B-humanist-gguf/Anubis-Mini-8B-humanist-Q4_K_M.gguf` | `adapters/humanist_v1/` (Humanist char.) | 2026-04-01 | Iter 50 checkpoint. Benchmark comparison in progress. |

---

## Known issues / gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `NotImplementedError: Conversion of quantized models is not yet supported` | Used `--export-gguf` with quantized base | Remove `--export-gguf`; use the 4-step pipeline |
| `ValueError: Can not map tensor 'lm_head.biases'` | Skipped dequantize step; artifact tensors present | Run Step 2 (dequantize_mlx.py) before converting |
| `AttributeError: MISTRAL4` | Used `seventh_gen` Python for convert_hf_to_gguf.py | Use `village` Python instead |
| `mlx_lm.fuse` crashes on fused model with no adapters dir | Trying to re-fuse an already-fused model | Use `dequantize_mlx.py` directly on fused model; it bypasses adapter check |
| OOM during dequantize | Tried to run while Village inference was active | Kill inference process first; dequantize needs ~14GB |
| `--dequantize` flag produces garbage output | Broken with 4-bit quantized base in current mlx_lm | Do not use `--dequantize` flag; use the Python API in `dequantize_mlx.py` |
