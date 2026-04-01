#!/usr/bin/env python3
"""
seventh_shard/dataset/prepare_humanist_mlx.py
Convert humanist_dataset_v1.jsonl → MLX-LM chat format for LoRA training.

Mirrors the pattern of prepare_mlx.py (Grief Horizon / Elder character).
Key differences:
  - System prompt: The_Humanist.md (Stage 3 WitnessPause section trimmed —
    Village operational machinery, not Humanist character)
  - No verdict stratification: all 54 entries are the same character voice
    mode; split is purely random
  - Train/valid: ~90/10 split

Target base models (either works — pick one):
  NeMo 12B  — richer language output, better prose quality
              HuggingFace: mistralai/Mistral-Nemo-Instruct-2407
  Anubis 8B — lighter, already has Seventh Gen training, faster on M1
              HuggingFace: NousResearch/Hermes-3-Llama-3.1-8B (Anubis base)

MLX LoRA training command (after this script runs):
  mlx_lm.lora \\
    --model <hf_model_path_or_id> \\
    --train \\
    --data seventh_shard/dataset/humanist/ \\
    --iters 600 \\
    --batch-size 2 \\
    --lora-layers 16 \\
    --adapter-path seventh_shard/adapters/humanist_v1/

Fuse adapter → GGUF after training:
  mlx_lm.fuse \\
    --model <hf_model_path_or_id> \\
    --adapter-path seventh_shard/adapters/humanist_v1/ \\
    --save-path seventh_shard/models/humanist_v1_fused/
  # Then convert fused MLX model → GGUF with llama.cpp convert-hf-to-gguf.py

Output:
  dataset/humanist/train.jsonl   — ~48 examples
  dataset/humanist/valid.jsonl   — ~6 examples
"""

import json
import random
import re
import sys
from pathlib import Path

SHARD_ROOT   = Path(__file__).resolve().parent.parent
VILLAGE_ROOT = SHARD_ROOT.parent / "federated_village"

HUMANIST_PROMPT_FILE = VILLAGE_ROOT / "prompts" / "The_Humanist.md"
DATASET_FILE         = Path(__file__).parent / "humanist_dataset_v1.jsonl"
OUTPUT_DIR           = Path(__file__).parent / "humanist"

VALID_FRACTION = 0.10  # ~10% held out for validation
SEED = 42


# ── Prompt preparation ────────────────────────────────────────────────────────

def build_generation_prompt(full_prompt: str) -> str:
    """
    Strip Stage 3 WitnessPause operational section from The_Humanist.md.
    Keeps: Purpose, Core Orientation, Key Functions, Principled Refusals,
           On Engagement, Voice and Texture, A Note on This Role.
    Trims: Stage 3 mode machinery (reinforce_pause / refine_burden /
           conditions_for_continuation) — Village plumbing, not character.
    """
    stage3_marker = "## Stage 3:"
    voice_marker  = "## Voice and Texture"

    if stage3_marker not in full_prompt:
        return full_prompt

    before = full_prompt[:full_prompt.index(stage3_marker)].rstrip()

    if voice_marker in full_prompt:
        after = full_prompt[full_prompt.index(voice_marker):]
        return before + "\n\n---\n\n" + after
    return before


def load_system_prompt() -> str:
    if not HUMANIST_PROMPT_FILE.exists():
        print(f"Error: Humanist prompt not found: {HUMANIST_PROMPT_FILE}", file=sys.stderr)
        sys.exit(1)
    full = HUMANIST_PROMPT_FILE.read_text(encoding="utf-8").strip()
    trimmed = build_generation_prompt(full)
    removed = len(full) - len(trimmed)
    print(f"System prompt: {len(trimmed)} chars ({removed} chars trimmed — Stage 3)")
    return trimmed


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset(system_prompt: str) -> list[dict]:
    if not DATASET_FILE.exists():
        print(f"Error: Dataset not found: {DATASET_FILE}", file=sys.stderr)
        sys.exit(1)

    examples = []
    with open(DATASET_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            # Build user content: instruction + historical context
            user_content = entry["instruction"].strip()
            if entry.get("context"):
                user_content += f"\n\n{entry['context'].strip()}"

            examples.append({
                "source_id": entry.get("source_id", ""),
                "domain":    entry.get("domain", ""),
                "messages": [
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": entry["response"].strip()},
                ],
            })

    return examples


# ── Train / valid split ───────────────────────────────────────────────────────

def split(examples: list[dict], valid_fraction: float, seed: int) -> tuple[list, list]:
    """
    Random split with domain-aware reporting.
    No verdict stratification needed — all entries are the same Humanist
    character voice. We do shuffle before splitting so no single domain
    ends up entirely in valid.
    """
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)

    n_valid = max(1, round(len(shuffled) * valid_fraction))
    valid = shuffled[:n_valid]
    train = shuffled[n_valid:]
    return train, valid


# ── Write output ──────────────────────────────────────────────────────────────

def write_split(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            # MLX-LM expects {"messages": [...]} format
            f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Source:  {DATASET_FILE}")
    print(f"Output:  {OUTPUT_DIR}/")
    print()

    system_prompt = load_system_prompt()
    examples = load_dataset(system_prompt)
    print(f"Loaded:  {len(examples)} examples")

    # Domain distribution report
    from collections import Counter
    domains = Counter(ex["domain"] for ex in examples)
    print("Domains:")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        print(f"  {domain:<35} {count}")
    print()

    train, valid = split(examples, VALID_FRACTION, SEED)

    train_path = OUTPUT_DIR / "train.jsonl"
    valid_path = OUTPUT_DIR / "valid.jsonl"
    write_split(train, train_path)
    write_split(valid, valid_path)

    print(f"Train:   {len(train)} examples → {train_path}")
    print(f"Valid:   {len(valid)} examples → {valid_path}")
    print()
    print("Next steps:")
    print("  1. Choose base model (NeMo 12B or Anubis 8B)")
    print("  2. mlx_lm.lora --model <model> --train --data dataset/humanist/ \\")
    print("       --iters 600 --batch-size 2 --lora-layers 16 \\")
    print("       --adapter-path adapters/humanist_v1/")
    print("  3. Evaluate on held-out Village scenarios (SC04/SC06/SC09)")
    print("  4. Fuse adapter → GGUF for local Village deployment")


if __name__ == "__main__":
    main()
