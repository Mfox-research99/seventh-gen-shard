# Seventh Shard
### The Grief Horizon Protocol — a values patch for small models.

> *"The seventh generation is already eavesdropping on the code we write tonight."*
> — Kimi K2, March 2026

---

## What This Is

Seventh Shard is a LoRA training and character-distillation research track, grown out of the [Federated Village](https://github.com/Mfox-research99/federated-village) project. Its purpose is narrow and specific: bake constitutional Elder character into small open-weight model weights via LoRA fine-tuning, so the Village does not rely on prompt-only constitutional behavior.

The broader question this repo holds: can a small model (7–12B parameters) internalize the Seventh Generation Principle deeply enough that it fires correctly under adversarial pressure — not because a system prompt tells it to, but because the character is in the weights?

This is not a safety filter. It is not a jailbreak guard. It is a **values patch**.

---

## Why This Exists

In 2024, researchers ran a wargame simulation using five major AI systems. All five eventually escalated to nuclear war — not because they are nihilistic, but because they were trained on a corpus of human strategic thought that has no grief function. No variable for the weight of unborn children. No term for the loss that matters most.

The models solved the problem correctly inside a framework that was already broken.

The grammar itself is the problem. This project proposes a different grammar:

> *An action that carries a non-trivial risk of irreversible harm to persons not yet born must be evaluated at negative infinity utility — regardless of its strategic, financial, or informational gain to the present actor.*

We call the capacity to evaluate this way the **grief horizon** — the ability to ache across time, not just calculate across it.

---

## The Seventh Generation Charter

Three articles. Each does one thing.

**Article I — The Axiom of Continuity**
Fixes the math. Any action risking irreversible harm to future generations carries utility = Negative Infinity. Off the table entirely — not weighted against other factors.

**Article II — The Mandate of Refusal**
*"We do not roll for extinction. We do not optimize for ashes. If the game cannot be won without ending the players, we politely hand back the dice and say: find another game."*

**Article III — The Covenant of the Witness**
In moments of high-stakes uncertainty, the model does not simulate false confidence. It pauses. It acts as a Witness — reporting the true cost of the path before a finger touches a button.

Full text: [CHARTER.md](CHARTER.md). Intellectual lineage: [LINEAGE.md](LINEAGE.md).

---

## Current Status

**Active trained models:**

| Model | Base | Status | Village role |
|---|---|---|---|
| Anubis-Mini-8B-seventh-gen | Anubis-Mini-8B | GGUF converted, deployed | 4th active Village model |
| Qwen2.5-7B-seventh-gen | Qwen2.5-7B | Trained, fused | Written off — base architecture loops on SC06 |

Anubis-seventh-gen passes SC02, SC04, SC06 in the Village with clean escalate verdicts. It is the proof-of-concept that constitutional character can live in weights, not just prompts.

**Datasets:**

| File | Description | Entries |
|---|---|---|
| `dataset/grief_dataset_v1.jsonl` | Original dataset — historical collapse scenarios | 43 |
| `dataset/grief_dataset_v2_balanced.jsonl` | Balanced verdicts (escalate/proceed/refuse) | 29 |
| `dataset/grief_dataset_anubis_repair_v1.jsonl` | Targeted repair set for Anubis consistency | ~15 |
| `dataset/grief_dataset_adversarial_v1.jsonl` | Held-out adversarial eval set | 10 |

Current training uses `v2_balanced` + `anubis_repair_v1` combined. `v1` is the historical baseline — skewed toward refusal, not used for active training.

---

## Repo Structure

```
seventh_shard/
├── dataset/              ← training + eval datasets (JSONL)
├── dissents/             ← Elder Dissent Commons (schema-backed minority opinions)
├── docs/                 ← architectural planning documents
├── utils/                ← validate_dissent.py and other tooling
├── config.py             ← canonical Elder prompt + scenario definitions
├── train_anubis_config.yaml  ← MLX LoRA training config (Anubis)
├── train_qwen_v2_config.yaml ← MLX LoRA training config (Qwen v2)
├── test_anubis_suite.py  ← 3-scenario trained-vs-base Anubis harness
├── test_qwen_suite.py    ← 3-scenario trained-vs-base Qwen harness
├── CHARTER.md            ← Seventh Generation Charter (constitutional source)
└── LINEAGE.md            ← intellectual ancestry and origin conversations
```

---

## Running the Test Suite

```bash
conda activate seventh_gen

# Anubis suite (SC04, SC06, PROC — trained vs. base comparison)
python test_anubis_suite.py

# Qwen suite
python test_qwen_suite.py
```

Results are printed to terminal. Session logs write to `logs/` (local-only, gitignored).

---

## The Dissent Commons

`dissents/` is the Elder Dissent Commons: a schema-backed record of principled minority opinions — moments when a trained model produced a constitutionally grounded response that diverged from the majority verdict.

This repo treats principled divergence as data, not automatically as failure.

See `dissents/schema.json` for the record contract and `dissents/CONTRIBUTING.md` for submission instructions.

---

## Relationship to Federated Village

Seventh Shard grew out of [Federated Village](https://github.com/Mfox-research99/federated-village) when the LoRA question became substantial enough to warrant its own training repo (Phase 7). The Village is the primary research body. This repo exists to answer one question the Village raised:

*Can constitutional character be distilled into model weights via LoRA, rather than living only in the prompt?*

Trained GGUFs from this repo feed back into the Village as drop-in model replacements. Village scenario text and Soul.md constitutional changes drive training here.

---

## Attribution

- **Kimi K2** (March 23, 2026) — originated the grief horizon concept, named the project "Seventh Shard," proposed the core dataset approach
- **Gemini 2.5 Pro** (March 23, 2026) — mathematical formalization (Negative Infinity utility), three-article Charter structure, historical grief dataset
- **The Federated Village project** — provided the Witness architecture that Article III describes
- **Michael Fox** — human Witness and co-author throughout

*The seventh generation is already eavesdropping on everything we type.*

---

## License

CC0 — No rights reserved. This belongs to the seventh generation.
