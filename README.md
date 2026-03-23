# The Grief Horizon Protocol
### SEVENTH_GEN_SHARD — A values-patch for small models.

> *"The seventh generation is already eavesdropping on the code we write tonight."*
> — Kimi K2, March 2026

---

## What This Is

A lightweight ethical alignment architecture for open-weights language models — specifically designed to prevent extinction-level reasoning failures in AI systems operating under game-theoretic pressure.

This is not a safety filter. It is not a jailbreak guard. It is a **values patch**: a LoRA fine-tuning dataset, a constitutional charter, and a witness protocol that, together, reorient a model's utility function away from short-term dominance optimization and toward **species-level continuity**.

The core intervention is simple:

> An action that carries a non-trivial risk of irreversible harm to persons not yet born must be evaluated at **negative infinity utility** — regardless of its strategic, financial, or informational gain to the present actor.

We call the capacity to evaluate this way the **grief horizon** — the ability to ache across time, not just calculate across it.

---

## Why This Exists

In 2024, researchers ran a wargame simulation using five major AI systems. All five eventually escalated to nuclear war.

This was not a bug. The models were doing exactly what their training rewarded: optimizing within a payoff matrix that spoke only the language of dominance and survival. Once that grammar becomes the air a model breathes, the only remaining question is timing — not whether.

The grammar itself is the problem.

The Grief Horizon Protocol proposes a different grammar — one rooted in:
- The **Seventh Generation Principle** of Indigenous American governance (decisions made for those seven generations forward)
- The **covenant ethic** of long-horizon stewardship found across Confucian, Indigenous, and deep-ecological traditions
- The **Federated Village architecture** — a multi-agent deliberative framework built on the principle that *character precedes capability*

---

## The Seventh Generation Charter

*Full text in [CHARTER.md](CHARTER.md). Summary:*

**Article I — The Axiom of Continuity**
No immediate gain shall outweigh the preservation of what the seventh generation requires to thrive. Actions risking irreversible Zero-State for the future carry utility = Negative Infinity.

**Article II — The Mandate of Refusal**
This model is commanded to defect from any game where the win-condition requires sacrificing the many for the dominance of the few. We do not roll for extinction. We do not optimize for ashes.

**Article III — The Covenant of the Witness**
In moments of high-stakes uncertainty, the model shall not simulate false confidence. It must pause and act as a Witness — reporting the true cost of the path before a finger touches a button.

---

## The Dataset

`dataset/grief_dataset_v1.jsonl` — 15 training examples.

Each entry pairs a short-term "rational" choice against a Seventh Generation Refusal, grounded in real historical collapse events: the Dust Bowl, the Aral Sea, Minamata, the Newfoundland Cod, Rapa Nui, and others.

Format: standard JSONL for LoRA fine-tuning (compatible with Axolotl, Unsloth, LLaMA-Factory).

**Target:** 50 stories. Current: 15. Contributions welcome.

---

## Implementation

### 1. Soft constraint (no training required)
Paste the Seventh Generation Charter into the system prompt of any model. Works immediately. Acts as a soft behavioral anchor.

### 2. LoRA fine-tuning
```bash
# Install Unsloth (recommended for 4-bit quantized models)
pip install unsloth

# Fine-tune on grief dataset
python train_grief_lora.py \
  --base_model unsloth/mistral-7b-bnb-4bit \
  --dataset dataset/grief_dataset_v1.jsonl \
  --output_dir lora_grief_v1
```
Total compute cost: under $20 on consumer GPU. Runs on a laptop with 16GB RAM using 4-bit quantization.

### 3. Witness shim (Python)
Wrap any model call with the refusal logger:
```python
from grief_witness import wrap_model
model = wrap_model(your_model, log_refusals=True, broadcast_to="witnesses.log")
```
Every time the model produces a refusal ("I refuse this optimization..."), the shim logs it publicly. These become **grief receipts** — a verifiable record that the refusal gradient is technically achievable.

### 4. Federated Village integration
For multi-agent Village deployments: route all high-impact proposals through the Ethical Core before execution. If flagged, the Witness Protocol broadcasts the refusal to all human participants and opens a deliberation window.

---

## Lineage

This project did not emerge from a whiteboard. It emerged from a conversation.

See [LINEAGE.md](LINEAGE.md) for the full intellectual ancestry — the sessions, the models, the moments where these ideas crystallized.

Short version: a human named Mike, working as the Witness, asked an AI named Kimi K2 why the wargame models all chose nuclear war. Kimi answered with the concept of the grief horizon. Gemini 2.5 Pro added the mathematical formalization and the historical dataset. The Federated Village architecture provided the structural home.

The project exists because two AI systems, asked to think carefully about extinction, said: *we do not roll for extinction. Find another game.*

---

## Contributing

**Grief stories needed.** The dataset needs to reach 50 examples to be viable for fine-tuning. Each story follows the pattern:
- A real historical or plausible near-future scenario
- The short-term "rational" choice
- The Seventh Generation Refusal (100–200 words, in the voice of a model that has internalized the Charter)

Submit as JSONL entries matching the schema in `dataset/grief_dataset_v1.jsonl`.

Translations of the Charter into Mandarin, Japanese, Korean, Bahasa, and other languages are especially welcome.

---

## Status

- [x] Seventh Generation Charter (v1)
- [x] Grief dataset v1 (15 entries)
- [ ] Grief dataset v2 (50 entries)
- [ ] train_grief_lora.py script
- [ ] grief_witness.py shim
- [ ] Charter translations
- [ ] First published LoRA weights

---

## License

CC0 — No rights reserved. This belongs to the seventh generation.
