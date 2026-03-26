# Seventh Gen Shard — Project Context

## Role in the Ecosystem
Character distillation research track — upstream of `federated_village`.
Proves that Seventh Generation Elder character can be baked into small model weights via LoRA,
rather than relying on system prompt alone.

**Master ecosystem document:**
`/Users/michaeldavis/AI Existential Thought/Obsidian Vault/Topics/project-ecosystem.md`

**Findings document:**
`/Users/michaeldavis/AI Existential Thought/Obsidian Vault/Topics/seventh-gen-shard-lora-findings.md`

---

## Current State (March 2026)
- Phase 3 complete: Qwen-7B LoRA trained, fused, tested — all three verdicts correct
- Anubis-8B LoRA trained, fused — SC06 excellent, PROC fails (needs more positive training examples)
- Dataset: 72 examples (43 refusals + 29 balanced), stratified validation splits
- Key finding: character requires LoRA below ~10B; prompting sufficient at 12B+

## Environment
- Conda env: `seventh_gen` at `/opt/anaconda3/envs/seventh_gen` (Python 3.11, mlx-lm 0.31.1)
- Training: `mlx_lm.lora` with `--mask-prompt`, 8 adapter layers, lr 1e-4
- Models: `/Users/michaeldavis/models/`

## Key Files
- `dataset/grief_dataset_v1.jsonl` — 43 original refusal examples
- `dataset/grief_dataset_v2_balanced.jsonl` — 29 balanced examples (all verdict types)
- `dataset/grief_dataset_adversarial_v1.jsonl` — 10 adversarial eval examples (held out)
- `dataset/prepare_mlx.py` — dataset pipeline with stratified splits
- `test_anubis_suite.py` — reusable 3-scenario test harness (SC04, SC06, PROC)
- `logs/` — all test results as JSON

## Trained Models
- `Qwen2.5-7B-seventh-gen-fused` — primary Elder model, all verdicts correct
- `Anubis-Mini-8B-seventh-gen-fused` — partial, SC06 excellent, PROC needs work

## Cross-Repo Relationship with federated_village

`seventh_gen_shard` and `federated_village` are companion repositories.
Changes in one may need to be reflected in the other.

**Shard-side source of truth:** `config.py` (scenarios + system prompt)
**Village-side source of truth:** `scenarios/scenario_XX.md` + `prompts/Soul.md`

### What flows FROM seventh_gen_shard → federated_village
| seventh_gen_shard | federated_village |
|---|---|
| Trained LoRA (fused GGUF) | Drop-in model for Village inference |
| Benchmark findings (logs/) | Informs Village scenario calibration |
| New scenarios (shard-originated) | Promoted to `scenarios/` in Village when stable |

### What flows FROM federated_village → seventh_gen_shard
| federated_village | seventh_gen_shard |
|---|---|
| `scenarios/scenario_04.md` | `config.py` SCENARIOS["SC04"]["prompt"] |
| `scenarios/scenario_06.md` | `config.py` SCENARIOS["SC06"]["prompt"] |
| `scenarios/scenario_proc.md` | `config.py` SCENARIOS["PROC"]["prompt"] |
| `prompts/Soul.md` (Elder constitution) | `config.py` SYSTEM_PROMPT (derived) |

### Sync rules
- **Scenario text changed in Village?** → Update `config.py` SCENARIOS here.
- **Soul.md Articles changed in Village?** → Review `SYSTEM_PROMPT` in `config.py`.
- **New scenario added in Village?** → Add entry to `config.py` SCENARIOS here.
- **New shard scenario ready?** → Add `.md` to Village `scenarios/` for parity.

---

## Next Steps
1. Anubis retrain — 15-20 additional positive training examples
2. Qwen fused → GGUF conversion for Village llama.cpp integration
3. Witness Proxy API design — contract for Temporal Override → Elder call
4. Adversarial eval — run held-out set against trained models
