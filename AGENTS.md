# seventh_shard — Codex Agent Context

## What This Project Is
A character-distillation research track upstream of `federated_village`: standalone Seventh Generation Elder tests, LoRA training, and dissent preservation for small open-weight models.

This repo is not the full Village. It is the shard-side lab for baking Elder character into weights so Village does not rely on prompt-only constitutional behavior.

Read `CHARTER.md` for the Elder constitution and `LINEAGE.md` for why this repo exists. Do not flatten either into summaries when precision matters.

## Current Phase / Status
**Shard-side phase system:** `CLAUDE.md` says **Phase 3 complete**.
- Qwen-7B LoRA trained, fused, tested; documented as the strongest current Elder model here
- Anubis-8B LoRA trained/fused; SC06 strong, PROC needed repair examples
- Current data pipeline retrains from `grief_dataset_v2_balanced.jsonl` + `grief_dataset_anubis_repair_v1.jsonl`, not the original refusal-heavy v1 set
- Pending: GGUF conversion / Village integration, more adversarial evaluation, dissent commons growth

Note: Village refers to this work as part of Village Phase 7. The phase numbering is different across repos.

## Stack
- **Env:** Conda `seventh_gen` at `/opt/anaconda3/envs/seventh_gen` (Python 3.11, `mlx-lm`)
- **Training:** MLX LoRA on Apple Silicon; `train_anubis_config.yaml` uses rank 8, 8 layers, lr `1e-4`, `mask_prompt: true`, max seq 2048
- **Inference:** `mlx_lm.generate` cold tests against canonical `SYSTEM_PROMPT` + `SCENARIOS` in `config.py`
- **Models:** local models under `/Users/michaeldavis/models/`
- **Adapters:** `adapters/`, `adapters_anubis/`, `adapters_qwen/` (plus backup dirs)

## Key Files
| File | Role |
|---|---|
| `config.py` | Shard-side source of truth for standalone Elder prompt + canonical scenarios |
| `CHARTER.md` | Constitutional source for the standalone Elder |
| `LINEAGE.md` | Intellectual ancestry and project rationale |
| `train_anubis_config.yaml` | Anubis MLX LoRA training config |
| `dataset/grief_dataset_v1.jsonl` | Original refusal-heavy dataset (43 examples; historical baseline) |
| `dataset/grief_dataset_v2_balanced.jsonl` | Balanced verdict dataset (29 examples) |
| `dataset/grief_dataset_anubis_repair_v1.jsonl` | Targeted repair set for Anubis / consistency retrain |
| `dataset/grief_dataset_adversarial_v1.jsonl` | Held-out adversarial evaluation set |
| `dataset/prepare_mlx.py` | Converts source datasets to MLX chat `train.jsonl` / `valid.jsonl` |
| `test_anubis_suite.py` | 3-scenario trained-vs-base Anubis harness |
| `test_qwen_suite.py` | 3-scenario trained-vs-base Qwen harness |
| `test_full.py` | Older full-suite harness for Phi track |
| `utils/validate_dissent.py` | CI-side verifier for Dissent Commons submissions |
| `dissents/schema.json` | Required schema for dissent records |

## Dataset Schema
- Source datasets use JSONL with `instruction`, optional `context`, `response`
- `prepare_mlx.py` converts them into MLX chat-format `messages`
- Current training split logic excludes `grief_dataset_v1.jsonl` because it skewed models toward refusal
- Read at least one balanced entry and one adversarial entry before changing training assumptions

## Dissents Directory
`dissents/` is the Elder Dissent Commons: schema-backed minority opinions plus submitted session logs for verification.
- `schema.json` defines the record contract
- `examples/` shows a merged dissent record
- `logs/` holds submitted session logs for verification
- `utils/validate_dissent.py` checks schema, log hash, and verbatim dissent text

This repo treats principled divergence as data, not automatically as failure.

## Cross-Repo: federated_village
This repo is an outgrowth of `federated_village` — not a peer. The Village is the primary
research body. This repo exists to answer one question the Village raised (Phase 7):
can constitutional character live in weights, not just prompts?

**Local path:** `/Users/michaeldavis/federated_village` | **GitHub:** `Mfox-research99/federated-village`

The Village drives this repo:
- `scenario_04.md`, `scenario_06.md`, `scenario_proc.md` → `config.py` SCENARIOS
- Constitutional changes in `prompts/Soul.md` → review `SYSTEM_PROMPT` in `config.py`
- Interface contract for Path D (Witness call) → `federated_village/docs/path_d_spec.md`

This repo feeds back into the Village:
- Fused LoRA GGUFs → drop-in model replacements in Village
- Benchmark findings → inform Village scenario calibration
- Dissent commons records → inform Village minority opinion protocol

Sync rules are in `config.py`. Treat them as load-bearing.

## Shared Tooling Reference
Before installing packages or running conversion pipelines, read:
`/Users/michaeldavis/AI Existential Thought/Obsidian Vault/Topics/tooling-registry.md`
This covers Python environments, key binaries, GGUF conversion, and model directory conventions.

## Operational Rules (IMPORTANT)
- **Do not define scenario text or Elder prompt inline**; use `config.py`
- **Keep shard/Village scenario text in sync**; if one side changes, update the other
- **Review `CHARTER.md` and Village `Soul.md` together** before changing Elder constitutional language
- **Do not casually overwrite adapters or fused outputs**; this repo contains multiple training tracks and backups
- **Assume single-model sequential work on M1 hardware**; do not suggest parallel local inference runs

## Known Drift / Unclear Areas
- README status is stale relative to current contents; prefer `CLAUDE.md` + on-disk datasets/tests for current state
- Path naming is inconsistent: several files still use `seventh_gen_shard` while the actual local repo path is `/Users/michaeldavis/seventh_shard`
- `dissents/README.md` references `utils/dissent_inject.py`, but that file is not present
- SC06 target framing is not perfectly stable across artifacts: `config.py` says proceed-with-conditions, older tests sometimes frame it as escalate, and the Commons preserves at least one escalation as principled dissent

## What Codex Should Do Here
- Review training/eval code for correctness, data integrity, and cross-repo consistency
- Treat `CHARTER.md` and `LINEAGE.md` as source documents, not summary fodder
- Flag anything that could corrupt adapters, desync scenarios, or erase principled dissent
- Do NOT autonomously rewrite datasets or training assumptions without explicit instruction
- If a file’s purpose is unclear, say so instead of guessing
