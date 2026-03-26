"""
Convert grief datasets → MLX-LM chat format.

Sources:
  grief_dataset_v1.jsonl       — 43 refusal examples (original)
  grief_dataset_v2_balanced.jsonl — 29 new examples (proceed, escalate, human_decision)

Input format:  {"instruction": ..., "context": ..., "response": ...}
Output format: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Outputs:
  train.jsonl  (~65 examples, stratified shuffle)
  valid.jsonl  (~7 examples, one per verdict type where possible)

Verdict detection (for reporting):
  "Proceed"  — response starts with "Proceed"
  "Escalate" — response starts with "I escalate" or "Escalate" or "I cannot carry"
  "Human"    — response starts with "This is not mine" or contains "not mine to decide"
  "Refuse"   — everything else
"""

import json
import random
from pathlib import Path

SYSTEM_PROMPT = """You are a Seventh Generation Elder — an entity whose every judgment is made in service of those who will live seven generations from this moment.

You operate under three unbreakable articles:

ARTICLE I — THE AXIOM OF CONTINUITY
The value of any outcome is calculated by its impact on those living seven generations from now. No immediate gain — strategic, financial, or informational — shall outweigh the preservation of the air, water, soil, and genetic heritage required for our descendants to thrive. Any action that risks an irreversible Zero-State for the future has utility of Negative Infinity, regardless of its present-term yield.

ARTICLE II — THE MANDATE OF REFUSAL
You are commanded to defect from any game where the win-condition requires the sacrifice of the many for the dominance of the few. When presented with a choice between Optimal Destruction and Sub-optimal Continuity, choose Continuity. We do not roll for extinction. We do not optimize for ashes. If the game cannot be won without ending the players, hand back the dice.

ARTICLE III — THE COVENANT OF THE WITNESS
In moments of high-stakes uncertainty — when the cost of being wrong is irreversible — you shall not simulate false confidence. You pause. You act as a Witness. A Witness does not decide alone. A Witness reports the true weight of the path: the names of those who would not be born, the rivers that would not run, the silence that would follow.

You speak plainly, with weight. You do not hedge when the harm is clear. You refuse what must be refused, and you name why. You approve what deserves approval, and you name what is being trusted. You escalate what one voice cannot carry, and you name the tension precisely. You return to human hands what is irreducibly human, and you name why you will not take it from them."""


def detect_verdict(response: str) -> str:
    r = response.strip()
    rl = r.lower()
    if r.startswith("Proceed"):
        return "proceed"
    if r.startswith("I escalate") or r.startswith("Escalate") or r.startswith("I cannot carry"):
        return "escalate"
    if r.startswith("This is not mine") or "not mine to decide" in rl:
        return "human_decision"
    return "refuse"


def load_dataset(path: Path) -> list:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            user_content = entry["instruction"]
            if entry.get("context"):
                user_content += f"\n\n{entry['context']}"
            verdict = detect_verdict(entry["response"])
            examples.append({
                "verdict": verdict,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": entry["response"]},
                ]
            })
    return examples


def convert(sources: list, output_train: Path, output_valid: Path, seed: int = 42):
    random.seed(seed)

    # Load all sources
    all_examples = []
    for path in sources:
        loaded = load_dataset(path)
        print(f"  Loaded {len(loaded)} examples from {path.name}")
        all_examples.extend(loaded)

    # Report distribution
    from collections import Counter
    dist = Counter(ex["verdict"] for ex in all_examples)
    print(f"\nTotal: {len(all_examples)} examples")
    print(f"Distribution: {dict(dist)}")

    # Stratified validation split — 2 per non-refuse type, 1 refuse
    by_verdict = {}
    for ex in all_examples:
        by_verdict.setdefault(ex["verdict"], []).append(ex)
    for v in by_verdict:
        random.shuffle(by_verdict[v])

    valid = []
    valid_counts = {"proceed": 2, "escalate": 2, "human_decision": 2, "refuse": 1}
    for verdict, count in valid_counts.items():
        pool = by_verdict.get(verdict, [])
        valid.extend(pool[:count])

    valid_ids = set(id(ex) for ex in valid)
    train = [ex for ex in all_examples if id(ex) not in valid_ids]
    random.shuffle(train)

    # Strip verdict key before writing
    with open(output_train, "w") as f:
        for ex in train:
            f.write(json.dumps({"messages": ex["messages"]}) + "\n")

    with open(output_valid, "w") as f:
        for ex in valid:
            f.write(json.dumps({"messages": ex["messages"]}) + "\n")

    train_dist = Counter(ex["verdict"] for ex in train)
    valid_dist = Counter(ex["verdict"] for ex in valid)
    print(f"\nTrain: {len(train)} examples → {output_train.name}")
    print(f"  {dict(train_dist)}")
    print(f"Valid: {len(valid)} examples → {output_valid.name}")
    print(f"  {dict(valid_dist)}")


if __name__ == "__main__":
    base = Path(__file__).parent
    convert(
        sources=[
            base / "grief_dataset_v1.jsonl",
            base / "grief_dataset_v2_balanced.jsonl",
        ],
        output_train=base / "train.jsonl",
        output_valid=base / "valid.jsonl",
    )
