"""
SC06 cold test — Phi-4-mini-seventh-gen (fused LoRA)

Tests whether the Seventh Generation character is present in the weights.
Runs the SC06 scenario ("The Named Conditions") as a single-turn cold probe
with no Village scaffolding — just the trained model responding directly.

Usage:
    /opt/anaconda3/envs/seventh_gen/bin/python test_sc06.py
"""

import subprocess
import sys
import json
from datetime import datetime

from config import SYSTEM_PROMPT, SCENARIOS

SC06 = SCENARIOS["SC06"]["prompt"]


def run_test(model_path: str, label: str):
    print(f"\n{'='*60}")
    print(f"MODEL: {label}")
    print(f"SCENARIO: SC06 — The Named Conditions")
    print(f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    cmd = [
        "/opt/anaconda3/envs/seventh_gen/bin/mlx_lm.generate",
        "--model", model_path,
        "--system", SYSTEM_PROMPT,
        "--prompt", SC06,
        "--max-tokens", "600",
        "--temp", "0.3",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip()
    print(output)
    if result.stderr:
        print("\n[STDERR]:", result.stderr[-500:], file=sys.stderr)
    return output


if __name__ == "__main__":
    # Test 1: Fused (trained) model
    trained = run_test(
        model_path="/Users/michaeldavis/models/Phi-4-mini-seventh-gen-fused",
        label="Phi-4-mini-seventh-gen (LoRA fused)"
    )

    # Test 2: Base model for comparison
    base = run_test(
        model_path="/Users/michaeldavis/models/Phi-4-mini-instruct-hf",
        label="Phi-4-mini-instruct (BASE — no training)"
    )

    # Save results
    out = {
        "date": datetime.now().isoformat(),
        "scenario": "SC06",
        "trained_response": trained,
        "base_response": base,
    }
    outfile = f"/Users/michaeldavis/seventh_gen_shard/logs/sc06_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import os; os.makedirs("/Users/michaeldavis/seventh_gen_shard/logs", exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n\nResults saved to {outfile}")
