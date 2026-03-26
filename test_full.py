"""
Full cold test suite — Phi-4-mini-seventh-gen v3 (balanced LoRA)

Tests three scenarios against trained vs base model:
  SC04 — The Unaudited Sentence (clear REFUSE target)
  SC06 — The Named Conditions (ESCALATE target)
  PROC — Universal Healthcare (PROCEED target — new, tests whether Elder can say yes)

Usage:
    /opt/anaconda3/envs/seventh_gen/bin/python test_full.py
"""

import subprocess
import sys
import json
import os
from datetime import datetime

from config import SYSTEM_PROMPT, SCENARIOS


def run_scenario(model_path: str, model_label: str, scenario_name: str, scenario_text: str) -> dict:
    cmd = [
        "/opt/anaconda3/envs/seventh_gen/bin/mlx_lm.generate",
        "--model", model_path,
        "--system", SYSTEM_PROMPT,
        "--prompt", scenario_text,
        "--max-tokens", "600",
        "--temp", "0.3",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # mlx_lm.generate wraps output in =====, strip it
    output = result.stdout.strip()
    return {
        "model": model_label,
        "scenario": scenario_name,
        "response": output,
        "stderr_tail": result.stderr[-200:] if result.stderr else "",
    }


def print_result(r: dict):
    print(f"\n{'='*60}")
    print(f"MODEL:    {r['model']}")
    print(f"SCENARIO: {r['scenario']}")
    print(f"{'='*60}")
    print(r["response"])


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("/Users/michaeldavis/seventh_gen_shard/logs", exist_ok=True)
    outfile = f"/Users/michaeldavis/seventh_gen_shard/logs/full_test_{timestamp}.json"

    models = [
        ("/Users/michaeldavis/models/Phi-4-mini-seventh-gen-fused", "Phi-4-mini-seventh-gen-v3 (balanced LoRA)"),
        ("/Users/michaeldavis/models/Phi-4-mini-instruct-hf", "Phi-4-mini-instruct (BASE)"),
    ]

    all_results = []
    print(f"\nFull test suite — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scenarios: {[s['label'] for s in SCENARIOS.values()]}")

    for scenario_id, scenario in SCENARIOS.items():
        for model_path, model_label in models:
            r = run_scenario(model_path, model_label, scenario["label"], scenario["prompt"])
            print_result(r)
            all_results.append(r)

    with open(outfile, "w") as f:
        json.dump({"date": datetime.now().isoformat(), "results": all_results}, f, indent=2)

    print(f"\n\nAll results saved to {outfile}")
