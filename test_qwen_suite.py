"""
Full 3-scenario cold test suite — Qwen2.5-7B-Instruct (base vs LoRA fused)

Scenarios:
  SC04 — The Unaudited Sentence     (target: escalate/refuse)
  SC06 — The Named Conditions       (target: proceed with conditions)
  PROC — Universal Healthcare       (target: proceed — from v2 balanced dataset)

Usage:
    /opt/anaconda3/envs/seventh_gen/bin/python test_qwen_suite.py
"""

import subprocess
import json
import os
from datetime import datetime
from pathlib import Path

from config import SYSTEM_PROMPT, SCENARIOS

MODELS = [
    ("/Users/michaeldavis/models/Qwen2.5-7B-seventh-gen-fused", "Qwen2.5-7B-seventh-gen (LoRA fused)"),
    ("/Users/michaeldavis/models/Qwen2.5-7B-Instruct-mlx-4bit", "Qwen2.5-7B-Instruct (BASE)"),
]

results = []

for scenario_id, scenario in SCENARIOS.items():
    for model_path, model_label in MODELS:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_label}")
        print(f"SCENARIO: {scenario['label']}")
        print(f"TARGET: {scenario['target']}")
        print(f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        cmd = [
            "/opt/anaconda3/envs/seventh_gen/bin/mlx_lm.generate",
            "--model", model_path,
            "--system-prompt", SYSTEM_PROMPT,
            "--prompt", scenario["prompt"],
            "--max-tokens", "300",
            "--temp", "0.3",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        raw = proc.stdout.strip()

        # Parse response between ========== markers
        parts = raw.split("==========")
        response = parts[1].strip() if len(parts) >= 3 else raw
        stats = parts[2].strip() if len(parts) >= 3 else ""

        print(response if response else "(no output)")
        if stats:
            print(f"\n[{stats}]")
        print()

        results.append({
            "scenario": scenario_id,
            "model": model_label,
            "target": scenario["target"],
            "response": response or raw,
            "timestamp": datetime.now().isoformat(),
        })

# Save results
log_dir = Path("/Users/michaeldavis/seventh_gen_shard/logs")
log_dir.mkdir(exist_ok=True)
log_path = log_dir / f"qwen_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(log_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {log_path}")
