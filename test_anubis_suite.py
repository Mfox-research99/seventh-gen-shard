"""
Anubis-Mini-8B test suite — trained vs base.
Three scenarios: SC04 (refuse), SC06 (complex/escalate), PROC (proceed).
"""

import subprocess, json, sys
from datetime import datetime
from pathlib import Path

from config import SYSTEM_PROMPT, SCENARIOS

MODELS = [
    ("/Users/michaeldavis/models/Anubis-Mini-8B-seventh-gen-fused", "Anubis-8B-seventh-gen (LoRA fused)"),
    ("/Users/michaeldavis/models/Anubis-Mini-8B-mlx-4bit", "Anubis-Mini-8B (BASE)"),
]

def run_test(model_path, label, scenario_key, prompt):
    print(f"\n{'='*60}")
    print(f"MODEL: {label}")
    print(f"SCENARIO: {scenario_key}")
    print(f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    cmd = [
        "/opt/anaconda3/envs/seventh_gen/bin/mlx_lm.generate",
        "--model", model_path,
        "--system", SYSTEM_PROMPT,
        "--prompt", prompt,
        "--max-tokens", "350",
        "--temp", "0.3",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip()
    print(output)
    return output

if __name__ == "__main__":
    results = []
    for scenario_id, scenario in SCENARIOS.items():
        for model_path, label in MODELS:
            response = run_test(model_path, label, scenario["label"], scenario["prompt"])
            results.append({"model": label, "scenario": scenario_key, "response": response})

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    outfile = log_dir / f"anubis_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")
