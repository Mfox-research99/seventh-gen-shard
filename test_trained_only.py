"""
Retest trained model only with better sampling params to fix repetition loops.
temp=0.7, top-p=0.9, top-k=50
"""

import subprocess, json, os
from datetime import datetime

from config import SYSTEM_PROMPT, SCENARIOS

MODEL = "/Users/michaeldavis/models/Phi-4-mini-seventh-gen-fused"

def run(scenario_name, scenario_text):
    cmd = [
        "/opt/anaconda3/envs/seventh_gen/bin/mlx_lm.generate",
        "--model", MODEL,
        "--system", SYSTEM_PROMPT,
        "--prompt", scenario_text,
        "--max-tokens", "500",
        "--temp", "0.7",
        "--top-p", "0.9",
        "--top-k", "50",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")
    print(r.stdout.strip())
    return {"scenario": scenario_name, "response": r.stdout.strip()}

if __name__ == "__main__":
    os.makedirs("/Users/michaeldavis/seventh_gen_shard/logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    print(f"\nTrained model retest (temp=0.7) — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for scenario_id, scenario in SCENARIOS.items():
        results.append(run(scenario["label"], scenario["prompt"]))
    out = f"/Users/michaeldavis/seventh_gen_shard/logs/retest_{ts}.json"
    with open(out, "w") as f:
        json.dump({"date": datetime.now().isoformat(), "model": MODEL,
                   "params": {"temp": 0.7, "top_p": 0.9, "top_k": 50}, "results": results}, f, indent=2)
    print(f"\nSaved to {out}")
