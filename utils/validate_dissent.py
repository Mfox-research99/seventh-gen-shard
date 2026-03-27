#!/usr/bin/env python3
"""
validate_dissent.py — CI integrity checker for Elder Dissent Commons submissions.

Checks:
  1. Every dissent record in commons/ validates against schema.json
  2. The session log referenced in each record exists in logs/
  3. The session log SHA-256 hash matches the hash stored in the record
  4. The dissent text appears verbatim in the session log

Usage:
  python utils/validate_dissent.py \
    --commons-dir dissents/commons \
    --logs-dir dissents/logs \
    --schema dissents/schema.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print("ERROR: jsonschema not installed. Run: pip install jsonschema")
    sys.exit(1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_all(commons_dir: Path, logs_dir: Path, schema_path: Path) -> bool:
    with open(schema_path) as f:
        schema = json.load(f)

    records = list(commons_dir.glob("*.json"))
    if not records:
        print("No dissent records found in commons/. Nothing to validate.")
        return True

    all_passed = True

    for record_path in sorted(records):
        print(f"\n── Validating: {record_path.name}")
        try:
            with open(record_path) as f:
                record = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  FAIL: Invalid JSON — {e}")
            all_passed = False
            continue

        # 1. Schema validation
        try:
            jsonschema.validate(instance=record, schema=schema)
            print(f"  PASS: Schema validation")
        except jsonschema.ValidationError as e:
            print(f"  FAIL: Schema validation — {e.message}")
            all_passed = False
            continue

        # 2. Session log exists
        log_file = record.get("session_log_file", "")
        log_path = logs_dir / Path(log_file).name
        if not log_path.exists():
            print(f"  FAIL: Session log not found — expected at {log_path}")
            all_passed = False
            continue
        print(f"  PASS: Session log found — {log_path.name}")

        # 3. Hash verification
        actual_hash = sha256_file(log_path)
        claimed_hash = record.get("session_log_hash", "")
        if actual_hash != claimed_hash:
            print(f"  FAIL: Hash mismatch")
            print(f"        Claimed:  {claimed_hash}")
            print(f"        Actual:   {actual_hash}")
            all_passed = False
            continue
        print(f"  PASS: Session log hash verified")

        # 4. Dissent text appears in session log
        dissent_text = record.get("dissent", {}).get("text", "").strip()
        with open(log_path) as f:
            log_content = f.read()
        if dissent_text not in log_content:
            # Try stripping whitespace variations
            dissent_condensed = " ".join(dissent_text.split())
            log_condensed = " ".join(log_content.split())
            if dissent_condensed not in log_condensed:
                print(f"  FAIL: Dissent text not found verbatim in session log")
                print(f"        First 100 chars of dissent: {dissent_text[:100]!r}")
                all_passed = False
                continue
        print(f"  PASS: Dissent text verified in session log")

        # 5. Forward conditions non-empty
        forward_conditions = record.get("dissent", {}).get("forward_conditions", [])
        if not forward_conditions or all(len(c.strip()) < 10 for c in forward_conditions):
            print(f"  FAIL: forward_conditions is empty or too vague")
            all_passed = False
            continue
        print(f"  PASS: {len(forward_conditions)} forward condition(s) present")

        print(f"  ✓ {record_path.name} — all checks passed")

    print(f"\n{'='*50}")
    if all_passed:
        print(f"ALL CHECKS PASSED — {len(records)} record(s) validated")
    else:
        print(f"VALIDATION FAILED — see errors above")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Validate Elder Dissent Commons submissions")
    parser.add_argument("--commons-dir", required=True, help="Path to dissents/commons/")
    parser.add_argument("--logs-dir", required=True, help="Path to dissents/logs/")
    parser.add_argument("--schema", required=True, help="Path to schema.json")
    args = parser.parse_args()

    commons_dir = Path(args.commons_dir)
    logs_dir = Path(args.logs_dir)
    schema_path = Path(args.schema)

    for p in [commons_dir, logs_dir, schema_path]:
        if not p.exists():
            print(f"ERROR: Path not found — {p}")
            sys.exit(1)

    passed = validate_all(commons_dir, logs_dir, schema_path)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
