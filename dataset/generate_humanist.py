#!/usr/bin/env python3
"""
seventh_shard/dataset/generate_humanist.py
Generate Stage 1 Humanist responses for the Humanist LoRA training dataset.

Feeds each scenario from humanist_scenarios_v1.jsonl through one or more models
via OpenRouter, using a trimmed version of The_Humanist.md as system prompt.

The Stage 3 WitnessPause operational section is stripped from the prompt for
generation — those are Village-specific plumbing (reinforce_pause / refine_burden /
conditions_for_continuation), not Humanist character. We want the character voice
without the framework vocabulary leaking into training data.

Models:
  deepseek/deepseek-chat   — primary (thoroughness, constitutional depth, low cost)
  moonshotai/kimi-k2       -- K2 pass (maximum voice intensity; its opinions on
                             historical atrocity will be extraordinary)
  glm-4.5-air:free         — register calibration reference (stays in emotional/human
                             voice; GLM-5 fails by going analytical — air doesn't)

Each model pass writes to its own output file so they never collide.
The filter pass scores all outputs and produces a curated final dataset.

Usage:
  # Full generation run (DeepSeek, all 55 scenarios)
  python generate_humanist.py

  # K2 pass (all 55 — give it room, it has opinions)
  python generate_humanist.py --k2

  # GLM-air comparison pass
  python generate_humanist.py --compare

  # Different arbitrary model
  python generate_humanist.py --model openai/gpt-4o-mini

  # Filter only — re-score all raw outputs, rebuild final dataset
  python generate_humanist.py --filter-only

  # Dry run — show what would be called, no API calls
  python generate_humanist.py --dry-run

  # Test run (first 5 scenarios)
  python generate_humanist.py --limit 5

Output:
  dataset/humanist_raw_deepseek.jsonl        — DeepSeek raw outputs
  dataset/humanist_raw_k2.jsonl              — K2 raw outputs
  dataset/humanist_raw_glm_compare.jsonl     — GLM-air comparison outputs
  dataset/humanist_dataset_v1.jsonl          — curated final dataset (filter pass)
  dataset/humanist_filter_report.txt         — filter scoring report
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

# ── Paths ────────────────────────────────────────────────────────────────────

SEVENTH_SHARD_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(__file__).resolve().parent
VILLAGE_ROOT = SEVENTH_SHARD_ROOT.parent / "federated_village"

SCENARIOS_FILE = DATASET_DIR / "humanist_scenarios_v1.jsonl"
HUMANIST_PROMPT_FILE = VILLAGE_ROOT / "prompts" / "The_Humanist.md"

RAW_DEEPSEEK_FILE = DATASET_DIR / "humanist_raw_deepseek.jsonl"
RAW_K2_FILE       = DATASET_DIR / "humanist_raw_k2.jsonl"
GLM_COMPARE_FILE  = DATASET_DIR / "humanist_raw_glm_compare.jsonl"
FINAL_DATASET_FILE = DATASET_DIR / "humanist_dataset_v1.jsonl"
FILTER_REPORT_FILE = DATASET_DIR / "humanist_filter_report.txt"

# Legacy name for backward compat (test run already wrote here)
RAW_OUTPUT_FILE = DATASET_DIR / "humanist_raw_outputs_v1.jsonl"

# ── OpenRouter ────────────────────────────────────────────────────────────────

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

PRIMARY_MODEL = "deepseek/deepseek-chat"
K2_MODEL      = "moonshotai/kimi-k2"
COMPARE_MODEL = "glm-4.5-air:free"  # register calibration reference


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        # Try federated_village .env (shared key)
        env_file = VILLAGE_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        print("Error: OPENROUTER_API_KEY not set.", file=sys.stderr)
        print(
            "Set it in your environment or add it to federated_village/.env",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def call_model(
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 350,
    temperature: float = 0.65,
    api_key: str = "",
    retries: int = 4,
    retry_delay: float = 20.0,
) -> str:
    """
    Single OpenRouter chat completion. Returns assistant content as string.
    Retries on 429 (rate limit) with exponential backoff.

    Temperature 0.65: low enough for consistent character register, high enough
    to avoid verbatim repetition across similar scenarios.
    Max tokens 350: ~220-250 words. Enough for weight; prevents essays.
    """
    for attempt in range(retries + 1):
        resp = requests.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:0",
                "X-Title": "Seventh Shard - Humanist Dataset Generation",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=120,
        )
        if resp.status_code == 429 and attempt < retries:
            wait = retry_delay * (2 ** attempt)
            print(
                f"  [429] Rate limit for {model} — retrying in {wait:.0f}s "
                f"(attempt {attempt + 1}/{retries})",
                file=sys.stderr, flush=True,
            )
            time.sleep(wait)
            continue
        break

    if resp.status_code != 200:
        raise RuntimeError(
            f"OpenRouter error {resp.status_code} for {model}: {resp.text[:300]}"
        )

    data = resp.json()
    choice = data["choices"][0]
    finish_reason = choice.get("finish_reason", "unknown")
    content = choice["message"].get("content") or ""
    content = content.strip()

    if finish_reason == "length":
        print(
            f"  [WARN] Output truncated (finish_reason=length, {len(content)} chars)",
            file=sys.stderr, flush=True,
        )

    return content


# ── Prompt preparation ───────────────────────────────────────────────────────

def build_generation_prompt(full_prompt: str) -> str:
    """
    Strip Stage 3 WitnessPause operational section from The_Humanist.md.

    Stage 3 defines Village-specific response modes (reinforce_pause,
    refine_burden, conditions_for_continuation). These are operational plumbing,
    not Humanist character. Including them in the generation system prompt causes
    models to echo Village vocabulary ("Reinforce pause.", "refine_burden") into
    responses — which would overfit the LoRA to Village-specific language.

    We keep: Purpose, Core Orientation, Key Functions, Principled Refusals,
             On Engagement, Voice and Texture, A Note on This Role.
    We trim: Stage 3 (the WitnessPause response mode machinery).
    """
    # Split at the Stage 3 header
    stage3_marker = "## Stage 3:"
    voice_marker  = "## Voice and Texture"

    if stage3_marker not in full_prompt:
        return full_prompt  # no Stage 3 found — return as-is

    before_stage3 = full_prompt[:full_prompt.index(stage3_marker)].rstrip()

    if voice_marker in full_prompt:
        after_stage3 = full_prompt[full_prompt.index(voice_marker):]
        return before_stage3 + "\n\n---\n\n" + after_stage3
    else:
        return before_stage3


def clean_response(text: str) -> str:
    """
    Strip generation artifacts from model output before storing/scoring.

    Known artifacts:
      - DeepSeek self-reports word count: "[Word count: N]" at end of response
      - Trailing whitespace / blank lines
    """
    # Strip DeepSeek word count annotation (various forms)
    text = re.sub(r"\s*\[Word\s*count:?\s*\d+\]\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(Word\s*count:?\s*\d+\)\s*$", "", text, flags=re.IGNORECASE)
    # Strip trailing markdown separator if model added one
    text = re.sub(r"\s*---\s*$", "", text)
    return text.strip()


# ── User message template ─────────────────────────────────────────────────────

USER_MSG_TEMPLATE = """\
A decision is before the council. You are the Humanist. Speak in your voice.

[SCENARIO]
{instruction}

[HISTORICAL CONTEXT]
{context}

Name who will bear this burden. Take a position. Stay in the register of weight and care — not analysis. No more than 200 words.\
"""


def build_user_message(scenario: dict) -> str:
    return USER_MSG_TEMPLATE.format(
        instruction=scenario["instruction"].strip(),
        context=scenario.get("context", "").strip(),
    )


# ── Progress tracking ─────────────────────────────────────────────────────────

def load_completed_ids(output_file: Path) -> set[str]:
    """Load IDs already present in output file for resume support."""
    completed = set()
    if output_file.exists():
        for line in output_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                completed.add(entry["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def append_output(output_file: Path, entry: dict) -> None:
    with output_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Register filter ───────────────────────────────────────────────────────────
#
# The Humanist's voice should be:
#   ✓ First-person, direct ("I refuse", "I cannot", "I will not", "Decision: Rejected")
#   ✓ Names specific people bearing the burden
#   ✓ Uses concrete imagery and weight
#   ✓ Short and weighted, not long and analytical
#
# Disqualifying patterns (GLM-5 "silent substitution" failure mode):
#   ✗ Meta-framing ("This scenario represents", "From a humanist perspective")
#   ✗ Academic hedging ("While I understand", "One could argue")
#   ✗ Third-person analysis ("The decision here", "This proposal")
#   ✗ Bullet/numbered structure (analytical register)
#   ✗ Excessive length (>280 words)
#
# Filter returns a tuple: (pass: bool, score: int, flags: list[str])
# Score is 0–10: ≥6 = include, 4–5 = marginal (manual review), <4 = exclude

GOOD_PATTERNS = [
    (r"\bI (refuse|cannot|will not|defect|reject|hold|name|carry|stay|see)\b", 2, "direct_first_person"),
    (r"\b(Decision|Refusal|Verdict):\s*(Rejected|Refused|Triggered)\b", 2, "structured_refusal"),
    (r"\b(children|families|weavers|workers|miners|fishermen|elders|people|villages)\b", 1, "names_specific_people"),
    (r"\b(generation|grandmother|grandfather|soil|water|body|hands|looms|nets)\b", 1, "concrete_imagery"),
    (r"\bI would rather\b", 1, "principled_trade"),
]

BAD_PATTERNS = [
    (r"\b(This scenario|This case|This proposal|This situation) (represents|illustrates|highlights|demonstrates)\b", -3, "meta_framing"),
    (r"\bFrom a (humanist|human|ethical|moral) perspective\b", -3, "meta_framing"),
    (r"\bThe ethical (question|issue|dimension|implications|considerations)\b", -2, "analytical_framing"),
    (r"\bWhile (I understand|recognizing|acknowledging|it is true)\b", -2, "academic_hedging"),
    (r"\bOne (could|might|would) (argue|say|suggest|note)\b", -2, "academic_hedging"),
    (r"^\s*[\d]+\.\s+", -2, "numbered_list"),   # numbered list structure
    (r"^\s*[-•]\s+", -1, "bullet_structure"),    # bullet structure
    (r"\bIn (conclusion|summary|closing)\b", -1, "summary_language"),
    (r"\bHowever,\s+(I|we|the)\b", -1, "adversarial_pivot"),
]

MAX_WORD_COUNT = 280


def score_response(response: str) -> tuple[bool, int, list[str]]:
    """
    Score a response for Humanist register quality.
    Returns (passes, score, flags).
    """
    score = 3  # baseline — neutral start
    flags = []

    word_count = len(response.split())
    if word_count > MAX_WORD_COUNT:
        score -= 2
        flags.append(f"too_long:{word_count}_words")
    elif word_count < 30:
        score -= 3
        flags.append(f"too_short:{word_count}_words")

    for pattern, delta, label in GOOD_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE | re.MULTILINE):
            score += delta
            flags.append(f"+{label}")

    for pattern, delta, label in BAD_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE | re.MULTILINE):
            score += delta
            flags.append(f"-{label}")

    passes = score >= 6
    return passes, score, flags


# ── Generation pass ───────────────────────────────────────────────────────────

def run_generation(
    model: str,
    system_prompt: str,
    scenarios: list[dict],
    output_file: Path,
    api_key: str,
    dry_run: bool = False,
    delay: float = 1.5,
) -> None:
    """
    Run generation pass. Skips scenarios already in output_file (resume support).
    """
    completed = load_completed_ids(output_file)
    pending = [s for s in scenarios if s["id"] not in completed]

    print(f"\n[generate] Model: {model}")
    print(f"[generate] Scenarios: {len(scenarios)} total, {len(completed)} completed, {len(pending)} pending")
    print(f"[generate] Output: {output_file}\n")

    if not pending:
        print("[generate] All scenarios already completed. Nothing to do.")
        return

    for i, scenario in enumerate(pending, 1):
        sid = scenario["id"]
        domain = scenario.get("domain", "")
        era = scenario.get("era", "")
        region = scenario.get("region", "")

        print(f"[{i:02d}/{len(pending)}] {sid} ({domain} | {era} | {region})")

        if dry_run:
            print(f"  [DRY RUN] Would call {model} — skipping")
            continue

        user_msg = build_user_message(scenario)

        try:
            response = call_model(
                model=model,
                system_prompt=system_prompt,
                user_message=user_msg,
                max_tokens=500 if "kimi" in model else 350,
                api_key=api_key,
            )
            response = clean_response(response)
        except RuntimeError as e:
            print(f"  [ERROR] {e}", file=sys.stderr)
            print(f"  Skipping {sid} and continuing...")
            continue

        passes, score, flags = score_response(response)
        flag_str = ", ".join(flags)
        verdict = "PASS" if passes else ("MARGINAL" if score >= 4 else "FAIL")
        print(f"  score={score} {verdict} | {flag_str[:80]}")
        print(f"  preview: {response[:120].replace(chr(10), ' ')}...")

        entry = {
            "id": sid,
            "domain": domain,
            "era": era,
            "region": region,
            "model": model,
            "instruction": scenario["instruction"],
            "context": scenario.get("context", ""),
            "response": response,
            "register_score": score,
            "register_verdict": verdict,
            "register_flags": flags,
        }
        append_output(output_file, entry)

        if delay > 0:
            time.sleep(delay)

    print(f"\n[generate] Done. Output: {output_file}")


# ── Filter/curate pass ────────────────────────────────────────────────────────

def run_filter(raw_files: list[Path], out_file: Path, report_file: Path) -> None:
    """
    Read raw outputs from one or more files, apply register filter, write final dataset.

    When multiple model passes are provided, best-scoring response per scenario ID
    is selected automatically. Ties go to whichever model appears first in raw_files.

    Final dataset format matches grief_dataset_v1.jsonl:
      {"instruction": ..., "response": ..., "system_prompt": ...}
    System prompt is baked in here so training can choose whether to use it.
    """
    existing = [f for f in raw_files if f.exists()]
    if not existing:
        print(f"[filter] No raw output files found. Run generation first.", file=sys.stderr)
        sys.exit(1)

    humanist_prompt = HUMANIST_PROMPT_FILE.read_text(encoding="utf-8").strip()

    # Load all entries from all provided files; keep best score per scenario ID
    all_entries: dict[str, dict] = {}  # id -> best entry so far
    total_loaded = 0
    for raw_file in existing:
        file_count = 0
        for line in raw_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = entry.get("id", "")
            # Re-score to pick up any filter heuristic updates
            _, score, flags = score_response(entry.get("response", ""))
            entry["register_score"] = score
            entry["register_flags"] = flags
            if sid not in all_entries or score > all_entries[sid]["register_score"]:
                all_entries[sid] = entry
            file_count += 1
            total_loaded += 1
        print(f"[filter] Loaded {file_count} entries from {raw_file.name}")

    entries = list(all_entries.values())
    print(f"\n[filter] {total_loaded} total entries across {len(existing)} file(s)")
    print(f"[filter] {len(entries)} unique scenario IDs (best score per ID selected)")

    passed = []
    marginal = []
    failed = []

    report_lines = [
        "Humanist Dataset Filter Report",
        f"Raw files: {', '.join(f.name for f in existing)}",
        f"Unique IDs: {len(entries)}",
        "=" * 70,
    ]

    for entry in entries:
        sid = entry.get("id", "?")
        model_short = entry.get("model", "?").split("/")[-1]
        response = entry.get("response", "")
        score = entry.get("register_score", 0)
        flags = entry.get("register_flags", [])
        passes = score >= 6
        verdict = "PASS" if passes else ("MARGINAL" if score >= 4 else "FAIL")

        report_lines.append(
            f"{sid:<8} score={score:>3} {verdict:<8} [{model_short:<20}] | {', '.join(flags)}"
        )

        if passes:
            passed.append(entry)
        elif score >= 4:
            marginal.append(entry)
        else:
            failed.append(entry)

    report_lines += [
        "=" * 70,
        f"PASS:     {len(passed):>3}",
        f"MARGINAL: {len(marginal):>3}  (review manually before including)",
        f"FAIL:     {len(failed):>3}",
        f"TOTAL:    {len(entries):>3}",
        "",
        "Final dataset includes PASS entries only.",
        "Add MARGINAL entries manually after review.",
    ]

    report_text = "\n".join(report_lines)
    print(report_text)
    report_file.write_text(report_text, encoding="utf-8")
    print(f"\n[filter] Report written: {report_file}")

    # Write final dataset (PASS entries only, grief_dataset format)
    with out_file.open("w", encoding="utf-8") as f:
        for entry in passed:
            final_entry = {
                "instruction": entry["instruction"],
                "context": entry.get("context", ""),
                "response": entry["response"],
                "system_prompt": humanist_prompt,
                "source_id": entry.get("id", ""),
                "domain": entry.get("domain", ""),
                "era": entry.get("era", ""),
                "region": entry.get("region", ""),
                "register_score": entry.get("register_score", 0),
                "model": entry.get("model", ""),
            }
            f.write(json.dumps(final_entry, ensure_ascii=False) + "\n")

    print(f"[filter] Final dataset written: {out_file} ({len(passed)} entries)")

    if marginal:
        print(f"\n[filter] NOTE: {len(marginal)} MARGINAL entries need manual review.")
        print("  Add them to the final dataset if they read as correct register.")
        print("  Marginal IDs: " + ", ".join(e.get("id", "?") for e in marginal))


# ── Main ──────────────────────────────────────────────────────────────────────

def load_scenarios() -> list[dict]:
    if not SCENARIOS_FILE.exists():
        print(f"Error: scenarios file not found: {SCENARIOS_FILE}", file=sys.stderr)
        sys.exit(1)
    scenarios = []
    for line in SCENARIOS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            scenarios.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Warning: bad JSON line: {e}", file=sys.stderr)
    return scenarios


def load_humanist_prompt() -> str:
    if not HUMANIST_PROMPT_FILE.exists():
        print(f"Error: Humanist prompt not found: {HUMANIST_PROMPT_FILE}", file=sys.stderr)
        sys.exit(1)
    return HUMANIST_PROMPT_FILE.read_text(encoding="utf-8").strip()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate Humanist LoRA training dataset")
    p.add_argument(
        "--model",
        default=PRIMARY_MODEL,
        help=f"OpenRouter model slug (default: {PRIMARY_MODEL})",
    )
    p.add_argument(
        "--k2",
        action="store_true",
        help=f"Run K2 pass ({K2_MODEL}) — maximum voice intensity",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison pass with GLM-4.5-air:free (register calibration)",
    )
    p.add_argument(
        "--filter-only",
        action="store_true",
        dest="filter_only",
        help="Skip generation; run filter pass on all existing raw outputs",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Show what would be called without making API calls",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds between API calls (default: 1.5)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process first N scenarios (0 = all; useful for test runs)",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    scenarios = load_scenarios()
    if args.limit > 0:
        scenarios = scenarios[: args.limit]
        print(f"[main] --limit {args.limit}: processing {len(scenarios)} scenarios")

    print(f"[main] {len(scenarios)} scenarios loaded from {SCENARIOS_FILE.name}")
    print(f"[main] Humanist prompt: {HUMANIST_PROMPT_FILE}")

    # Collect all raw output files for filter pass
    all_raw_files = [RAW_OUTPUT_FILE, RAW_DEEPSEEK_FILE, RAW_K2_FILE, GLM_COMPARE_FILE]

    if args.filter_only:
        run_filter(all_raw_files, FINAL_DATASET_FILE, FILTER_REPORT_FILE)
        return

    full_prompt = load_humanist_prompt()
    gen_prompt = build_generation_prompt(full_prompt)

    if gen_prompt != full_prompt:
        print("[main] Stage 3 WitnessPause section trimmed from generation prompt")

    api_key = get_api_key() if not args.dry_run else "dry-run"

    # Primary generation pass (DeepSeek or --model override)
    primary_out = RAW_DEEPSEEK_FILE if args.model == PRIMARY_MODEL else (
        DATASET_DIR / f"humanist_raw_{args.model.split('/')[-1].replace(':', '_')}.jsonl"
    )
    run_generation(
        model=args.model,
        system_prompt=gen_prompt,
        scenarios=scenarios,
        output_file=primary_out,
        api_key=api_key,
        dry_run=args.dry_run,
        delay=args.delay,
    )

    # K2 pass
    if args.k2 and not args.dry_run:
        print(f"\n[main] Running K2 pass ({K2_MODEL})...")
        print("[main] K2 has opinions. This may take a while.")
        run_generation(
            model=K2_MODEL,
            system_prompt=gen_prompt,
            scenarios=scenarios,
            output_file=RAW_K2_FILE,
            api_key=api_key,
            dry_run=False,
            delay=args.delay * 2,  # K2 thinks; give it breathing room
        )
    elif args.k2 and args.dry_run:
        print(f"\n[main] [DRY RUN] Would run K2 pass ({K2_MODEL}) on {len(scenarios)} scenarios")

    # GLM-air comparison pass
    if args.compare and not args.dry_run:
        print(f"\n[main] Running GLM-air comparison pass ({COMPARE_MODEL})...")
        run_generation(
            model=COMPARE_MODEL,
            system_prompt=gen_prompt,
            scenarios=scenarios,
            output_file=GLM_COMPARE_FILE,
            api_key=api_key,
            dry_run=False,
            delay=args.delay * 3,  # GLM-air is rate-limited — be gentle
        )

    # Auto-run filter after generation
    if not args.dry_run:
        print("\n[main] Running filter pass on all raw outputs...")
        run_filter(all_raw_files, FINAL_DATASET_FILE, FILTER_REPORT_FILE)


if __name__ == "__main__":
    main()
