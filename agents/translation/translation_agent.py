"""
Main Translation Agent — orchestrates the full Hindi ↔ English translation workflow.

Flow:
  1. Sub-Agent 1 (hindi_sentence_agent)   — generates N Hindi sentences
  2. Main agent                           — presents each sentence, collects user translations
  3. Sub-Agent 2 (english_evaluator_agent) — evaluates every translation and reports

Usage:
    python translation_agent.py           # 10 sentences (default)
    python translation_agent.py --count 5 # custom number
"""

import argparse
import os
import sys
from pathlib import Path

from openai import OpenAI

# Allow imports from the same agents/ directory
sys.path.insert(0, str(Path(__file__).parent))

import hindi_sentence_agent as agent1
import english_evaluator_agent as agent2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_COUNT = 10
DIFF_EMOJI = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _banner(title: str, width: int = 60) -> None:
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}\n")


def _collect_translations(sentences: list[dict]) -> list[dict]:
    """Show each Hindi sentence and collect the user's English translation."""
    _banner("STEP 2 / 3 — TRANSLATE EACH SENTENCE")
    print("Type your English translation and press Enter.")
    print("Leave blank and press Enter to skip a sentence.\n")

    pairs = []
    for s in sentences:
        emoji = DIFF_EMOJI.get(s["difficulty"], "⚪")
        print(f"[{s['id']:02d}/{len(sentences)}]  {emoji} {s['difficulty'].upper()}"
              f"  |  topic: {s['topic']}")
        print(f"        Hindi  : {s['hindi']}")
        answer = input("        English: ").strip()
        print()
        pairs.append({
            "id":                s["id"],
            "hindi":             s["hindi"],
            "english_reference": s["english_reference"],
            "user_translation":  answer if answer else "(skipped)",
        })

    return pairs


def _summary_line(evaluations: list[dict]) -> None:
    avg = round(sum(e["overall"] for e in evaluations) / len(evaluations), 2)
    grade = agent2._grade(avg)
    print(f"\n  ✦  Session complete — your average: {avg:.2f}/5  [{grade}]")
    print(f"  ✦  Full report saved to: {agent2.OUTPUT_FILE}\n")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main(count: int = DEFAULT_COUNT) -> None:
    # ── Auth ────────────────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        print("Set it with:  set OPENAI_API_KEY=your-key-here", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # ── Step 1: Sub-Agent 1 — generate Hindi sentences ──────────────────────
    _banner("STEP 1 / 3 — GENERATING HINDI SENTENCES  [Sub-Agent 1]")
    print(f"  Requesting {count} Hindi sentences from {agent1.MODEL} …\n")

    sentences, usage1 = agent1.generate(client, count=count)
    agent1.display(sentences)
    agent1.save(sentences, usage1)

    input("  Press Enter when you're ready to start translating …")

    # ── Step 2: Collect user translations ────────────────────────────────────
    pairs = _collect_translations(sentences)

    # ── Step 3: Sub-Agent 2 — evaluate translations ──────────────────────────
    _banner("STEP 3 / 3 — EVALUATING YOUR TRANSLATIONS  [Sub-Agent 2]")
    print(f"  Sending {len(pairs)} translations to {agent2.MODEL} for evaluation …\n")

    source_model = agent1.MODEL
    evaluations, usage2 = agent2.evaluate(client, pairs)
    agent2.display_report(evaluations)
    agent2.save(evaluations, source_model, usage2)

    # ── Final summary ─────────────────────────────────────────────────────────
    _summary_line(evaluations)

    total_tokens = (
        (usage1.prompt_tokens + usage1.completion_tokens) +
        (usage2.prompt_tokens + usage2.completion_tokens)
    )
    print(f"  Total tokens used this session: {total_tokens}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hindi → English translation practice with AI evaluation."
    )
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT,
        help=f"Number of Hindi sentences to generate (default: {DEFAULT_COUNT})"
    )
    args = parser.parse_args()
    main(count=args.count)
