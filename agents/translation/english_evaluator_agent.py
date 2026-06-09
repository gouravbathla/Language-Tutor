"""
Sub-Agent 2 — English Translation Evaluator.

Receives pairs of (Hindi sentence, reference English, user's English) and
scores each user translation on four criteria:
  - accuracy     : meaning fully preserved
  - grammar      : grammatically correct English
  - naturalness  : sounds like fluent, idiomatic English
  - completeness : no parts of the Hindi omitted

Scores are 1–5 per criterion; overall = average of the four.
Results are saved to:
    data/translation_evaluation.json

Can be run standalone or imported by the main translation_agent.
"""

import json
import os
import sys
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "gpt-4o-mini"
DATA_DIR = Path(__file__).parent.parent / "data"
SENTENCES_FILE = DATA_DIR / "hindi_sentences.json"
OUTPUT_FILE    = DATA_DIR / "translation_evaluation.json"

SYSTEM_PROMPT = """You are an expert bilingual evaluator with native-level proficiency
in both Hindi and English. Your role is to assess how accurately and fluently a learner
has translated a Hindi sentence into English.

Scoring rubric (1–5 for each criterion):
  5 = Perfect / Excellent
  4 = Good, minor issues
  3 = Acceptable, noticeable but not critical errors
  2 = Weak, significant errors that affect understanding
  1 = Poor, translation is largely incorrect or missing

Criteria:
  accuracy     — Does the English preserve the full meaning of the Hindi?
  grammar      — Is the English grammatically correct?
  naturalness  — Does it sound like natural, idiomatic English?
  completeness — Is every part of the Hindi sentence represented?

Be fair but strict. Accept paraphrases that preserve meaning even if word-for-word
different from the reference translation.

Return ONLY a JSON array, no markdown, no extra text.
Each element must have:
  "id"           : integer  — matches the input id
  "accuracy"     : integer 1-5
  "grammar"      : integer 1-5
  "naturalness"  : integer 1-5
  "completeness" : integer 1-5
  "feedback"     : string   — one clear sentence of constructive feedback
  "ideal"        : string   — the best possible English translation of this Hindi sentence"""

CRITERIA = ["accuracy", "grammar", "naturalness", "completeness"]

# ---------------------------------------------------------------------------
# Core function (importable by main agent)
# ---------------------------------------------------------------------------

def evaluate(client: OpenAI, pairs: list[dict]) -> tuple[list[dict], object]:
    """
    Evaluate a list of translation pairs.

    Each pair dict must have: id, hindi, english_reference, user_translation.
    Returns (evaluations_list, usage).
    """
    user_prompt = (
        "Evaluate each of the following Hindi-to-English translation attempts "
        "and return a JSON array as described in your instructions.\n\n"
        + json.dumps(pairs, indent=2, ensure_ascii=False)
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=3000,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()

    evaluations = json.loads(raw)

    # Attach computed fields
    for ev, pair in zip(evaluations, pairs):
        ev["hindi"]             = pair["hindi"]
        ev["english_reference"] = pair["english_reference"]
        ev["user_translation"]  = pair["user_translation"]
        ev["overall"]           = round(sum(ev[c] for c in CRITERIA) / len(CRITERIA), 2)
        ev["grade"]             = _grade(ev["overall"])

    return evaluations, response.usage


def _grade(score: float) -> str:
    if score >= 4.5: return "Excellent"
    if score >= 3.5: return "Good"
    if score >= 2.5: return "Fair"
    return "Needs Work"


def save(evaluations: list[dict], source_model: str, usage) -> None:
    avg = round(sum(e["overall"] for e in evaluations) / len(evaluations), 2)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_model": source_model,
        "evaluator_model": MODEL,
        "total": len(evaluations),
        "average_overall": avg,
        "evaluations": evaluations,
    }
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nEvaluation saved to: {OUTPUT_FILE}")
    print(f"Token usage — prompt: {usage.prompt_tokens} | "
          f"completion: {usage.completion_tokens} | total: {usage.total_tokens}")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_report(evaluations: list[dict]) -> None:
    grade_emoji = {
        "Excellent": "🌟", "Good": "✅", "Fair": "🟡", "Needs Work": "❌"
    }
    sep = "─" * 66

    print(f"\n{'='*66}")
    print(f"  TRANSLATION EVALUATION REPORT  ({len(evaluations)} sentences)")
    print(f"{'='*66}\n")

    criterion_totals = {c: 0 for c in CRITERIA}

    for ev in evaluations:
        emoji = grade_emoji.get(ev["grade"], "")
        print(f"[{ev['id']:02d}]  {emoji} Overall: {ev['overall']:.1f}/5  — {ev['grade']}")
        print(f"      Hindi    : {ev['hindi']}")
        print(f"      Yours    : {ev['user_translation']}")
        print(f"      Ideal    : {ev['ideal']}")
        print(f"      Scores   : accuracy {ev['accuracy']}  grammar {ev['grammar']}  "
              f"naturalness {ev['naturalness']}  completeness {ev['completeness']}")
        print(f"      Feedback : {ev['feedback']}")
        print(sep)
        for c in CRITERIA:
            criterion_totals[c] += ev[c]

    n = len(evaluations)
    avg_overall = round(sum(e["overall"] for e in evaluations) / n, 2)

    print(f"\n{'─'*66}")
    print(f"  YOUR AVERAGE SCORES  (across {n} sentences)")
    print(f"{'─'*66}")
    for c in CRITERIA:
        avg = round(criterion_totals[c] / n, 2)
        filled = int(avg)
        bar = "█" * filled + "░" * (5 - filled)
        print(f"  {c.capitalize():<13} {bar}  {avg:.2f}/5")
    print(f"{'─'*66}")
    print(f"  Overall score   {avg_overall:.2f}/5  [{_grade(avg_overall)}]")
    print(f"{'─'*66}\n")

    # Highlight weakest sentence
    worst = min(evaluations, key=lambda e: e["overall"])
    best  = max(evaluations, key=lambda e: e["overall"])
    print(f"  🏆 Best translation  : Q{best['id']:02d} ({best['overall']:.1f}/5)")
    print(f"  📌 Needs most work   : Q{worst['id']:02d} ({worst['overall']:.1f}/5) — {worst['feedback']}\n")


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

def _load_sentences() -> tuple[str, list[dict]]:
    if not SENTENCES_FILE.exists():
        print(f"Error: {SENTENCES_FILE} not found.", file=sys.stderr)
        print("Run hindi_sentence_agent.py first to generate sentences.", file=sys.stderr)
        sys.exit(1)
    data = json.loads(SENTENCES_FILE.read_text(encoding="utf-8"))
    return data.get("model", "unknown"), data.get("sentences", [])


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    source_model, sentences = _load_sentences()
    if not sentences:
        print("Error: no sentences found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  HINDI → ENGLISH TRANSLATION PRACTICE")
    print(f"{'='*60}")
    print("Type your English translation for each Hindi sentence.")
    print("Press Enter to submit.\n")

    diff_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
    pairs = []

    for s in sentences:
        emoji = diff_emoji.get(s["difficulty"], "⚪")
        print(f"[{s['id']:02d}] {emoji} {s['difficulty'].upper()}  ({s['topic']})")
        print(f"     Hindi: {s['hindi']}")
        user_input = input("     Your English: ").strip()
        print()
        pairs.append({
            "id":                 s["id"],
            "hindi":              s["hindi"],
            "english_reference":  s["english_reference"],
            "user_translation":   user_input or "(no answer)",
        })

    client = OpenAI(api_key=api_key)
    print(f"Evaluating {len(pairs)} translations via {MODEL} …")

    evaluations, usage = evaluate(client, pairs)
    display_report(evaluations)
    save(evaluations, source_model, usage)


if __name__ == "__main__":
    main()
