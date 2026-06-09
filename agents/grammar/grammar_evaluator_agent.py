"""
Grammar Evaluator Agent — evaluates MCQs produced by grammar_mcq_agent.py.

For each question it scores:
  - clarity       : Is the question stem unambiguous and grammatically correct?
  - correctness   : Is the marked answer actually correct?
  - distractors   : Are the wrong options plausible but clearly incorrect?
  - explanation   : Is the explanation accurate and helpful?
  - topic_fit     : Does the question genuinely test the stated grammar topic?

Overall score per question: average of the five criteria (1–5 scale).
A final summary report is printed and saved to data/grammar_evaluation.json.
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
INPUT_FILE = DATA_DIR / "grammar_questions.json"
OUTPUT_FILE = DATA_DIR / "grammar_evaluation.json"

EVALUATOR_SYSTEM_PROMPT = """You are a senior English language assessment specialist with
expertise in psychometrics and grammar pedagogy. Your job is to critically evaluate
multiple-choice grammar questions produced by another AI agent.

Be objective and strict. A score of 5 means excellent; a score of 1 means poor.

You will receive a batch of questions formatted as JSON. Return a JSON array where
each element corresponds to one question and has exactly these fields:
  "number"      : integer — the question number
  "clarity"     : integer 1-5 — stem is unambiguous and well-worded
  "correctness" : integer 1-5 — the marked answer is definitively correct
  "distractors" : integer 1-5 — wrong options are plausible but clearly incorrect
  "explanation" : integer 1-5 — explanation is accurate and instructive
  "topic_fit"   : integer 1-5 — question genuinely tests the grammar concept claimed
  "issues"      : string — one sentence describing the main weakness (or "None" if none)

Return ONLY the JSON array, no markdown fences, no extra text."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_questions(path: Path) -> tuple[str, list[dict]]:
    """Load questions from the JSON file produced by grammar_mcq_agent."""
    if not path.exists():
        print(f"Error: input file not found: {path}", file=sys.stderr)
        print("Run grammar_mcq_agent.py first to generate questions.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("model", "unknown"), data.get("questions", [])


def build_evaluation_prompt(questions: list[dict]) -> str:
    """Serialize questions into the prompt sent to the evaluator model."""
    slim = [
        {
            "number": q["number"],
            "question": q["question"],
            "options": q["options"],
            "answer": q["answer"],
            "explanation": q["explanation"],
        }
        for q in questions
    ]
    return json.dumps(slim, indent=2, ensure_ascii=False)


def call_evaluator(client: OpenAI, questions: list[dict]) -> list[dict]:
    """Send all questions to the model in one call and parse the JSON response."""
    user_content = (
        "Please evaluate the following grammar MCQs and return a JSON array "
        "as specified in your instructions:\n\n"
        + build_evaluation_prompt(questions)
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        temperature=0.2,          # low temperature for consistent scoring
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    evaluations = json.loads(raw)
    return evaluations, response.usage


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

CRITERIA = ["clarity", "correctness", "distractors", "explanation", "topic_fit"]


def overall(ev: dict) -> float:
    return round(sum(ev[c] for c in CRITERIA) / len(CRITERIA), 2)


def grade(score: float) -> str:
    if score >= 4.5:
        return "Excellent"
    if score >= 3.5:
        return "Good"
    if score >= 2.5:
        return "Fair"
    return "Poor"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_report(evaluations: list[dict], source_model: str) -> None:
    sep = "─" * 64

    print(f"\n{'='*64}")
    print(f"  GRAMMAR MCQ EVALUATION REPORT")
    print(f"  Source model : {source_model}")
    print(f"  Evaluator    : {MODEL}")
    print(f"  Questions    : {len(evaluations)}")
    print(f"{'='*64}\n")

    totals = {c: 0 for c in CRITERIA}

    for ev in evaluations:
        score = overall(ev)
        totals_update = {c: totals[c] + ev[c] for c in CRITERIA}
        totals = totals_update

        print(f"Q{ev['number']:02d}  Overall: {score:.1f}/5  [{grade(score)}]")
        print(f"     Clarity     {ev['clarity']}/5  |  "
              f"Correctness  {ev['correctness']}/5  |  "
              f"Distractors  {ev['distractors']}/5")
        print(f"     Explanation {ev['explanation']}/5  |  "
              f"Topic fit    {ev['topic_fit']}/5")
        if ev.get("issues") and ev["issues"].lower() != "none":
            print(f"     ⚠  {ev['issues']}")
        print(sep)

    n = len(evaluations)
    avg_overall = round(sum(overall(e) for e in evaluations) / n, 2)

    print(f"\n{'─'*64}")
    print(f"  AVERAGE SCORES  (across {n} questions)")
    print(f"{'─'*64}")
    for c in CRITERIA:
        avg = round(totals[c] / n, 2)
        bar = "█" * int(avg) + "░" * (5 - int(avg))
        print(f"  {c.capitalize():<13} {bar}  {avg:.2f}/5")
    print(f"{'─'*64}")
    print(f"  Overall quality  {avg_overall:.2f}/5  [{grade(avg_overall)}]")
    print(f"{'─'*64}\n")

    # Flag any question scoring below 2.5 overall
    weak = [e for e in evaluations if overall(e) < 2.5]
    if weak:
        print(f"  ⚠  {len(weak)} question(s) rated POOR — review recommended:")
        for e in weak:
            print(f"     Q{e['number']:02d}: {e.get('issues', '')}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        print("Set it with:  set OPENAI_API_KEY=your-key-here", file=sys.stderr)
        sys.exit(1)

    source_model, questions = load_questions(INPUT_FILE)

    if not questions:
        print("Error: no questions found in input file.", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(questions)} questions from {INPUT_FILE.name} …")

    client = OpenAI(api_key=api_key)
    evaluations, usage = call_evaluator(client, questions)

    # Attach overall score to each evaluation dict for easy downstream use
    for ev in evaluations:
        ev["overall"] = overall(ev)
        ev["grade"] = grade(ev["overall"])

    display_report(evaluations, source_model)

    # Save full report
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "source_file": str(INPUT_FILE),
        "source_model": source_model,
        "evaluator_model": MODEL,
        "total_questions": len(evaluations),
        "average_overall": round(
            sum(e["overall"] for e in evaluations) / len(evaluations), 2
        ),
        "evaluations": evaluations,
    }
    OUTPUT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluation saved to: {OUTPUT_FILE}")

    # Token usage
    print(f"\nToken usage — prompt: {usage.prompt_tokens} | "
          f"completion: {usage.completion_tokens} | total: {usage.total_tokens}")


if __name__ == "__main__":
    main()
