"""
Grammar MCQ Agent — generates 20 English grammar multiple-choice questions using OpenAI.
"""

import json
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "gpt-4o-mini"
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "grammar_questions.json"

SYSTEM_PROMPT = """You are an expert English grammar teacher with 20 years of experience
designing standardized tests and assessments. Your questions are clear, unambiguous, and
pedagogically sound. Each question tests a distinct grammar concept and has exactly one
correct answer. Distractors (wrong options) are plausible but clearly incorrect to someone
who knows the rule."""

USER_PROMPT = """Generate exactly 20 multiple-choice questions (MCQs) covering English grammar.
Cover these topics (at least 2 questions each):
- Verb tenses (simple, perfect, continuous, past/present/future)
- Parts of speech (nouns, verbs, adjectives, adverbs, pronouns)
- Subject-verb agreement
- Punctuation (commas, apostrophes, semicolons, colons)
- Articles (a, an, the — definite vs indefinite)
- Prepositions (in, on, at, by, for, with, etc.)
- Conjunctions (coordinating, subordinating, correlative)
- Sentence structure (simple, compound, complex; clauses; fragments)

Format each question EXACTLY like this (no extra blank lines between fields):

Q1. [Question text here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Answer: [Letter]) [Correct option text]
Explanation: [One sentence explaining why this is correct]

Q2. [Question text here]
...and so on through Q20.

Do not include any preamble or closing remarks — output only the 20 questions."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_mcqs(text: str) -> list[dict]:
    """Parse the raw model output into a list of structured question dicts."""
    questions = []

    # Split on question markers like "Q1.", "Q2.", ...
    blocks = re.split(r'\n(?=Q\d+\.)', text.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Extract question number and text
        q_match = re.match(r'Q(\d+)\.\s*(.+?)(?=\nA\))', block, re.DOTALL)
        if not q_match:
            continue
        q_num = int(q_match.group(1))
        q_text = q_match.group(2).strip()

        # Extract options A–D
        options = {}
        for letter in "ABCD":
            opt_match = re.search(rf'{letter}\)\s*(.+?)(?=\n[B-D]\)|\nAnswer:)', block, re.DOTALL)
            if opt_match:
                options[letter] = opt_match.group(1).strip()

        # Extract answer letter and explanation
        ans_match = re.search(r'Answer:\s*([A-D])\)\s*(.+?)(?=\nExplanation:|\Z)', block, re.DOTALL)
        exp_match = re.search(r'Explanation:\s*(.+)', block, re.DOTALL)

        answer_letter = ans_match.group(1).strip() if ans_match else ""
        explanation = exp_match.group(1).strip() if exp_match else ""

        if q_text and options and answer_letter:
            questions.append({
                "number": q_num,
                "question": q_text,
                "options": options,
                "answer": answer_letter,
                "explanation": explanation,
            })

    return questions


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_questions(questions: list[dict]) -> None:
    """Print all questions in a clean, readable format."""
    separator = "─" * 60

    print(f"\n{'='*60}")
    print(f"  ENGLISH GRAMMAR MCQ — {len(questions)} Questions")
    print(f"{'='*60}\n")

    for q in questions:
        print(f"Q{q['number']}. {q['question']}")
        for letter, text in q["options"].items():
            marker = "✓" if letter == q["answer"] else " "
            print(f"  {marker} {letter}) {text}")
        print(f"  → Answer: {q['answer']}")
        print(f"  → {q['explanation']}")
        print(separator)

    print(f"\nTotal: {len(questions)} questions generated.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        print("Set it with:  set OPENAI_API_KEY=your-key-here", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print("Generating 20 English grammar MCQs via OpenAI…")

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )

    raw_text = response.choices[0].message.content

    questions = parse_mcqs(raw_text)

    if len(questions) < 20:
        print(f"Warning: parsed only {len(questions)} questions (expected 20). "
              "Saving what was parsed.", file=sys.stderr)

    # Display in terminal
    display_questions(questions)

    # Save to JSON
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": MODEL,
        "topic": "English Grammar",
        "total": len(questions),
        "questions": questions,
    }
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Questions saved to: {OUTPUT_FILE}")

    # Show token usage
    usage = response.usage
    print(f"\nToken usage — prompt: {usage.prompt_tokens} | completion: {usage.completion_tokens} | total: {usage.total_tokens}")


if __name__ == "__main__":
    main()
