"""
Sub-Agent 1 — Hindi Sentence Generator.

Generates Hindi sample sentences at three difficulty levels, each paired
with a reference English translation. Results are saved to:
    data/hindi_sentences.json

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
OUTPUT_FILE = DATA_DIR / "hindi_sentences.json"
DEFAULT_COUNT = 10

SYSTEM_PROMPT = """You are a professional Hindi language teacher and linguist with deep
expertise in Hindi grammar, vocabulary, and everyday usage. You create pedagogically
rich sentences that help learners practise real-world Hindi comprehension and
English translation skills.

Guidelines:
- Use standard Modern Standard Hindi (Devanagari script).
- Easy sentences: present tense, common everyday vocabulary (≤ 8 words).
- Medium sentences: past/future tense, compound verbs, slightly richer vocab (8–14 words).
- Hard sentences: complex clauses, idioms or formal register (14+ words).
- Each sentence must be natural, grammatically correct, and culturally appropriate.
- The English reference translation must be accurate, fluent, and idiomatic."""

# ---------------------------------------------------------------------------
# Core function (importable by main agent)
# ---------------------------------------------------------------------------

def generate(client: OpenAI, count: int = DEFAULT_COUNT) -> list[dict]:
    """
    Ask the model to generate `count` Hindi sentences and return them as a list
    of dicts with keys: id, hindi, english_reference, topic, difficulty.
    """
    easy   = max(1, count // 3)
    hard   = max(1, count // 5)
    medium = count - easy - hard

    user_prompt = f"""Generate exactly {count} Hindi sentences for translation practice.
Distribution:
  - {easy} EASY   sentences (present tense, everyday vocab, ≤ 8 words)
  - {medium} MEDIUM sentences (past/future tense, compound verbs, 8–14 words)
  - {hard} HARD   sentences (complex clauses or idioms, 14+ words)

Cover varied topics: daily routine, food, travel, family, work, weather,
feelings, shopping, education, nature.

Return ONLY a JSON array — no markdown fences, no extra text.
Each element must have exactly these fields:
  "id"                : integer (1-based)
  "hindi"             : string  (Devanagari script)
  "english_reference" : string  (accurate, fluent English translation)
  "topic"             : string  (one or two words)
  "difficulty"        : string  ("easy" | "medium" | "hard")"""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        temperature=0.8,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()

    sentences = json.loads(raw)

    # Guarantee 1-based IDs regardless of what the model returned
    for i, s in enumerate(sentences, start=1):
        s["id"] = i

    return sentences, response.usage


def save(sentences: list[dict], usage) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": MODEL,
        "total": len(sentences),
        "sentences": sentences,
    }
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Sentences saved to: {OUTPUT_FILE}")
    print(f"Token usage — prompt: {usage.prompt_tokens} | "
          f"completion: {usage.completion_tokens} | total: {usage.total_tokens}")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display(sentences: list[dict]) -> None:
    diff_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}

    print(f"\n{'='*60}")
    print(f"  HINDI SENTENCES  ({len(sentences)} generated)")
    print(f"{'='*60}\n")

    for s in sentences:
        emoji = diff_emoji.get(s["difficulty"], "⚪")
        print(f"[{s['id']:02d}] {emoji} {s['difficulty'].upper()}  |  {s['topic']}")
        print(f"     Hindi   : {s['hindi']}")
        print(f"     English : {s['english_reference']}")
        print()


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        print("Set it with:  set OPENAI_API_KEY=your-key-here", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    print(f"Generating {DEFAULT_COUNT} Hindi sentences via {MODEL} …\n")

    sentences, usage = generate(client, count=DEFAULT_COUNT)
    display(sentences)
    save(sentences, usage)


if __name__ == "__main__":
    main()
