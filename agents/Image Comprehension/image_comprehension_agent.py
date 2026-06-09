"""
Main Image Comprehension Agent — orchestrates the full workflow.

  Step 1  Sub-Agent 1 (image_generator_agent)
            • Generates an image with DALL-E 3
            • Saves it locally
            • Produces the authoritative reference description via GPT-4o Vision

  Step 2  Main agent
            • Opens the image in the OS default viewer
            • Lets the user study it as long as they wish

  Step 3  Sub-Agent 2 (speech_evaluator_agent)
            • Records the user's spoken description (microphone → WAV)
            • Transcribes with Whisper
            • Evaluates against the reference description using GPT-4o Vision
            • Prints scores, feedback, and an improved version

Usage:
    python image_comprehension_agent.py
    python image_comprehension_agent.py --prompt "a dog playing in the park"
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

# Allow sibling-agent imports
sys.path.insert(0, str(Path(__file__).parent))

import image_generator_agent  as agent1
import speech_evaluator_agent as agent2

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _banner(step: str, title: str, width: int = 66) -> None:
    print(f"\n{'='*width}")
    print(f"  {step}  {title}")
    print(f"{'='*width}\n")


def _open_image(path: Path) -> None:
    """Open the image in the default OS viewer."""
    try:
        os.startfile(str(path))           # Windows
    except AttributeError:
        try:
            subprocess.Popen(["xdg-open", str(path)])   # Linux
        except FileNotFoundError:
            subprocess.Popen(["open", str(path)])        # macOS


def _summary(meta: dict, result: dict) -> None:
    grade_emoji = {
        "Excellent": "🌟", "Good": "✅", "Fair": "🟡", "Needs Work": "❌"
    }
    emoji = grade_emoji.get(result["grade"], "")
    total_tokens = (
        agent1.MODEL_DALLE  # not token-based, skip
    )
    print("  ╔══════════════════════════════════════════╗")
    print(f"  ║  SESSION COMPLETE                         ║")
    print(f"  ║  Image   : {Path(meta['image_path']).name:<30} ║")
    print(f"  ║  Score   : {result['overall']:.1f}/5  {emoji}  {result['grade']:<15} ║")
    print(f"  ║  Report  : {Path(agent2.OUTPUT_FILE).name:<30} ║")
    print("  ╚══════════════════════════════════════════╝\n")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main(prompt: str | None = None) -> None:
    # ── Auth ────────────────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        print("       set OPENAI_API_KEY=your-key-here", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # ── Step 1 : Generate image ──────────────────────────────────────────────
    _banner("STEP 1/3", "IMAGE GENERATION  [Sub-Agent 1 · DALL-E 3 + GPT-4o Vision]")
    meta = agent1.run(client, prompt)

    image_path = Path(meta["image_path"])
    print(f"  Reference (first 100 chars): {meta['actual_description'][:100]}…")

    # ── Step 2 : Let user study the image ────────────────────────────────────
    _banner("STEP 2/3", "STUDY THE IMAGE")
    print(f"  Opening  →  {image_path.name}")
    _open_image(image_path)
    print("\n  Look at the image carefully — observe colours, objects, mood, composition.")
    print("  When you are ready to describe it by voice, press Enter here.\n")
    input("  Press Enter to proceed to the recording step …")

    # ── Step 3 : Record speech & evaluate ────────────────────────────────────
    _banner("STEP 3/3", "SPEECH DESCRIPTION & EVALUATION  [Sub-Agent 2 · Whisper + GPT-4o]")
    print("  You will be asked to speak a description of the image.")
    print("  Press Enter to start recording, describe the image,")
    print("  then press Enter again to stop.\n")

    result = agent2.run(
        client=client,
        actual_description=meta["actual_description"],
        image_path=image_path,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    _summary(meta, result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Image comprehension practice: "
            "generate an image → study it → describe it by voice → get AI feedback."
        )
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Custom DALL-E prompt (default: random topic)",
    )
    args = parser.parse_args()
    main(prompt=args.prompt)
