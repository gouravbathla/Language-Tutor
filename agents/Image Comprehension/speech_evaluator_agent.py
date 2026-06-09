"""
Sub-Agent 2 — Speech Input & Description Evaluator.

  1. Records the user's spoken image description from the microphone.
  2. Transcribes the recording with OpenAI Whisper.
  3. Evaluates the transcription against the reference description using
     GPT-4o Vision (which also sees the actual image for maximum accuracy).

Evaluation criteria (1–5 each):
  accuracy     — correctly identifies the main subjects and scene
  detail       — level of specific visual detail provided
  vocabulary   — quality and range of descriptive language used
  completeness — coverage of all major visible elements

Outputs:
  data/user_speech.wav
  data/image_evaluation.json

Can be run standalone or imported by image_comprehension_agent.
"""

import base64
import json
import os
import sys
import wave
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_WHISPER = "whisper-1"
MODEL_EVAL    = "gpt-4o"

DATA_DIR    = Path(__file__).parent.parent / "data"
META_FILE   = DATA_DIR / "generated_image.json"
AUDIO_FILE  = DATA_DIR / "user_speech.wav"
OUTPUT_FILE = DATA_DIR / "image_evaluation.json"

SAMPLE_RATE = 16000   # Whisper works best at 16 kHz
CHANNELS    = 1

CRITERIA = ["accuracy", "detail", "vocabulary", "completeness"]

EVAL_SYSTEM = """\
You are an expert visual-language evaluator. You receive:
  • The actual image.
  • A reference description produced by an AI image analyst.
  • A description given by a human user (transcribed from speech).

Score the human's description on four criteria (integer 1–5 each):
  accuracy     — correctly identifies main subjects and scene
  detail       — level of specific visual detail
  vocabulary   — quality and range of descriptive language
  completeness — covers all major visible elements

Return ONLY a JSON object with exactly these fields (no markdown):
  "accuracy"             : integer 1–5
  "detail"               : integer 1–5
  "vocabulary"           : integer 1–5
  "completeness"         : integer 1–5
  "correct_observations" : list[str] — things the user described correctly
  "missed_elements"      : list[str] — important things the user omitted
  "feedback"             : str — 2–3 sentences of constructive feedback
  "improved_version"     : str — a polished rewrite of the user's description\
"""

# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------

def record_speech() -> Path | None:
    """
    Stream microphone audio until the user presses Enter.
    Saves a 16-bit mono WAV file to AUDIO_FILE.
    """
    frames: list[np.ndarray] = []
    stop_event = threading.Event()

    def _callback(indata, frame_count, time_info, status):
        if not stop_event.is_set():
            frames.append(indata.copy())

    print("\n  🎤  Press Enter to START recording your description …", flush=True)
    input()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=_callback,
    )
    stream.start()
    print("  🔴  Recording … Describe the image aloud. Press Enter to STOP.", flush=True)
    input()

    stop_event.set()
    stream.stop()
    stream.close()

    if not frames:
        print("  Warning: no audio captured.", file=sys.stderr)
        return None

    audio = np.concatenate(frames, axis=0)
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with wave.open(str(AUDIO_FILE), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)        # 16-bit = 2 bytes per sample
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

    duration = len(audio) / SAMPLE_RATE
    print(f"  ✅  Recording saved ({duration:.1f}s) → {AUDIO_FILE}")
    return AUDIO_FILE


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(client: OpenAI, wav_path: Path) -> str:
    """Send WAV file to Whisper and return the transcript."""
    with open(wav_path, "rb") as fh:
        resp = client.audio.transcriptions.create(
            model=MODEL_WHISPER,
            file=fh,
            language="en",
        )
    return resp.text.strip()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_description(
    client: OpenAI,
    user_description: str,
    actual_description: str,
    image_path: Path,
) -> tuple[dict, object]:
    """Compare user's description against reference, looking at the image directly."""
    with open(image_path, "rb") as fh:
        b64 = base64.standard_b64encode(fh.read()).decode()

    user_prompt = (
        f"Reference description (AI analyst):\n{actual_description}\n\n"
        f"Human user's description:\n{user_description}\n\n"
        "Evaluate the human's description as instructed."
    )

    resp = client.chat.completions.create(
        model=MODEL_EVAL,
        max_tokens=800,
        temperature=0.2,
        messages=[
            {"role": "system", "content": EVAL_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
    )

    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    result = json.loads(raw)
    result["overall"] = round(sum(result[c] for c in CRITERIA) / len(CRITERIA), 2)
    result["grade"]   = _grade(result["overall"])
    return result, resp.usage


def _grade(score: float) -> str:
    if score >= 4.5: return "Excellent"
    if score >= 3.5: return "Good"
    if score >= 2.5: return "Fair"
    return "Needs Work"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

GRADE_EMOJI = {"Excellent": "🌟", "Good": "✅", "Fair": "🟡", "Needs Work": "❌"}


def display_evaluation(
    transcript: str,
    actual_description: str,
    result: dict,
) -> None:
    sep = "─" * 66

    print(f"\n{'='*66}")
    print("  IMAGE DESCRIPTION EVALUATION")
    print(f"{'='*66}\n")

    print("  YOUR DESCRIPTION (transcribed from speech):")
    print(f"  \"{transcript}\"\n")
    print(sep)
    print("  REFERENCE DESCRIPTION:")
    print(f"  \"{actual_description}\"\n")
    print(sep)

    emoji = GRADE_EMOJI.get(result["grade"], "")
    print(f"\n  Overall  {result['overall']:.1f}/5  {emoji}  {result['grade']}\n")

    for c in CRITERIA:
        bar = "█" * result[c] + "░" * (5 - result[c])
        print(f"  {c.capitalize():<14} {bar}  {result[c]}/5")

    print(f"\n{sep}")

    if result.get("correct_observations"):
        print("  ✓  What you got right:")
        for obs in result["correct_observations"]:
            print(f"       • {obs}")

    if result.get("missed_elements"):
        print("\n  ✗  Elements you missed:")
        for el in result["missed_elements"]:
            print(f"       • {el}")

    print(f"\n  📝  Feedback:")
    print(f"       {result['feedback']}")

    if result.get("improved_version"):
        print(f"\n  💡  Improved version:")
        print(f"       {result['improved_version']}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Public API (called by main agent)
# ---------------------------------------------------------------------------

def run(
    client: OpenAI,
    actual_description: str,
    image_path: Path,
) -> dict:
    """
    Record → transcribe → evaluate. Saves results. Returns evaluation dict.
    """
    # 1. Record
    wav_path = record_speech()
    if wav_path is None:
        sys.exit(1)

    # 2. Transcribe
    print("\n  🔄  Transcribing with Whisper …")
    transcript = transcribe(client, wav_path)
    print(f"  📝  Transcript: \"{transcript}\"\n")

    # 3. Evaluate
    print(f"  🔍  Evaluating with {MODEL_EVAL} Vision …")
    result, usage = evaluate_description(client, transcript, actual_description, image_path)

    # 4. Display
    display_evaluation(transcript, actual_description, result)

    # 5. Save
    payload = {
        "evaluator_model":    MODEL_EVAL,
        "whisper_model":      MODEL_WHISPER,
        "audio_file":         str(AUDIO_FILE),
        "user_transcript":    transcript,
        "actual_description": actual_description,
        "evaluation":         result,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Evaluation saved → {OUTPUT_FILE}")
    print(f"  Tokens — prompt: {usage.prompt_tokens} | "
          f"completion: {usage.completion_tokens} | total: {usage.total_tokens}")

    return result


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    if not META_FILE.exists():
        print(f"Error: {META_FILE} not found.", file=sys.stderr)
        print("Run image_generator_agent.py first.", file=sys.stderr)
        sys.exit(1)

    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    client = OpenAI(api_key=api_key)
    run(client, meta["actual_description"], Path(meta["image_path"]))


if __name__ == "__main__":
    main()
