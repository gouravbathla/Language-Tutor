"""
Sub-Agent 1 — Image Generator.

  1. Uses DALL-E 3 to generate an image from a prompt.
  2. Downloads the image locally.
  3. Uses GPT-4o Vision to produce the authoritative reference description.

Outputs:
  data/generated_image.png
  data/generated_image.json  (prompt, revised_prompt, image_path, actual_description)

Can be run standalone or imported by image_comprehension_agent.
"""

import base64
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DALLE  = "dall-e-3"
MODEL_VISION = "gpt-4o"

DATA_DIR   = Path(__file__).parent.parent / "data"
IMAGE_FILE = DATA_DIR / "generated_image.png"
META_FILE  = DATA_DIR / "generated_image.json"

DEFAULT_TOPICS = [
    "a busy street market in an ancient Middle-Eastern city at golden hour",
    "a futuristic underwater research station surrounded by bioluminescent sea life",
    "a cozy log cabin in a snow-covered pine forest under the northern lights",
    "a vibrant carnival parade through a colourful South American town",
    "an astronaut planting a flag on a red alien planet with two moons in the sky",
    "a sunlit Japanese tea garden in full cherry blossom season",
    "a medieval blacksmith workshop with glowing forge, tools, and armour on the walls",
    "a treehouse village built high in an ancient rainforest canopy at sunrise",
]

VISION_SYSTEM = (
    "You are an objective image analyst. Describe the image in exactly 4–6 sentences, "
    "covering: main subjects, colours, composition, setting, mood, and any notable "
    "details. Be specific, concrete, and precise — avoid vague terms like 'beautiful'."
)

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _generate_image(client: OpenAI, prompt: str) -> tuple[str, str]:
    """Call DALL-E 3. Returns (url, revised_prompt)."""
    resp = client.images.generate(
        model=MODEL_DALLE,
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return resp.data[0].url, resp.data[0].revised_prompt


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest))


def _describe(client: OpenAI, image_path: Path) -> str:
    """Use GPT-4o Vision to get the reference description."""
    with open(image_path, "rb") as fh:
        b64 = base64.standard_b64encode(fh.read()).decode()

    resp = client.chat.completions.create(
        model=MODEL_VISION,
        max_tokens=512,
        messages=[
            {"role": "system", "content": VISION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            },
        ],
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Public API (called by main agent)
# ---------------------------------------------------------------------------

def run(client: OpenAI, prompt: str | None = None) -> dict:
    """
    Generate image, download it, describe it. Return metadata dict with keys:
      prompt, revised_prompt, image_path, actual_description
    """
    if not prompt:
        prompt = random.choice(DEFAULT_TOPICS)

    print(f"  Prompt         : {prompt}")
    print(f"  Generating with {MODEL_DALLE} …")
    url, revised_prompt = _generate_image(client, prompt)

    print(f"  Downloading image …")
    _download(url, IMAGE_FILE)
    print(f"  Saved           : {IMAGE_FILE}")

    print(f"  Describing with {MODEL_VISION} vision …")
    description = _describe(client, IMAGE_FILE)

    meta = {
        "prompt":              prompt,
        "revised_prompt":      revised_prompt,
        "image_path":          str(IMAGE_FILE),
        "actual_description":  description,
    }
    META_FILE.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Metadata saved  : {META_FILE}\n")
    return meta


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    prompt = input("Image prompt (Enter for random): ").strip() or None
    meta = run(client, prompt)

    print("\n── Reference Description ──────────────────────────────")
    print(meta["actual_description"])
    print("───────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
