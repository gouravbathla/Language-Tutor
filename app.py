
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import OpenAI

# Import your existing functions
from agents.grammar.grammar_mcq_agent import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    parse_mcqs,
)

from agents.grammar.grammar_evaluator_agent import (
    build_evaluation_prompt,
    overall,
    grade,
    EVALUATOR_SYSTEM_PROMPT,
)

# ---------------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------------

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set.")

client = OpenAI(api_key=api_key)

# ---------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------

app = FastAPI(
    title="Grammar Quiz API",
    description="Generate and evaluate English grammar MCQs using OpenAI",
    version="1.0.0"
)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

QUESTIONS_FILE = DATA_DIR / "grammar_questions.json"
EVALUATION_FILE = DATA_DIR / "grammar_evaluation.json"

MODEL = "gpt-4o-mini"

# ---------------------------------------------------------
# Request Models
# ---------------------------------------------------------

class GenerateRequest(BaseModel):
    total_questions: Optional[int] = 20


class EvaluateRequest(BaseModel):
    questions: list


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Grammar Quiz API is running"
    }


# ---------------------------------------------------------
# Generate Questions
# ---------------------------------------------------------

@app.post("/generate")
def generate_questions(req: GenerateRequest):

    dynamic_prompt = USER_PROMPT.replace(
        "Generate exactly 20 multiple-choice questions",
        f"Generate exactly {req.total_questions} multiple-choice questions"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": dynamic_prompt
                },
            ],
        )

        raw_text = response.choices[0].message.content

        questions = parse_mcqs(raw_text)

        payload = {
            "model": MODEL,
            "topic": "English Grammar",
            "total": len(questions),
            "questions": questions,
        }

        QUESTIONS_FILE.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        return {
            "success": True,
            "total_questions": len(questions),
            "questions": questions,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# Evaluate Questions
# ---------------------------------------------------------

@app.post("/evaluate")
def evaluate_questions(req: EvaluateRequest):

    try:

        user_content = (
            "Please evaluate the following grammar MCQs and return "
            "a JSON array as specified:\n\n"
            + build_evaluation_prompt(req.questions)
        )

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": EVALUATOR_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content
                },
            ],
        )

        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        evaluations = json.loads(raw)

        for ev in evaluations:
            ev["overall"] = overall(ev)
            ev["grade"] = grade(ev["overall"])

        report = {
            "total_questions": len(evaluations),
            "evaluations": evaluations,
        }

        EVALUATION_FILE.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        return {
            "success": True,
            "report": report,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# Get Latest Generated Questions
# ---------------------------------------------------------

@app.get("/questions")
def get_questions():

    if not QUESTIONS_FILE.exists():
        raise HTTPException(status_code=404, detail="No questions found")

    data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    return data


# ---------------------------------------------------------
# Get Latest Evaluation
# ---------------------------------------------------------

@app.get("/evaluation")
def get_evaluation():

    if not EVALUATION_FILE.exists():
        raise HTTPException(status_code=404, detail="No evaluation found")

    data = json.loads(EVALUATION_FILE.read_text(encoding="utf-8"))
    return data

