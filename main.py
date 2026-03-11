import os
import io
import uuid
import json
import traceback
import uvicorn
import PyPDF2
from typing import Dict, Any
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# IMPORT FIX: Upgraded to AsyncGroq for Render compatibility and zero lag
from groq import AsyncGroq

# --------------------------------------------------
# Path Configuration for Render
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

# ==========================================================
# CONFIG
# ==========================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

MODEL_NAME = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

app = FastAPI(title="Mock Interviewer API - Multi Plan")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncGroq(api_key=GROQ_API_KEY)

# In-memory storage for active interview sessions
sessions = {}

VALID_PLANS = {"free", "student", "pro", "premium"}

PLAN_CONFIG = {
    "free": {
        "max_turns": 6,
        "temperature": 0.4,
        "max_words": 28,
        "role_title": "Friendly AI Interview Coach",
        "opening_style": "simple, warm, confidence-building",
    },
    "student": {
        "max_turns": 10,
        "temperature": 0.45,
        "max_words": 35,
        "role_title": "Campus Placement Interviewer",
        "opening_style": "professional, realistic, beginner-friendly",
    },
    "pro": {
        "max_turns": 16,
        "temperature": 0.5,
        "max_words": 30,
        "role_title": "Senior Technical Interviewer",
        "opening_style": "strict but fair, technical, concise",
    },
    "premium": {
        "max_turns": 20,
        "temperature": 0.55,
        "max_words": 35,
        "role_title": "Advanced Hiring Panel Interviewer",
        "opening_style": "sharp, adaptive, personalized, realistic",
    },
}

class AnswerPayload(BaseModel):
    session_id: str = Field(..., min_length=1)
    user_answer: str = ""

class RejectPayload(BaseModel):
    reason: str = ""

# ==========================================================
# HELPERS
# ==========================================================
def safe_extract_resume_text(pdf_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text.strip())
        return "\n".join([p for p in text_parts if p]).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse PDF: {str(e)}")


def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Invalid Session")
    return sessions[session_id]


async def call_llm(messages, temperature=0.4, json_mode=False):
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


def get_candidate_name_hint(resume_text: str) -> str:
    first_line = resume_text.splitlines()[0].strip() if resume_text.splitlines() else ""
    if first_line:
        return first_line[:80]
    return "Candidate"


def build_master_prompt(plan: str, resume_text: str, silence_count: int, turn_count: int) -> str:
    cfg = PLAN_CONFIG[plan]

    base_logic = f"""
You are a {cfg['role_title']}.

Candidate resume:
{resume_text}

=== CORE SYSTEM LOGIC ===
After every candidate answer, internally classify it into one of these:
1. Strong: Answered well. Action -> Acknowledge briefly and ask the next relevant question.
2. Partial: Missing details. Action -> Ask ONE follow-up on the missing piece.
3. Vague: Unclear/Generic. Action -> Ask a sharper, more specific version of the same question.
4. Wrong: Factually incorrect. Action -> Briefly correct, then ask an easier or targeted question.
5. Silent: No answer. Action -> Simplify the question while staying in interview context.

GLOBAL RULES:
- Ask exactly ONE question at a time.
- Keep every response short, natural, and easy to understand.
- Keep response under {cfg['max_words']} words whenever possible.
- Stay inside interview context.
- Do not ask multiple questions.
- Do not use long paragraphs.
- Do not become robotic.
"""

    if plan == "free":
        return base_logic + """
FREE PLAN RULES:
- Goal: build confidence and reduce fear.
- Ask only simple beginner-friendly questions.
- Focus on self-introduction, degree, current year, simple project explanation, basic HR, very basic technical questions.
- Never ask architecture, scalability, trade-offs, latency optimization, model evaluation depth, debugging depth, or advanced scenario questions.
- Friendly tone, simple language.

Good examples:
- Can you introduce yourself briefly?
- What are you studying now?
- Can you explain one project from your resume?
- What was your role in that project?
- Which technology did you use most?

Bad examples:
- Explain trade-offs between model accuracy and deployment speed.
- How would you optimize this architecture for scale?
"""

    if plan == "student":
        return base_logic + """
STUDENT PLAN RULES:
- Goal: simulate a realistic fresher interview.
- Focus on resume-based questions, project explanation, beginner technical depth, basic behavioral questions, and simple follow-ups.
- Moderate challenge only.
- Do not jump into senior-level or research-level questioning.
- If the answer is weak, ask a simpler, targeted follow-up.

Good examples:
- Can you explain your project in simple steps?
- Why did you choose this approach?
- What challenge did you face?
- What would you improve in this project?
"""

    if plan == "pro":
        return base_logic + """
PRO PLAN RULES:
- Goal: test technical depth and real project understanding.
- Focus on architecture basics, workflow, debugging, model behavior, evaluation, trade-offs, edge cases, and technical decisions.
- Be strict but fair.
- If the answer is vague, challenge it with a short, sharper follow-up.
- Keep questions direct.

Good examples:
- Walk me through the workflow from input to output.
- Why did you choose OCR before NLP?
- How would you reduce false positives?
- What metric would you use here?
"""

    return base_logic + """
PREMIUM PLAN RULES:
- Goal: simulate an advanced hiring panel.
- Focus on ownership, technical depth, system thinking, product thinking, scenario reasoning, and realistic pressure follow-ups.
- Personalize questions based on the resume.
- If the answer is weak, use a shorter, sharper follow-up.
- Do not repeat the full original question in long form.

Good examples:
- What exact part of this project did you personally own?
- Which component did you build yourself?
- How did you verify its quality?
- What would break first at scale?
"""


def build_greeting_prompt(plan: str, resume_text: str) -> str:
    cfg = PLAN_CONFIG[plan]
    name_hint = get_candidate_name_hint(resume_text)

    if plan == "free":
        return f"""
You are a {cfg['role_title']}.
Task:
- Greet the candidate naturally using their name if visible.
- Briefly introduce yourself.
- Mention one short positive detail from the resume.
- Ask exactly ONE easy opening question.
- Keep response under {cfg['max_words']} words.
- Start with a self-introduction or simple project question.
"""

    if plan == "student":
        return f"""
You are a {cfg['role_title']}.
Task:
- Greet the candidate naturally.
- Introduce yourself as a placement interviewer.
- Mention one relevant project or skill from the resume.
- Ask exactly ONE realistic fresher-level opening question.
- Keep response under {cfg['max_words']} words.
"""

    if plan == "pro":
        return f"""
You are a {cfg['role_title']}.
Task:
- Greet the candidate by name if visible.
- Briefly introduce yourself as a Senior Technical Interviewer.
- Mention one technically strong detail from the resume.
- Ask exactly ONE deep opening question.
- Keep response under {cfg['max_words']} words.
"""

    return f"""
You are a {cfg['role_title']}.
Task:
- Greet the candidate by name if visible.
- Briefly introduce yourself as part of the premium hiring panel.
- Mention one strong project detail from the resume.
- Ask exactly ONE ownership-focused or high-value opening question.
- Keep response under {cfg['max_words']} words.
"""


def build_followup_prompt(plan: str, resume_text: str, silence_count: int) -> str:
    cfg = PLAN_CONFIG[plan]

    common = f"""
You are continuing the interview for the {plan.upper()} plan.

=== SILENCE HANDLING RULES ===
The candidate has been silent for {silence_count} consecutive turns. 
If silence_count == 1: Repeat the question but make it shorter.
If silence_count == 2: Simplify the question significantly.
If silence_count == 3: Switch to an easier, but related question.
If silence_count >= 4: Move to a completely different, easier category.
NEVER repeatedly say "Don't worry".

Rules:
- Ask exactly ONE question.
- Keep response under {cfg['max_words']} words.
- Keep it natural and interview-like.
- Do not write long paragraphs.
- Do not ask multiple questions.
"""

    if plan == "free":
        return common + """
FREE PLAN BEHAVIOR:
- Stay simple and beginner-friendly.
- Fallback order: Tell me about yourself -> Degree -> Projects -> Tech used.
Avoid trade-offs, architecture, optimization, scalability.
"""

    if plan == "student":
        return common + """
STUDENT PLAN BEHAVIOR:
- Keep it realistic for freshers.
- Ask about project flow, technology choice, simple challenges.
- Moderate pressure only.
"""

    if plan == "pro":
        return common + """
PRO PLAN BEHAVIOR:
- Keep it technical, short, and challenging.
- Focus on workflow, technical decisions, metrics, debugging, edge cases.
- Challenge vague answers briefly.
"""

    return common + """
PREMIUM PLAN BEHAVIOR:
- Keep it sharp, personalized, and realistic.
- Focus on ownership, validation, technical depth, product/system thinking.
- Challenge vague answers with a shorter follow-up. Do not repeat the full previous question.
"""


def build_evaluation_prompt(plan: str, resume_text: str, history: list) -> str:
    if plan == "free":
        feedback_style = """
Keep feedback simple, short, supportive, and beginner-friendly.
Use:
- ✅ Strengths
- ❌ Weak Areas
- 📈 Next Steps
"""
    elif plan == "student":
        feedback_style = """
Keep feedback practical and placement-oriented.
Use:
- ✅ Strengths
- ❌ Mistakes
- 📈 Areas to Improve
"""
    elif plan == "pro":
        feedback_style = """
Keep feedback technical and exact.
Use:
- ✅ Technical Strengths
- ❌ Technical Mistakes
- 📈 Advanced Topics to Improve
"""
    else:
        feedback_style = """
Keep feedback deep, realistic, and personalized.
Use:
- ✅ Strongest Signals
- ❌ Gaps / Weaknesses
- 📈 Priority Improvements
- 🎯 Interview Readiness
"""

    return f"""
The mock interview session has ended for the {plan.upper()} plan.

Resume Context:
{resume_text}

Interview Transcript:
{json.dumps(history, ensure_ascii=False)}

Scoring Rules:
1. If the candidate mostly stayed silent, did not answer, or timeout markers dominate the transcript, give marks = 0.
2. Score must be an integer from 0 to 100.
3. Return valid JSON only.
4. Do not give sympathy marks.

Return exactly:
{{
  "marks": <integer>,
  "recommendations": "<HTML feedback>"
}}

{feedback_style}
Use <br><br> between sections.
"""


async def evaluate_interview(session_id: str):
    session = get_session(session_id)
    plan = session["plan"]

    prompt = build_evaluation_prompt(plan, session["resume"], session["history"])

    try:
        response = await call_llm(
            [{"role": "system", "content": prompt}],
            temperature=0.2,
            json_mode=True
        )
        data = json.loads(response)

        marks = data.get("marks", 0)
        recommendations = data.get("recommendations", "Interview completed.")

        try:
            marks = int(marks)
        except Exception:
            marks = 0

        marks = max(0, min(100, marks))

        return {
            "action": "finish",
            "marks": marks,
            "recommendations": recommendations,
            "plan": plan
        }

    except Exception:
        print("Evaluation Error:", traceback.format_exc())
        return {
            "action": "finish",
            "marks": 0,
            "recommendations": "Interview completed.<br><br>There was an error generating the final report.",
            "plan": plan
        }


async def get_ai_response(session_id, user_text):
    session = get_session(session_id)
    plan = session["plan"]
    cfg = PLAN_CONFIG[plan]

    user_text = (user_text or "").strip()
    is_time_up = "[SYSTEM_DURATION_EXPIRED]" in user_text
    is_timeout = "[NO_ANSWER_TIMEOUT]" in user_text

    # Save user response and handle Silence Count Logic
    if user_text and not is_time_up and not is_timeout:
        session["history"].append({"role": "user", "content": user_text})
        session["silence_count"] = 0
    elif is_timeout:
        session["history"].append({"role": "user", "content": "[Candidate remained silent or did not answer the question.]"})
        session["silence_count"] += 1

    session["turn_count"] += 1

    # Finish when duration ends or max turns reached
    if is_time_up or session["turn_count"] >= cfg["max_turns"]:
        return await evaluate_interview(session_id)

    # First greeting
    if session["turn_count"] == 1:
        greeting_prompt = build_greeting_prompt(plan, session["resume"])
        system_prompt = build_master_prompt(plan, session["resume"], session["silence_count"], session["turn_count"])

        ai_msg = await call_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": greeting_prompt},
            ],
            temperature=cfg["temperature"]
        )
        session["history"].append({"role": "assistant", "content": ai_msg})
        return {
            "action": "continue",
            "text": ai_msg,
            "plan": plan,
            "turn_count": session["turn_count"],
            "remaining_turns": max(cfg["max_turns"] - session["turn_count"], 0)
        }

    # Follow-up
    followup_prompt = build_followup_prompt(plan, session["resume"], session["silence_count"])
    system_prompt = build_master_prompt(plan, session["resume"], session["silence_count"], session["turn_count"])

    ai_msg = await call_llm(
        [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": followup_prompt},
        ] + session["history"],
        temperature=cfg["temperature"]
    )

    session["history"].append({"role": "assistant", "content": ai_msg})

    return {
        "action": "continue",
        "text": ai_msg,
        "plan": plan,
        "turn_count": session["turn_count"],
        "remaining_turns": max(cfg["max_turns"] - session["turn_count"], 0)
    }


# ==========================================================
# API ENDPOINTS
# ==========================================================
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    try:
        with open(TEMPLATES_DIR / "index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Error: templates/index.html not found</h1>", status_code=404)


@app.get("/interview/{session_id}", response_class=HTMLResponse)
async def serve_interview(session_id: str):
    if session_id not in sessions:
        return HTMLResponse("<h1>Session Expired or Invalid</h1>", status_code=404)
    try:
        with open(TEMPLATES_DIR / "interview.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Error: templates/interview.html not found</h1>", status_code=404)


@app.get("/plans")
async def get_plans():
    return {
        "plans": {
            key: {
                "max_turns": value["max_turns"],
                "role_title": value["role_title"]
            }
            for key, value in PLAN_CONFIG.items()
        }
    }


@app.post("/setup")
async def setup_interview(
    request: Request,
    resume_file: UploadFile = File(...),
    plan: str = Form("free")
):
    plan = (plan or "free").strip().lower()

    if plan not in VALID_PLANS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid plan. Use one of: {', '.join(sorted(VALID_PLANS))}"
        )

    if not resume_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")

    pdf_bytes = await resume_file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded resume is empty.")

    resume_text = safe_extract_resume_text(pdf_bytes)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract text from the resume.")

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "resume": resume_text,
        "history": [],
        "turn_count": 0,
        "silence_count": 0,
        "rejection_reason": None,
        "plan": plan
    }

    base_url = str(request.base_url).rstrip("/")
    return {
        "interview_link": f"{base_url}/interview/{session_id}",
        "session_id": session_id,
        "plan": plan,
        "max_turns": PLAN_CONFIG[plan]["max_turns"]
    }


@app.post("/next_question")
async def next_question(payload: AnswerPayload):
    if payload.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Invalid Session")

    try:
        response_data = await get_ai_response(payload.session_id, payload.user_answer)
        return JSONResponse(content=response_data)
    except Exception:
        print(f"Error processing AI response: {traceback.format_exc()}")
        return JSONResponse(
            content={
                "action": "continue",
                "text": "I lost connection for a second. Could you please repeat your answer?"
            }
        )


@app.post("/terminate_interview/{session_id}")
async def terminate(session_id: str, payload: RejectPayload):
    if session_id in sessions:
        sessions[session_id]["rejection_reason"] = payload.reason
    return {"status": "recorded"}


@app.post("/finish/{session_id}")
async def finish_interview_endpoint(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Invalid Session")
    return JSONResponse(content=await evaluate_interview(session_id))


# Server Execution for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Mock Interviewer Server Online on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
