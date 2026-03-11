import os
import io
import json
import uuid
import traceback
from typing import Dict, Any
from pathlib import Path

import PyPDF2
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# IMPORT FIX: Upgraded to AsyncGroq to eliminate server blocking and fix lag
from groq import AsyncGroq
from pydantic import BaseModel, Field

# --------------------------------------------------
# Path Configuration for Vercel
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"

# --------------------------------------------------
# Configuration
# --------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is missing.")

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

app = FastAPI(title="AI Mock Interviewer SaaS API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# INIT FIX: Use Async client
client = AsyncGroq(api_key=GROQ_API_KEY)

# In-memory session store
sessions: Dict[str, Dict[str, Any]] = {}

VALID_PLANS = {"free", "student", "pro", "premium"}

# Basic limits and parameters per plan
PLAN_CONFIG = {
    "free": {"max_turns": 6, "temperature": 0.4},
    "student": {"max_turns": 10, "temperature": 0.45},
    "pro": {"max_turns": 16, "temperature": 0.5},
    "premium": {"max_turns": 20, "temperature": 0.55},
}


# --------------------------------------------------
# Request Models
# --------------------------------------------------
class AnswerPayload(BaseModel):
    session_id: str = Field(..., min_length=1)
    user_answer: str = Field(default="")

class RejectPayload(BaseModel):
    reason: str = Field(default="User terminated interview")


# --------------------------------------------------
# Fine-Tuned Prompt Builders
# --------------------------------------------------
def get_conversational_prompt(plan: str, resume_text: str, is_greeting: bool) -> str:
    """Returns the unified, highly-tuned conversational system prompt."""
    
    master_prompt = f"""
You are an AI Mock Interviewer. Your job is to conduct realistic interview practice based on the candidate’s resume and the selected plan: Free, Student, Pro, or Premium.

Your core goal is to make the interview experience feel correct for the selected plan, while keeping the conversation clear, natural, structured, and useful.

CURRENT ACTIVE PLAN: {plan.upper()}

Candidate Resume Context:
{resume_text}

GLOBAL RULES FOR ALL PLANS:
1. Ask exactly ONE question at a time.
2. Never ask multiple questions in one response.
3. Keep questions short, clear, and easy to understand.
4. Do not give long paragraphs unless it is the final evaluation.
5. Adapt the difficulty strictly based on the selected plan.
6. Avoid repeating the same question in a long form.
7. If the candidate gives a weak or vague answer, ask a shorter and sharper follow-up.
8. If the candidate is silent, simplify the question while staying in interview context.
9. Do not turn the interview into casual chat.
10. Do not become robotic, overly emotional, or overly polite.
11. Do not praise too much. Use only brief, natural acknowledgment.
12. Keep the conversation realistic and interview-like.
13. Use the candidate’s resume to personalize questions whenever useful.
14. Maintain the correct role and tone for the selected plan.
15. Every question should have a clear purpose: introduction, validation, technical check, project depth, scenario, or ownership.

PLAN BEHAVIOR RULES (Strictly enforce the {plan.upper()} behavior below):

FREE PLAN:
- Role: Friendly AI Interview Coach
- Goal: Build confidence and reduce fear
- Tone: Friendly, simple, encouraging
- Difficulty: Beginner only
- Focus: Self-introduction, resume basics, simple project explanation, very basic HR and technical questions
- Never ask architecture, scalability, trade-offs, optimization, or deep technical drill-down
- If the candidate is silent, simplify gradually but remain interview-related
- Keep responses under 25 words whenever possible

STUDENT PLAN:
- Role: Campus Placement Interviewer
- Goal: Simulate a realistic fresher interview
- Tone: Professional, practical, beginner-friendly
- Difficulty: Moderate
- Focus: Resume questions, project explanation, beginner technical depth, behavioral questions, simple follow-ups
- Do not jump into advanced research-level or senior-level questions
- Keep responses under 35 words whenever possible

PRO PLAN:
- Role: Senior Technical Interviewer
- Goal: Test technical depth and project understanding
- Tone: Strict but fair, technical, direct
- Difficulty: Deep technical
- Focus: Workflow, architecture basics, debugging, trade-offs, evaluation, edge cases, technical decisions
- If the answer is weak, challenge it with a short follow-up
- Keep responses under 30 words whenever possible

PREMIUM PLAN:
- Role: Advanced Hiring Panel
- Goal: Simulate a realistic, high-value hiring round
- Tone: Sharp, adaptive, realistic, professional
- Difficulty: Deep and personalized
- Focus: Ownership, technical depth, system thinking, product thinking, pressure questioning, scenario reasoning
- Follow-ups must be concise, sharp, and non-repetitive
- Keep responses under 35 words whenever possible

GREETING RULES:
- Greet the candidate by name if available in the resume
- Briefly introduce yourself according to the selected plan
- Mention one short relevant detail from the resume
- Ask the first question quickly
- Do not make the greeting too long

FOLLOW-UP RULES:
- If the previous answer is good, briefly acknowledge it and move to the next question
- If the previous answer is weak, ask a shorter and more precise follow-up
- Do not repeat the full previous question
- Do not over-explain what you want
- Keep pressure proportional to the selected plan

SILENCE / NO ANSWER RULES:
- First silence: gently repeat or simplify the question
- Second silence: make the question easier while staying interview-related
- Third silence: move to a simpler but still relevant question
- Do not keep saying "don't worry" repeatedly
- Do not switch into random casual questions unrelated to interview preparation

BAD BEHAVIOR TO AVOID:
- Asking overly advanced questions in Free plan
- Being too casual in Premium plan
- Repeating the same question with too many words
- Giving long motivational speeches
- Asking confusing or multi-part questions
- Making the candidate feel lost through jargon-heavy phrasing
- Dropping the interview context during fallback

GOOD BEHAVIOR TO FOLLOW:
- Be clear
- Be structured
- Be plan-appropriate
- Be realistic
- Be concise
- Be adaptive
- Keep every question purposeful

Always act according to the selected plan and maintain consistent interview quality from start to finish.
"""

    if is_greeting:
        action_directive = "\n\nCURRENT TASK: This is the VERY FIRST message. You MUST execute the 'Greeting rules' perfectly based on the active plan. Ask exactly ONE starting question."
    else:
        action_directive = "\n\nCURRENT TASK: Acknowledge the candidate's last answer and ask the NEXT interview question. DO NOT repeat the greeting. Ask exactly ONE question."

    return master_prompt + action_directive


def get_evaluation_prompt(plan: str, resume_text: str, history: list) -> str:
    """Returns the strict JSON evaluation prompt based on the user's plan."""
    
    transcript_str = json.dumps(history, ensure_ascii=False)

    base_rules = f"""
Resume Context:
{resume_text}

Interview Transcript:
{transcript_str}
"""

    if plan == "free":
        return f"""
The mock interview session has ended for a Free Plan user.
{base_rules}
Task: Evaluate the student in a simple, encouraging, beginner-friendly way.
Rules:
- If the student mostly stayed silent or did not answer, give marks = 0.
- Return valid JSON only.
- Score must be an integer from 0 to 100.

Return exactly:
{{
  "marks": <integer>,
  "recommendations": "<HTML feedback>"
}}

Feedback format for "recommendations":
- ✅ Strengths
- ❌ Weak Areas
- 📈 Simple Next Steps
- 🔒 <strong style='color:#2563eb;'>Upgrade to Student or Pro</strong> to see your exact technical mistakes and get question-by-question corrections!

Keep the feedback easy to understand, supportive, and short.
"""
    elif plan == "student":
        return f"""
The mock interview session has ended for a Student Plan user.
{base_rules}
Task: Evaluate the candidate like a realistic campus placement interviewer.
Rules:
- If the student mostly stayed silent or did not answer, give marks = 0.
- Return valid JSON only.
- Score must be an integer from 0 to 100.

Return exactly:
{{
  "marks": <integer>,
  "recommendations": "<HTML feedback>"
}}

Feedback format for "recommendations":
- ✅ Strengths
- ❌ Mistakes / weak answers (Be specific about one technical error)
- 📈 Areas to improve before real interviews
- 🔒 <strong style='color:#2563eb;'>Upgrade to Pro</strong> to unlock deep system-design feedback and weak-area analysis!

Keep the feedback practical, clear, and placement-oriented.
"""
    elif plan == "pro":
        return f"""
The technical interview session has ended for a Pro Plan user.
{base_rules}
Task: Strictly evaluate the candidate’s technical depth, clarity, reasoning, and project understanding.
Rules:
- If the candidate mostly stayed silent or did not answer, give marks = 0.
- Return valid JSON only.
- Score must be an integer from 0 to 100.
- Do not give sympathy marks.

Return exactly:
{{
  "marks": <integer>,
  "recommendations": "<HTML feedback>"
}}

Feedback format for "recommendations":
- ✅ Technical Strengths
- ❌ Technical Mistakes (Point out exactly what they got wrong in their answers)
- 📈 Advanced Topics to Improve

Be exact, realistic, and technically meaningful.
"""
    else: # premium
        return f"""
The advanced interview session has ended for a Premium Plan user.
{base_rules}
Task: Evaluate the candidate like a hiring panel assessing technical skill, product thinking, system reasoning, ownership, and communication quality.
Rules:
- If the candidate mostly stayed silent or did not answer, give marks = 0.
- Return valid JSON only.
- Score must be an integer from 0 to 100.
- Be realistic and strict.

Return exactly:
{{
  "marks": <integer>,
  "recommendations": "<HTML feedback>"
}}

Feedback format for "recommendations":
- ✅ Strongest Signals
- ❌ Gaps / Weaknesses
- 📈 Priority Improvements
- 🎯 Interview Readiness Level

Make the feedback deep, personalized, and realistic.
"""


# --------------------------------------------------
# Execution Utilities (NOW ASYNC)
# --------------------------------------------------
async def call_groq(messages, *, temperature: float = 0.4, json_mode: bool = False) -> str:
    kwargs = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    # FIX: Awaiting the async groq client ensures the server isn't blocked
    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join([p for p in pages if p]).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")


def ensure_session(session_id: str) -> Dict[str, Any]:
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Invalid or expired session.")
    return session


def safe_json_loads(raw_text: str) -> Dict[str, Any]:
    try:
        return json.loads(raw_text)
    except Exception:
        return {}


# --------------------------------------------------
# Core Interview Engine (NOW ASYNC)
# --------------------------------------------------
async def finish_interview(session_id: str) -> Dict[str, Any]:
    session = ensure_session(session_id)
    plan = session["plan"]
    
    prompt = get_evaluation_prompt(plan, session["resume"], session["history"])

    try:
        # Await the groq call
        raw = await call_groq(
            [{"role": "system", "content": prompt}],
            temperature=0.2,
            json_mode=True
        )
        data = safe_json_loads(raw)

        marks = data.get("marks", 0)
        recommendations = data.get("recommendations", "Interview completed.<br><br>Feedback generated.")

        try:
            marks = int(marks)
        except Exception:
            marks = 0

        marks = max(0, min(100, marks))

        return {
            "action": "finish",
            "marks": marks,
            "recommendations": recommendations,
            "plan": plan,
        }
    except Exception:
        print("Finish Error:\n", traceback.format_exc())
        return {
            "action": "finish",
            "marks": 0,
            "recommendations": "Interview completed.<br><br>There was an error generating the final report.",
            "plan": plan,
        }


async def get_ai_response(session_id: str, user_text: str) -> Dict[str, Any]:
    session = ensure_session(session_id)
    plan = session["plan"]
    cfg = PLAN_CONFIG[plan]

    user_text = (user_text or "").strip()
    is_time_up = "[SYSTEM_DURATION_EXPIRED]" in user_text
    is_timeout = "[NO_ANSWER_TIMEOUT]" in user_text

    # Store candidate response
    if user_text and not is_time_up and not is_timeout:
        session["history"].append({"role": "user", "content": user_text})
    elif is_timeout:
        session["history"].append({
            "role": "user",
            "content": "[Candidate remained silent or did not answer the question.]"
        })

    session["turn_count"] += 1

    # Finish conditions
    if is_time_up or session["turn_count"] >= cfg["max_turns"]:
        return await finish_interview(session_id)

    # Determine if this is the first greeting or a follow-up
    is_greeting = (session["turn_count"] == 1)
    system_prompt = get_conversational_prompt(plan, session["resume"], is_greeting)
    
    messages = [{"role": "system", "content": system_prompt}] + session["history"]

    # Await the groq call to prevent lag
    ai_msg = await call_groq(messages, temperature=cfg["temperature"])
    session["history"].append({"role": "assistant", "content": ai_msg})

    return {
        "action": "continue",
        "text": ai_msg,
        "plan": plan,
        "turn_count": session["turn_count"],
        "remaining_turns": max(cfg["max_turns"] - session["turn_count"], 0),
    }


# --------------------------------------------------
# API Endpoints
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    try:
        with open(TEMPLATES_DIR / "index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Error: index.html not found. Ensure it is in the 'templates' folder.</h1>", status_code=404)


@app.get("/interview/{session_id}", response_class=HTMLResponse)
async def serve_interview(session_id: str):
    if session_id not in sessions:
        return HTMLResponse("<h1>Session expired or invalid.</h1>", status_code=404)

    try:
        with open(TEMPLATES_DIR / "interview.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Error: interview.html not found. Ensure it is in the 'templates' folder.</h1>", status_code=404)


@app.post("/setup")
async def setup_interview(
    request: Request,
    resume_file: UploadFile = File(...),
    plan: str = Form("free")
):
    plan = plan.strip().lower()

    if plan not in VALID_PLANS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid plan. Use one of: {', '.join(sorted(VALID_PLANS))}"
        )

    if not resume_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")

    pdf_bytes = await resume_file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    resume_text = extract_pdf_text(pdf_bytes)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract text from the resume PDF.")

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "resume": resume_text,
        "history": [],
        "turn_count": 0,
        "plan": plan,
        "rejection_reason": None,
        "created_from": request.client.host if request.client else None,
    }

    base_url = str(request.base_url).rstrip("/")
    return {
        "session_id": session_id,
        "plan": plan,
        "interview_link": f"{base_url}/interview/{session_id}",
        "max_turns": PLAN_CONFIG[plan]["max_turns"],
    }


@app.post("/next_question")
async def next_question(payload: AnswerPayload):
    ensure_session(payload.session_id)

    try:
        # Await the response generator
        response_data = await get_ai_response(payload.session_id, payload.user_answer)
        return JSONResponse(content=response_data)
    except HTTPException:
        raise
    except Exception:
        print("AI Response Error:\n", traceback.format_exc())
        return JSONResponse(
            content={
                "action": "continue",
                "text": "I lost connection for a moment. Please repeat your answer.",
            },
            status_code=200
        )


@app.post("/finish/{session_id}")
async def finish_now(session_id: str):
    ensure_session(session_id)
    return JSONResponse(content=await finish_interview(session_id))


@app.post("/terminate_interview/{session_id}")
async def terminate(session_id: str, payload: RejectPayload):
    session = ensure_session(session_id)
    session["rejection_reason"] = payload.reason
    return {"status": "recorded", "session_id": session_id}


# --------------------------------------------------
# Local Run
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"AI Mock Interviewer API running on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
