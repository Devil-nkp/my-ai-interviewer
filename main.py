import os
import io
import uuid
import json
import traceback
import asyncio
import logging
import sqlite3
import re
import secrets
import time
from typing import Dict, Any, Optional
from pathlib import Path
from collections import defaultdict

import uvicorn
import PyPDF2
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form, Depends, Header, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# NEW IMPORT: FPDF for generating the downloadable interview report
from fpdf import FPDF

# IMPORT FIX: Upgraded to AsyncGroq for Render compatibility and zero lag
from groq import AsyncGroq

# ==========================================================
# ENTERPRISE LOGGING SETUP
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MockAI")

# --------------------------------------------------
# Path Configuration for Render
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
DB_PATH = BASE_DIR / "production_sessions.db"

# ==========================================================
# RATE LIMITING ENGINE (In-Memory Token Bucket)
# ==========================================================
class RateLimiter:
    def __init__(self, requests: int, window: int):
        self.requests = requests
        self.window = window
        self.ips = defaultdict(list)

    def check_rate_limit(self, request: Request):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        self.ips[client_ip] = [req_time for req_time in self.ips[client_ip] if now - req_time < self.window]
        
        if len(self.ips[client_ip]) >= self.requests:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
        
        self.ips[client_ip].append(now)

global_rate_limiter = RateLimiter(requests=60, window=60)

async def rate_limit_dependency(request: Request):
    global_rate_limiter.check_rate_limit(request)


# ==========================================================
# PERSISTENT DATABASE & STATE SETUP
# ==========================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (session_id TEXT PRIMARY KEY, access_token TEXT, state TEXT, created_at REAL, data TEXT)''')
    conn.commit()
    conn.close()

init_db()
session_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

def save_session(session_id: str, access_token: str, session_data: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO sessions (session_id, access_token, state, created_at, data) VALUES (?, ?, ?, ?, ?)",
        (session_id, access_token, session_data.get("state", "ACTIVE"), session_data.get("created_at", time.time()), json.dumps(session_data))
    )
    conn.commit()
    conn.close()

def get_session(session_id: str, token: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT access_token, data FROM sessions WHERE session_id = ?", (session_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Invalid or Expired Session")
    
    db_token, data_json = row
    if not secrets.compare_digest(db_token, token):
        raise HTTPException(status_code=403, detail="Unauthorized access token.")
        
    return json.loads(data_json)

def cleanup_stale_sessions():
    """TTL Cleanup: Deletes sessions older than 24 hours."""
    cutoff_time = time.time() - (24 * 3600)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM sessions WHERE created_at < ?", (cutoff_time,))
    conn.commit()
    conn.close()

# ==========================================================
# CONFIG
# ==========================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

MODEL_NAME = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="Mock Interviewer API - Multi Plan")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncGroq(api_key=GROQ_API_KEY)

VALID_PLANS = {"free", "student", "pro", "premium"}
MAX_HISTORY_TURNS = 8

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
        raw_text = "\n".join([p for p in text_parts if p]).strip()
        return raw_text[:5000] # Hard cap to prevent prompt injection and token explosion
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse PDF: {str(e)}")


async def call_llm(messages, temperature=0.4, json_mode=False, retries=3):
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(retries):
        try:
            response = await asyncio.wait_for(client.chat.completions.create(**kwargs), timeout=15.0)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM Error (Attempt {attempt+1}/{retries}): {e}")
        if attempt == retries - 1:
            raise HTTPException(status_code=503, detail="AI Provider is currently unavailable.")
        await asyncio.sleep(1.5)


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
- HIDDEN CLASSIFICATIONS: NEVER output your internal classification tags (like 'CLASSIFICATION: STRONG' or 'VAGUE:') to the user. Just speak the natural response.
- NORMALIZING SPEECH: If a candidate says 'llmp3' assume 'Llama 3'. If they say 'group model' assume 'Groq model'. Do not penalize speech-to-text glitches.
- HANDLING "DON'T KNOW": If the candidate says they forgot, don't recollect, or don't know: DO NOT keep asking about it. Briefly provide the correct answer or context, then move on to a completely new question.
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


def build_evaluation_prompt(plan: str, history: list) -> str:
    eval_history = json.dumps(history[-16:], ensure_ascii=False)
    
    return f"""
The mock interview session has ended for the {plan.upper()} plan.
Interview Transcript (Recent): {eval_history}

Scoring Rules:
1. Ignore minor speech-to-text typos. Score based on semantic understanding.
2. Provide a premium coaching breakdown based purely on fact. Do NOT return HTML. Return pure structured JSON.

Return EXACTLY this JSON structure:
{{
  "marks": <integer 0-100>,
  "strengths": ["string", "string"],
  "weaknesses": ["string", "string"],
  "ideal_answer_coaching": "String explaining how they should have answered the hardest question",
  "next_steps": ["string", "string"]
}}
"""


# ==========================================================
# PDF CLEANER HELPER
# ==========================================================
def clean_text_for_pdf(text: str) -> str:
    """Removes HTML tags and handles emoji characters so FPDF doesn't crash."""
    text = str(text)
    # Replace HTML line breaks with real line breaks
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    # Remove all other HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Translate Emojis to text equivalents for standard PDF fonts
    text = text.replace("✅", "[Strength] ")
    text = text.replace("❌", "[Weakness] ")
    text = text.replace("📈", "[Next Steps] ")
    text = text.replace("🎯", "[Target] ")
    text = text.replace("🔒", "[Locked] ")
    text = text.replace("💡", "[Coaching] ")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    
    # Safely encode to latin-1 to avoid PDF build errors
    return text.encode('latin-1', 'ignore').decode('latin-1')


def format_recommendations_html(eval_data: dict) -> str:
    """Safely builds HTML from the structured JSON to prevent LLM XSS attacks."""
    html = ""
    if eval_data.get("strengths"):
        html += "<strong>✅ Strengths:</strong><ul>" + "".join([f"<li>{clean_text_for_pdf(s)}</li>" for s in eval_data["strengths"]]) + "</ul><br>"
    if eval_data.get("weaknesses"):
        html += "<strong>❌ Areas to Improve:</strong><ul>" + "".join([f"<li>{clean_text_for_pdf(w)}</li>" for w in eval_data["weaknesses"]]) + "</ul><br>"
    if eval_data.get("ideal_answer_coaching"):
        html += f"<strong>💡 Coaching (Ideal Answer):</strong><br>{clean_text_for_pdf(eval_data['ideal_answer_coaching'])}<br><br>"
    if eval_data.get("next_steps"):
        html += "<strong>📈 Next Steps:</strong><ul>" + "".join([f"<li>{clean_text_for_pdf(n)}</li>" for n in eval_data["next_steps"]]) + "</ul>"
    return html


async def evaluate_interview(session_id: str, access_token: str):
    session = get_session(session_id, access_token)
    plan = session["plan"]

    if session["state"] == "FINISHED":
        return {"action": "finish", "marks": session.get("marks", 0), "recommendations": session.get("recommendations_html", ""), "plan": plan}

    prompt = build_evaluation_prompt(plan, session["history"])

    try:
        response = await call_llm(
            [{"role": "system", "content": prompt}],
            temperature=0.2,
            json_mode=True
        )
        data = json.loads(response)

        marks = max(0, min(100, int(data.get("marks", 0))))
        safe_html = format_recommendations_html(data)

        # IMPORTANT: Save data to session so the PDF generator can access it later
        session["marks"] = marks
        session["eval_json"] = data
        session["recommendations_html"] = safe_html
        session["state"] = "FINISHED"
        save_session(session_id, access_token, session)

        return {
            "action": "finish",
            "marks": marks,
            "recommendations": safe_html,
            "plan": plan
        }

    except Exception:
        print("Evaluation Error:", traceback.format_exc())
        session["state"] = "FINISHED"
        save_session(session_id, access_token, session)
        return {
            "action": "finish",
            "marks": 0,
            "recommendations": "Interview completed.<br><br>There was an error generating the detailed report.",
            "plan": plan
        }


async def get_ai_response(session_id: str, access_token: str, user_text: str):
    async with session_locks[session_id]:
        session = get_session(session_id, access_token)
        
        if session["state"] != "ACTIVE":
            if session["state"] == "FINISHED": return await evaluate_interview(session_id, access_token)
            raise HTTPException(status_code=400, detail="This interview session has been terminated.")

        plan = session["plan"]
        cfg = PLAN_CONFIG[plan]
        user_text = (user_text or "").strip()
        
        # === EXIT INTERVIEW DETECTION ===
        exit_phrases = [
            "exit interview", "end interview", "end the interview", 
            "stop the interview", "we can end it", "we can end up", 
            "that's it for now", "wrap it up"
        ]
        if any(phrase in user_text.lower() for phrase in exit_phrases) or "[USER_REQUESTED_END]" in user_text:
            session["history"].append({"role": "user", "content": user_text})
            save_session(session_id, access_token, session)
            return await evaluate_interview(session_id, access_token)
        # =====================================

        is_time_up = "[SYSTEM_DURATION_EXPIRED]" in user_text
        is_timeout = "[NO_ANSWER_TIMEOUT]" in user_text

        # Save user response and handle Silence Count Logic
        if user_text and not is_time_up and not is_timeout:
            session["history"].append({"role": "user", "content": user_text})
            session["silence_count"] = 0
        elif is_timeout:
            session["events"].append("timeout")
            session["silence_count"] += 1

        session["turn_count"] += 1
        save_session(session_id, access_token, session)

        # Finish when duration ends or max turns reached
        if is_time_up or session["turn_count"] >= cfg["max_turns"]:
            return await evaluate_interview(session_id, access_token)

        recent_history = session["history"][-(MAX_HISTORY_TURNS * 2):]

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
            
            # Clean accidental classification leakage
            ai_msg = re.sub(r'^(CLASSIFICATION:?\s*\w+\s*\|?\s*)', '', ai_msg, flags=re.IGNORECASE).strip()
            ai_msg = re.sub(r'^(\**\b(Strong|Partial|Vague|Wrong)\b\**:\s*)', '', ai_msg, flags=re.IGNORECASE).strip()

            session["history"].append({"role": "assistant", "content": ai_msg})
            save_session(session_id, access_token, session)
            
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
            ] + recent_history,
            temperature=cfg["temperature"]
        )
        
        if "[USER_REQUESTED_END]" in ai_msg:
            return await evaluate_interview(session_id, access_token)

        # Clean accidental classification leakage
        ai_msg = re.sub(r'^(CLASSIFICATION:?\s*\w+\s*\|?\s*)', '', ai_msg, flags=re.IGNORECASE).strip()
        ai_msg = re.sub(r'^(\**\b(Strong|Partial|Vague|Wrong)\b\**:\s*)', '', ai_msg, flags=re.IGNORECASE).strip()

        session["history"].append({"role": "assistant", "content": ai_msg})
        save_session(session_id, access_token, session)

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


@app.post("/setup", dependencies=[Depends(rate_limit_dependency)])
async def setup_interview(
    request: Request,
    background_tasks: BackgroundTasks,
    resume_file: UploadFile = File(...),
    plan: str = Form("free")
):
    background_tasks.add_task(cleanup_stale_sessions)
    
    plan = (plan or "free").strip().lower()

    if plan not in VALID_PLANS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid plan. Use one of: {', '.join(sorted(VALID_PLANS))}"
        )

    file_bytes = await resume_file.read()
    if len(file_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 5MB.")
    if not file_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be a valid PDF format.")

    resume_text = safe_extract_resume_text(file_bytes)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract text from the resume.")

    session_id = str(uuid.uuid4())
    access_token = secrets.token_urlsafe(32)
    
    session_data = {
        "resume": resume_text,
        "history": [],
        "events": [],
        "turn_count": 0,
        "silence_count": 0,
        "rejection_reason": None,
        "plan": plan,
        "marks": 0,
        "eval_json": {},
        "recommendations_html": "",
        "state": "ACTIVE",
        "created_at": time.time()
    }
    
    save_session(session_id, access_token, session_data)
    logger.info(f"Session {session_id} created securely for plan {plan}.")

    base_url = str(request.base_url).rstrip("/")
    return {
        "interview_link": f"{base_url}/interview/{session_id}",
        "session_id": session_id,
        "access_token": access_token,
        "plan": plan,
        "max_turns": PLAN_CONFIG[plan]["max_turns"]
    }


@app.post("/next_question", dependencies=[Depends(rate_limit_dependency)])
async def next_question(payload: AnswerPayload, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.split("Bearer ")[1]
    
    try:
        response_data = await get_ai_response(payload.session_id, token, payload.user_answer)
        return JSONResponse(content=response_data)
    except HTTPException:
        raise
    except Exception:
        print(f"Error processing AI response: {traceback.format_exc()}")
        return JSONResponse(
            content={
                "action": "continue",
                "text": "I lost connection for a second. Could you please repeat your answer?"
            }
        )


@app.post("/terminate_interview/{session_id}", dependencies=[Depends(rate_limit_dependency)])
async def terminate(session_id: str, payload: RejectPayload, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Auth Token")
    token = authorization.split("Bearer ")[1]
    
    async with session_locks[session_id]:
        session = get_session(session_id, token)
        session["rejection_reason"] = payload.reason
        session["state"] = "TERMINATED"
        save_session(session_id, token, session)
        logger.warning(f"Session {session_id} TERMINATED. Reason: {payload.reason}")
    return {"status": "recorded"}


@app.post("/finish/{session_id}", dependencies=[Depends(rate_limit_dependency)])
async def finish_interview_endpoint(session_id: str, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Auth Token")
    token = authorization.split("Bearer ")[1]
    
    async with session_locks[session_id]:
        session = get_session(session_id, token)
        if session["state"] == "TERMINATED":
            raise HTTPException(status_code=400, detail="Session was terminated.")
        return JSONResponse(content=await evaluate_interview(session_id, token))


# ==========================================================
# NEW FEATURE: PDF DOWNLOAD ENDPOINT
# ==========================================================
@app.get("/download_pdf/{session_id}", dependencies=[Depends(rate_limit_dependency)])
async def download_pdf_report(session_id: str, token: Optional[str] = None):
    if not token:
        raise HTTPException(status_code=401, detail="Missing Auth Token query parameter.")
    
    session = get_session(session_id, token)
    
    # Initialize PDF object
    pdf = FPDF()
    pdf.add_page()
    
    # TITLE
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 10, txt="MockAI Interview Report", ln=True, align='C')
    pdf.ln(5)
    
    # OVERVIEW SECTION
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=f"Plan Selected: {session['plan'].upper()}", ln=True, align='L')
    
    status_text = "Terminated Early" if session["state"] == "TERMINATED" else f"{session.get('marks', 0)} / 100"
    pdf.cell(0, 8, txt=f"Final Status: {status_text}", ln=True, align='L')
    pdf.ln(8)
    
    # AI FEEDBACK SECTION
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Performance Feedback & Coaching:", ln=True, align='L')
    pdf.set_font("Arial", size=11)
    
    eval_json = session.get("eval_json", {})
    if eval_json.get("strengths"):
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 6, txt="Strengths:", ln=True)
        pdf.set_font("Arial", size=11)
        for s in eval_json["strengths"]: pdf.multi_cell(0, 6, txt=f"- {clean_text_for_pdf(s)}")
        pdf.ln(4)
        
    if eval_json.get("weaknesses"):
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 6, txt="Areas to Improve:", ln=True)
        pdf.set_font("Arial", size=11)
        for w in eval_json["weaknesses"]: pdf.multi_cell(0, 6, txt=f"- {clean_text_for_pdf(w)}")
        pdf.ln(4)
        
    if eval_json.get("ideal_answer_coaching"):
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 6, txt="Coaching (Ideal Answer):", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 6, txt=clean_text_for_pdf(eval_json["ideal_answer_coaching"]))
        pdf.ln(4)
    
    pdf.ln(6)
    
    # TRANSCRIPT SECTION
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Full Interview Transcript:", ln=True, align='L')
    pdf.ln(2)
    
    for msg in session.get("history", []):
        role = "AI Interviewer" if msg["role"] == "assistant" else "Candidate"
        clean_content = clean_text_for_pdf(msg["content"])
        
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 8, txt=f"{role}:", ln=True, align='L')
        
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 6, txt=clean_content)
        pdf.ln(4)
        
    # Output to raw bytes
    pdf_out = pdf.output(dest='S')
    
    # Handle FPDF versioning differences (fpdf vs fpdf2 return types)
    if isinstance(pdf_out, str):
        pdf_bytes = pdf_out.encode('latin-1')
    else:
        pdf_bytes = bytes(pdf_out)
        
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=MockAI_Report_{session_id}.pdf"}
    )


# Server Execution for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Mock Interviewer Server Online on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
