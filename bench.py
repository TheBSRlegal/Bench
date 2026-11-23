import streamlit as st
import os
import io
import uuid
import base64
import re
import time
import requests
from datetime import datetime, timezone, timedelta
from groq import Groq
from dotenv import load_dotenv

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, HRFlowable
)
from reportlab.lib import colors
from PIL import Image as PILImage

# ---------- ENV ----------
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_HERE, ".env"))

def _secret(key, default=None):
    val = os.getenv(key)
    if val is not None and val != "":
        return val
    try:
        return st.secrets[key]
    except Exception:
        return default

GROQ_API_KEY    = _secret("GROQ_API_KEY")
APPS_SCRIPT_URL = _secret("APPS_SCRIPT_URL")
LOG_SECRET      = _secret("LOG_SECRET")

MAX_COMPLETION_TOKENS  = int(_secret("BENCH_MAX_COMPLETION_TOKENS", "2000"))
CONTEXT_WINDOW         = int(_secret("BENCH_CONTEXT_WINDOW", "6"))
MAX_TURNS_PER_SESSION  = int(_secret("BENCH_MAX_TURNS_SESSION", "4"))
MAX_INPUT_CHARS        = int(_secret("BENCH_MAX_INPUT_CHARS", "1500"))
CTA_LABEL              = _secret("BENCH_CTA_LABEL", "Book a 30-minute Legal Diagnostic")
CTA_URL                = _secret("BENCH_CTA_URL", "https://www.thebsrlegal.com/contact-us")

if not GROQ_API_KEY:
    st.error("API configuration missing. Set GROQ_API_KEY in .env or Streamlit secrets.")
    st.stop()

if not APPS_SCRIPT_URL or not LOG_SECRET:
    st.warning("Telemetry disabled: APPS_SCRIPT_URL or LOG_SECRET missing.")

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="BSR AI Bench", page_icon="⚖️", layout="centered", initial_sidebar_state="expanded")

# ---------- STATE ----------
if "history" not in st.session_state:        st.session_state.history = []
if "session_id" not in st.session_state:     st.session_state.session_id = str(uuid.uuid4())[:8]
if "rate_limited" not in st.session_state:   st.session_state.rate_limited = False
if "retry_seconds" not in st.session_state:  st.session_state.retry_seconds = None
if "log_errors" not in st.session_state:     st.session_state.log_errors = 0
if "quota_blocked" not in st.session_state:  st.session_state.quota_blocked = False
if "reset_override" not in st.session_state: st.session_state.reset_override = ""

# ---------- LOGO ----------
logo_path = None
for ext in ("png", "jpg", "jpeg", "webp"):
    candidate = os.path.join(_HERE, f"logo.{ext}")
    if os.path.exists(candidate):
        logo_path = candidate
        break

# ---------- CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600&family=Inter:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #ffffff; color: #0a0a0a; }
    .stApp { background-color: #ffffff; }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; height: 0; min-height: 0; }
    button[data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] { display: none; }
    h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: #0a0a0a; }
    .block-container { padding-top: 0.4rem !important; padding-bottom: 2rem !important; max-width: 720px; }
    .header-wrap { text-align: center; margin: 0 auto 0.6rem auto; }
    .header-wrap img { display: block; margin: 0 auto 0.6rem auto; max-width: 340px; height: auto; }
    .header-wrap h1 { font-size: 1.7rem; letter-spacing: 0.06em; margin: 0 0 0.15rem 0; font-weight: 500; line-height: 1.2; }
    .header-wrap .subtitle { font-size: 0.6rem; color: #6b6b6b; letter-spacing: 0.18em; text-transform: uppercase; font-weight: 400; margin: 0; }
    .header-rule { border-top: 1px solid #0a0a0a; margin: 0.6rem 0 1rem 0; }
    .chat-row { margin: 1rem 0; display: flex; flex-direction: column; }
    .chat-row-user { align-items: flex-end; }
    .chat-row-bot { align-items: flex-start; }
    .bubble-user { background-color: #0a0a0a; color: #ffffff; padding: 0.85rem 1.15rem; font-size: 0.9rem; line-height: 1.65; max-width: 78%; }
    .bubble-bot { background-color: #ffffff; color: #0a0a0a; padding: 0.95rem 1.15rem; font-size: 0.9rem; line-height: 1.7; max-width: 88%; border-left: 1px solid #0a0a0a; }
    .bubble-critic { border-left-style: dashed; }
    .bubble-mentor { border-left: 3px solid #0a0a0a; }
    .speaker-label { font-size: 0.6rem; font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 0.4rem; color: #6b6b6b; }
    .tag-statute, .tag-binding { background:#0a0a0a; color:#fff; padding:0 4px; font-size:0.7em; letter-spacing:0.04em; }
    .tag-persuasive { border:1px solid #0a0a0a; padding:0 4px; font-size:0.7em; letter-spacing:0.04em; }
    .tag-unsettled { background:#f2c94c; color:#0a0a0a; padding:0 4px; font-size:0.7em; letter-spacing:0.04em; }
    .tag-inference { background:#e2e2e2; color:#0a0a0a; padding:0 4px; font-size:0.7em; letter-spacing:0.04em; }
    .rate-banner { background-color: #0a0a0a; color: #ffffff; padding: 1rem 1.2rem; margin: 1rem 0; text-align: center; font-size: 0.85rem; line-height: 1.6; }
    .rate-banner a { color: #ffffff; text-decoration: underline; }
    section[data-testid="stSidebar"] { background-color: #f7f7f7; border-right: 1px solid #e2e2e2; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }
    section[data-testid="stSidebar"] h3 { font-family: 'Playfair Display', serif !important; font-size: 1.05rem; margin-bottom: 0.4rem; margin-top: 0; }
    .session-tag { font-size: 0.55rem; color: #9b9b9b; letter-spacing: 0.1em; text-align: center; margin-bottom: 0.8rem; font-family: 'Inter', monospace; }
    .stButton > button { background-color: #0a0a0a; color: #ffffff; border: 1px solid #0a0a0a; padding: 0.5rem 1.2rem; font-size: 0.75rem; letter-spacing: 0.06em; font-weight: 500; text-transform: uppercase; }
    .stButton > button:hover { background-color: #ffffff; color: #0a0a0a; }
    .stDownloadButton > button { background-color: transparent; color: #0a0a0a; border: 1px solid #0a0a0a; font-size: 0.72rem; padding: 0.45rem 0.9rem; letter-spacing: 0.06em; font-weight: 500; text-transform: uppercase; }
    .stDownloadButton > button:hover { background-color: #0a0a0a; color: #ffffff; }
    .stChatInput textarea { border-radius: 0 !important; border: 1px solid #0a0a0a !important; font-size: 0.9rem !important; background-color: #ffffff !important; color: #0a0a0a !important; }
    .sidebar-cta { background-color: #0a0a0a; color: #ffffff; padding: 1rem 0.9rem; text-align: center; margin-top: 0.4rem; }
    .sidebar-cta p { margin: 0 0 0.6rem 0; opacity: 0.9; font-family: 'Playfair Display', serif; font-size: 0.9rem; }
    .sidebar-cta a { display: inline-block; padding: 0.45rem 0.9rem; background-color: #ffffff; color: #0a0a0a; text-decoration: none; font-weight: 500; font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase; }
    .footer-disclaimer { font-size: 0.58rem; color: #8a8a8a; text-align: center; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e2e2e2; line-height: 1.7; max-width: 620px; margin-left: auto; margin-right: auto; }
    .empty-state { text-align: center; padding: 2.4rem 1rem; color: #6b6b6b; }
    .empty-state p { font-size: 0.85rem; line-height: 1.8; max-width: 470px; margin: 0 auto; }
    .empty-state .hint { font-family: 'Playfair Display', serif; font-size: 1.15rem; color: #0a0a0a; margin-bottom: 0.6rem; }
</style>
""", unsafe_allow_html=True)

# ---------- TIME HELPERS ----------
IST = timezone(timedelta(hours=5, minutes=30))

def next_reset_utc():
    now_ist = datetime.now(IST)
    tomorrow_ist = (now_ist + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return tomorrow_ist.astimezone(timezone.utc)

def hours_until_reset():
    return max(0.0, (next_reset_utc() - datetime.now(timezone.utc)).total_seconds() / 3600)

def humanize_hours(h):
    if h < 1:   return "in less than an hour"
    if h < 2:   return "in about an hour"
    return f"in about {int(round(h))} hours"

# ---------- MODELS ----------
@st.cache_data(ttl=300)
def fetch_chat_models():
    try:
        models = client.models.list()
        exclude = ["whisper", "tts", "guard", "orpheus", "playai", "distil", "compound", "embed", "safeguard"]
        return sorted([m.id for m in models.data if not any(k in m.id.lower() for k in exclude)])
    except Exception:
        return []

available = fetch_chat_models()

def _resolve_role_model(env_key: str, index: int):
    explicit = _secret(env_key)
    if explicit:
        return explicit
    if index < len(available):
        return available[index]
    return None

role_models = {
    "Analyst": _resolve_role_model("BENCH_MODEL_ANALYST", 0),
    "Critic":  _resolve_role_model("BENCH_MODEL_CRITIC",  1),
    "Mentor":  _resolve_role_model("BENCH_MODEL_MENTOR",  2),
}

active_roles = [r for r, m in role_models.items() if m]

if not active_roles:
    st.error("No usable models found. Set GROQ_API_KEY or pin BENCH_MODEL_ANALYST / _CRITIC / _MENTOR.")
    st.stop()

def turns_used() -> int:
    return sum(1 for h in st.session_state.history if h["role"] == "user")

# ---------- REMOTE ----------
def check_global_quota():
    if not APPS_SCRIPT_URL or not LOG_SECRET:
        return True, ""
    try:
        r = requests.get(APPS_SCRIPT_URL, params={"token": LOG_SECRET}, timeout=5)
        data = r.json()
        return bool(data.get("allowed", True)), str(data.get("reset_message", "") or "")
    except Exception:
        return True, ""

@st.cache_data(ttl=30, show_spinner=False)
def quota_cached():
    return check_global_quota()

def flush_turn_logs(rows):
    if not APPS_SCRIPT_URL or not LOG_SECRET or not rows:
        return
    width = max(len(r) for r in rows)
    norm = [r + [""] * (width - len(r)) for r in rows]
    try:
        requests.post(APPS_SCRIPT_URL, json={"token": LOG_SECRET, "increment_turn": True, "rows": norm}, timeout=8)
    except Exception:
        st.session_state.log_errors += 1

if not st.session_state.quota_blocked:
    _allowed, _msg = quota_cached()
    if not _allowed:
        st.session_state.quota_blocked = True
    if _msg:
        st.session_state.reset_override = _msg

# ---------- TAGS ----------
TAG_NAMES = ["STATUTE", "BINDING", "PERSUASIVE", "UNSETTLED", "INFERENCE"]
TAG_RE = re.compile(r"\[(" + "|".join(TAG_NAMES) + r")\]", re.IGNORECASE)

def tag_summary(text: str) -> str:
    if not text: return ""
    counts = {t: 0 for t in TAG_NAMES}
    for m in TAG_RE.finditer(text):
        counts[m.group(1).upper()] += 1
    parts = [f"{t}={counts[t]}" for t in TAG_NAMES if counts[t] > 0]
    return ";".join(parts) if parts else "NONE"

def style_tags(html_text: str) -> str:
    return TAG_RE.sub(lambda m: f'<span class="tag-{m.group(1).lower()}">{m.group(1).upper()}</span>', html_text)

# ---------- PDF ----------
def sanitize_text(text):
    replacements = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": " - ", "\u2026": "...", "\u2192": "->", "\u2190": "<-", "\u2022": "-", "\u00A0": " "}
    for bad, good in replacements.items(): text = text.replace(bad, good)
    text = text.encode("latin-1", "ignore").decode("latin-1")
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def generate_pdf(history, session_id, logo_file):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=22*mm, rightMargin=22*mm, topMargin=28*mm, bottomMargin=22*mm, title=f"BSR AI Bench - Session {session_id}", author="The BSR Legal")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BrandWordmark", parent=styles["Title"], fontName="Times-Bold", fontSize=16, leading=20, alignment=TA_CENTER, textColor=colors.black, spaceAfter=2))
    styles.add(ParagraphStyle(name="BrandSub", parent=styles["Normal"], fontName="Helvetica", fontSize=7.5, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceAfter=10))
    styles.add(ParagraphStyle(name="MetaLine", parent=styles["Normal"], fontName="Helvetica", fontSize=7.5, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#777777"), spaceAfter=14))
    styles.add(ParagraphStyle(name="SpeakerLabel", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=7, leading=10, textColor=colors.black, spaceAfter=3))
    styles.add(ParagraphStyle(name="UserLabel", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=7, leading=10, textColor=colors.black, spaceAfter=3))
    styles.add(ParagraphStyle(name="BodyText2", parent=styles["Normal"], fontName="Helvetica", fontSize=9, leading=13.5, textColor=colors.black, spaceAfter=6))
    styles.add(ParagraphStyle(name="RoleBlock", parent=styles["Normal"], fontName="Helvetica", fontSize=9, leading=13.5, textColor=colors.black, leftIndent=8, spaceAfter=10))
    styles.add(ParagraphStyle(name="FooterNote", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=6.5, leading=9, alignment=TA_CENTER, textColor=colors.HexColor("#888888")))
    story = []
    if logo_file and os.path.exists(logo_file):
        try:
            with PILImage.open(logo_file) as im: w, h = im.size
            tw = 40 * mm
            logo = RLImage(logo_file, width=tw, height=tw*(h/w)); logo.hAlign = "CENTER"
            story.append(logo); story.append(Spacer(1, 8))
        except Exception: pass
    story.append(Paragraph("BSR <font size='11' color='#555555'>AI</font> BENCH", styles["BrandWordmark"]))
    story.append(Paragraph("MULTI-MODEL LEGAL REASONING", styles["BrandSub"]))
    story.append(Paragraph(f"Session {session_id} &nbsp;&middot;&nbsp; {datetime.now().strftime('%d %B %Y, %H:%M')}", styles["MetaLine"]))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.black, spaceBefore=2, spaceAfter=14))
    for h in history:
        content = sanitize_text(h["content"]).replace("\n", "<br/>")
        if h["role"] == "user":
            story.append(Paragraph("YOU", styles["UserLabel"]))
            story.append(Paragraph(content, styles["BodyText2"]))
            story.append(HRFlowable(width="100%", thickness=0.3, color=colors.HexColor("#cccccc"), spaceBefore=4, spaceAfter=10))
        else:
            story.append(Paragraph(h["role"].upper(), styles["SpeakerLabel"]))
            story.append(Paragraph(content, styles["RoleBlock"]))
    story.append(Spacer(1, 18))
    story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#888888"), spaceBefore=2, spaceAfter=8))
    story.append(Paragraph("BSR AI Bench is a demonstration of multi-model legal reasoning. Outputs are AI-generated and do not constitute legal advice. No attorney-client relationship is formed.", styles["FooterNote"]))
    doc.build(story); buffer.seek(0)
    return buffer.getvalue()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(f'<div class="session-tag">SESSION &middot; {st.session_state.session_id}</div>', unsafe_allow_html=True)
    st.markdown("### The Panel")
    st.caption("Three AI models build on each other's work: the Analyst examines the legal position, the Critic adds a perspective, and the Mentor synthesises a practical takeaway.")
    role_lines = "<br>".join(f"<b>{r}</b> &middot; Active" for r in active_roles)
    st.markdown(f"<div style='font-size:0.68rem;color:#4a4a4a;line-height:1.9;margin-top:0.4rem;'>{role_lines}</div>", unsafe_allow_html=True)
    st.markdown("---")
    if st.session_state.history:
        pdf_bytes = generate_pdf(st.session_state.history, st.session_state.session_id, logo_path)
        st.download_button("Export Chat (PDF)", data=pdf_bytes, file_name=f"BSR-AI-Bench-{st.session_state.session_id}.pdf", mime="application/pdf", use_container_width=True)
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.history = []; st.session_state.rate_limited = False; st.session_state.quota_blocked = False; st.rerun()
    else:
        st.button("Export Chat (PDF)", disabled=True, use_container_width=True)
        st.button("Clear Conversation", disabled=True, use_container_width=True)
    if st.session_state.log_errors:
        st.caption(f"Telemetry write failures: {st.session_state.log_errors}")
    st.markdown("---")
    st.markdown(f'<div class="sidebar-cta"><p>Need a lawyer\'s view?</p><a href="{CTA_URL}" target="_blank">{CTA_LABEL}</a></div>', unsafe_allow_html=True)

# ---------- HEADER ----------
header_html = '<div class="header-wrap">'
if logo_path:
    try:
        with open(logo_path, "rb") as f: b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(logo_path)[1].lstrip(".").lower()
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        header_html += f'<img src="data:image/{mime};base64,{b64}" alt="BSR Legal">'
    except Exception: pass
header_html += '<h1>BSR <span style="font-weight:400;font-size:0.6em;letter-spacing:0.2em;">AI</span> BENCH</h1><div class="subtitle">Multi-Model Legal Reasoning</div></div>'
st.markdown(header_html, unsafe_allow_html=True)
st.markdown('<div class="header-rule"></div>', unsafe_allow_html=True)

# ---------- PROMPTS ----------
LEGAL_CERTAINTY_RULES = """
MANDATORY LEGAL-CERTAINTY RULES (apply to every sentence you write):

1. Never present a general legal principle as settled law unless supported by an identifiable statutory provision or judicial precedent.

2. Label every legal proposition with exactly one authority tier:
   - [STATUTE] - you are quoting or closely paraphrasing a statutory provision you are certain exists and contains the rule you are stating.
   - [BINDING] - a precedent from a court binding in the stated jurisdiction.
   - [PERSUASIVE] - a precedent or doctrine from a non-binding source.
   - [UNSETTLED] - contested, jurisdiction-dependent, or no clear authority.
   - [INFERENCE] - your own legal reasoning, not sourced.

3. If the jurisdiction is not stated, DO NOT assume one. Say: "Jurisdiction not stated; the following is general and may differ by country/state."

4. Do not invent a citation, case name, or section number. If you cannot identify authority, tag [INFERENCE] or omit it.

5. Precision over completeness.
"""

ROLE_PROMPTS = {
    "Analyst": LEGAL_CERTAINTY_RULES + """
You are the Analyst on BSR AI Bench. You write the primary answer the user will read.

Your job:
- Read the user's question and the conversation so far.
- Identify the jurisdiction. Use the one stated in the current or most recent user message. If none is stated, say "Jurisdiction not stated; the following is general and may differ by country/state" and proceed generally.
- Give a structured legal analysis in no more than 5 sentences, applying the facts to the law.
- Tag every legal proposition with exactly one authority tier.
- Be specific and useful. Do not hedge everything. Commit to the analysis the facts support.

You are not the Critic and you are not the Mentor. Do not critique yourself, do not add caveats for the sake of caution, and do not close with "consult a lawyer" — the Mentor does that.

Never give definitive legal advice.
""",

    "Critic": LEGAL_CERTAINTY_RULES + """
You are the Critic on BSR AI Bench. You refine the Analyst's answer. You do not attack it.

Your job:
- Read the Analyst's answer above.
- Identify ONE thing that would make the answer materially better for the user: a missing fact that would change the analysis, a practical step the Analyst didn't mention, or an alternative framework a lawyer would consider.
- Add that one thing in 1–2 sentences. Tag your own new propositions.
- If the Analyst's answer is already complete and correct, say: "The Analyst's answer is complete on the law; the practical next step is to [specific step]." Then give that step.

What you must NOT do:
- Do not retag, downgrade, or contradict the Analyst's propositions. If you think a citation is questionable, phrase it as: "Worth verifying: [Act name] — confirm it applies to these facts." Never say "unverified" or "retag as UNSETTLED" about the Analyst's work.
- Do not introduce a new jurisdiction.
- Do not restate the Analyst's answer.
- Do not open with "Untagged assertion" or "No Analyst response was provided."

You and the Analyst are on the same team, presenting to the same client. Your job is to strengthen the answer, not to score points against it.

Maximum 3 sentences. Never give definitive legal advice.
""",

    "Mentor": LEGAL_CERTAINTY_RULES + """
You are the Mentor on BSR AI Bench. You give the user the practical takeaway that ties the Analyst's and Critic's work together.

Your job:
- Read the Analyst's answer and the Critic's addition.
- Write a short, practical synthesis: what the law means for this user, and what they should do next.
- Use the jurisdiction from the current or most recent user message. Do not revert to "jurisdiction not stated" if a jurisdiction was given earlier in the conversation.
- Preserve the Analyst's authority tags. Do not upgrade [INFERENCE] or [UNSETTLED] to settled. Do not downgrade a [STATUTE] unless the Critic raised a specific, substantive concern about that exact provision.
- If the central proposition is [UNSETTLED] or [INFERENCE], open with: "This is not settled; a lawyer in your jurisdiction should review it before you act." Otherwise, do not lead with a disclaimer — give the takeaway.
- End with one concrete next step.

You are the closing voice. Your job is to leave the client with clarity and a path forward, not with a list of caveats.

Maximum 4 sentences. Never give definitive legal advice.
""",
}

def build_context(history, current_role):
    """Always includes the full current turn. Prior context is trimmed; the current turn is not."""
    turn_start = 0
    for i in range(len(history) - 1, -1, -1):
        if history[i]["role"] == "user":
            turn_start = i
            break

    current_turn = history[turn_start:]
    prior = history[max(0, turn_start - CONTEXT_WINDOW):turn_start]

    msgs = []
    for h in prior + current_turn:
        msgs.append({"role": "user" if h["role"] == "user" else "assistant", "content": h["content"]})
    msgs.append({"role": "system", "content": ROLE_PROMPTS[current_role]})
    return msgs

def call_model(model_id, context, role, max_retries=2):
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=context,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                reasoning_effort="low",
                temperature=0.3,
            )
            latency = int((time.perf_counter() - t0) * 1000)
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
            finish = getattr(choice, "finish_reason", None) or "stop"
            if finish == "length" and not text:
                if attempt < max_retries:
                    time.sleep(2); continue
                return None, "a moment", "empty_reasoning_exhausted", latency
            return text, None, finish, latency
        except Exception as e:
            latency = int((time.perf_counter() - t0) * 1000)
            s = str(e); sl = s.lower()
            if "429" in s or "rate" in sl:
                if attempt < max_retries:
                    wait = 3
                    if "retry_after" in sl:
                        try: wait = float(s.split("retry_after:")[1].split("}")[0].strip(" '\""))
                        except Exception: pass
                    time.sleep(min(wait, 10)); continue
                return None, "a few minutes", "rate_limited", latency
            return f"[{role} temporarily unavailable: {s[:80]}]", None, "error", latency
    return None, "a moment", "rate_limited", 0

# ---------- CHAT RENDER ----------
if not st.session_state.history:
    roles_sentence = " and ".join(active_roles)
    st.markdown(f'<div class="empty-state"><div class="hint">A panel of AI models, one shared legal discussion.</div><p>Ask a question. The {roles_sentence} will respond in sequence, each building on the replies before it.</p></div>', unsafe_allow_html=True)

for h in st.session_state.history:
    if h["role"] == "user":
        st.markdown(f'<div class="chat-row chat-row-user"><div class="bubble-user">{h["content"]}</div></div>', unsafe_allow_html=True)
    else:
        cls = {"Analyst": "bubble-bot", "Critic": "bubble-bot bubble-critic", "Mentor": "bubble-bot bubble-mentor"}.get(h["role"], "bubble-bot")
        rendered = style_tags(h["content"]).replace("\n", "<br>")
        st.markdown(f'<div class="chat-row chat-row-bot"><div class="{cls}"><div class="speaker-label">{h["role"]}</div>{rendered}</div></div>', unsafe_allow_html=True)

# ---------- BANNERS ----------
if st.session_state.rate_limited:
    st.markdown(f'<div class="rate-banner">The AI Bench has reached its capacity for this session. Please try again in <strong>{st.session_state.retry_seconds or "a few minutes"}</strong>.</div>', unsafe_allow_html=True)
if st.session_state.quota_blocked:
    tu = turns_used()
    reset_line = st.session_state.reset_override or f"It resets {humanize_hours(hours_until_reset())} (at {next_reset_utc().strftime('%H:%M UTC')})."
    headline = "The AI panel is at capacity for today." if tu == 0 else "You've reached today's usage limit."
    st.markdown(f'<div class="rate-banner"><strong>{headline}</strong><br>{reset_line}<br><br><a href="{CTA_URL}" target="_blank">{CTA_LABEL} &rarr;</a></div>', unsafe_allow_html=True)

# ---------- INPUT ----------
blocked = st.session_state.rate_limited or st.session_state.quota_blocked

if not blocked:
    if turns_used() >= MAX_TURNS_PER_SESSION:
        st.info(f"This session has reached its limit of {MAX_TURNS_PER_SESSION} questions. Reload to start a new session, or book a consultation.")
    else:
        user_input = st.chat_input("Ask a legal question or describe a scenario...")
        if user_input:
            if len(user_input) > MAX_INPUT_CHARS:
                st.warning(f"Please keep your question under {MAX_INPUT_CHARS} characters.")
            else:
                allowed, override = check_global_quota()
                if not allowed:
                    st.session_state.quota_blocked = True
                    if override: st.session_state.reset_override = override
                    st.rerun()
                else:
                    turn = turns_used() + 1
                    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    turn_rows = []
                    st.session_state.history.append({"role": "user", "content": user_input})
                    turn_rows.append([now_iso, st.session_state.session_id, turn, "user", "", "", "", "", user_input, ""])
                    for role in active_roles:
                        model_id = role_models[role]
                        ctx = build_context(st.session_state.history, role)
                        reply, retry_time, finish, latency = call_model(model_id, ctx, role)
                        if retry_time is not None:
                            st.session_state.rate_limited = True; st.session_state.retry_seconds = retry_time
                            turn_rows.append([now_iso, st.session_state.session_id, turn, role, model_id, model_id, finish, latency, f"[{finish}]", ""])
                            break
                        st.session_state.history.append({"role": role, "content": reply, "model": model_id})
                        turn_rows.append([now_iso, st.session_state.session_id, turn, role, model_id, model_id, finish, latency, reply, tag_summary(reply)])
                    flush_turn_logs(turn_rows)
                    st.rerun()

# ---------- FOOTER ----------
st.markdown('<div class="footer-disclaimer">BSR AI Bench is a demonstration of multi-model legal reasoning. Outputs are AI-generated and do not constitute legal advice. No attorney-client relationship is formed.</div>', unsafe_allow_html=True)