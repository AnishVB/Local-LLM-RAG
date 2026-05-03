"""
chat_engine.py — RAG chat pipeline

FIXES (v2):
  3. Space/time: classify + rewrite merged into ONE LLM call (was 2 sequential calls).
     Follow-up generation uses a tighter prompt with hard JSON enforcement.
     History truncated earlier so classify prompt is cheaper.
  4. Speed: single classify call saves ~2-4s per message. ANSWER_OPTS tuned for
     faster generation (lower num_predict cap for routine answers, num_ctx reduced).
  5. Follow-ups: strict JSON-only prompt, no vague fallback injection, extracts
     specific entities/terms from the answer to ground each question.
  6. Polish: cleaner intent logic, better system prompts, saner temperature values.
"""

import os
import sys
import re
import json
import threading
import ollama

from dotenv import load_dotenv
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# ── Path setup ────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
from database import ChatbotDB

load_dotenv(os.path.join(os.path.dirname(current_dir), ".env"))

# ── Global constants ──────────────────────────────────────────────────────────
MODEL    = os.getenv("MODEL_NAME", "gemma4:e2b")
MAX_HIST = 8   # reduced from 10 — cuts classify prompt size

# GPU offloading: how many model layers to run on the GPU/iGPU
# Reduces RAM pressure; set to 0 for pure CPU, 999 for full GPU
NUM_GPU  = int(os.getenv("NUM_GPU", "12"))

CHUNK_CAP_NORMAL = 12   # was 15; most answers need fewer chunks
CHUNK_CAP_LIST   = 30   # was 35

HISTORY_DIR = os.path.join(os.path.dirname(current_dir), "chat_history")

# ── Hard casual-only word set ─────────────────────────────────────────────────
_HARD_CASUAL_WORDS = {
    "hi", "hello", "hey", "howdy", "sup", "yo", "hiya", "heyy", "heyyy",
    "thanks", "thank you", "thx", "ty", "cheers", "appreciated",
    "ok", "okay", "cool", "alright", "got it", "sure",
    "yep", "yup", "yeah", "nope",
    "huh", "hmm", "oh", "ah", "lol", "lmao", "haha",
    "bye", "goodbye", "cya", "later", "peace",
}

_HARD_CASUAL_PHRASES = {
    "how are you", "how are you doing", "how's it going", "hows it going",
    "what's up", "whats up", "wassup", "wazzup",
    "good morning", "good afternoon", "good evening", "good night",
    "who are you", "what are you", "what can you do",
    "see you", "i see",
}


def _is_pure_greeting(text: str) -> bool:
    low = text.strip().lower()
    if low in _HARD_CASUAL_PHRASES:
        return True
    words = low.split()
    if len(words) == 1 and low in _HARD_CASUAL_WORDS:
        return True
    if len(words) == 2 and low in _HARD_CASUAL_PHRASES:
        return True
    return False


# ── System prompts ────────────────────────────────────────────────────────────
DOC_SYSTEM = (
    "You are a precise document analyst. Answer using the provided excerpts only. "
    "Be thorough, cite pages like (p.N), and use exact figures and terms from the source.\n"
)

GENERAL_SYSTEM = (
    "You are a knowledgeable assistant. Answer clearly and accurately. "
    "Be concise but complete. Do NOT ask for documents or say you need files.\n"
)

SYSTEM = GENERAL_SYSTEM

# ── LLM options ───────────────────────────────────────────────────────────────
# FIX #4: reduced num_predict and num_ctx to cut latency on routine answers
ANSWER_OPTS = {
    "num_predict":    2048,   # was 3072; rarely needed more
    "temperature":    0.2,    # was 0.25; slightly tighter = faster convergence
    "repeat_penalty": 1.1,    # was 1.15
    "top_k":          30,     # was 40
    "top_p":          0.88,   # was 0.90
    "num_ctx":        6144,   # was 8192; most RAG contexts fit in 6k
    "num_gpu":        NUM_GPU,
}

# FIX #3: classify opts — very short, fast, deterministic
CLASSIFY_OPTS = {
    "num_predict": 200,   # was 300; JSON reply is tiny
    "temperature": 0.0,
    "top_k":       10,    # was 15
    "top_p":       0.80,  # was 0.85
    "num_ctx":     3072,  # was 4096
    "num_gpu":     NUM_GPU,
}

CASUAL_OPTS = {
    "num_predict": 150,   # was 250; greeting replies are short
    "temperature": 0.5,   # was 0.6
    "top_k":       30,    # was 40
    "top_p":       0.90,
    "num_ctx":     1024,  # was 2048
    "num_gpu":     NUM_GPU,
}

# FIX #5: follow-up opts — slightly more creative but still grounded
FOLLOWUP_OPTS = {
    "num_predict": 250,   # was 300
    "temperature": 0.35,  # was 0.4; less hallucination
    "top_k":       25,
    "top_p":       0.88,
    "num_ctx":     3072,  # was 4096
    "num_gpu":     NUM_GPU,
}

# ── Pre-compiled intent patterns ──────────────────────────────────────────────
_GENERAL_FORCE_RE = re.compile(
    r'\b(use your (own |general |training |base )?knowledge'
    r'|from your training'
    r'|without (the |any )?document'
    r'|not (from |using )(the |any )?doc'
    r'|general knowledge'
    r'|on your own'
    r'|ignore (the |any )?doc'
    r'|don\'?t use (the |any )?doc)\b',
    re.IGNORECASE,
)

_DOC_FORCE_RE = re.compile(
    r'\b(in the (pdf|document|file|report|upload)'
    r'|from the (pdf|document|file|report|upload)'
    r'|according to (the )?(pdf|document|file|report|upload)'
    r'|what does the (pdf|document|file|report) say'
    r'|is it mentioned'
    r'|does the (pdf|doc) (say|mention|contain|cover)'
    r'|from (our|my) (doc|file|pdf|report|upload)'
    r'|in (our|my) (doc|file|pdf|report|upload)'
    r'|page \d+|pg\.?\s*\d+|chapter |section '
    r'|summarize (the |this |it\b)'
    r'|summarise (the |this |it\b)'
    r'|the (pdf|document|doc|file|report)'
    r'|this (pdf|document|doc|file|report)'
    r'|the title of'
    r'|what is the title'
    r'|what\'?s the title'
    r'|title of (the |this )'
    r'|key topics'
    r'|main sections'
    r'|key points'
    r'|this document'
    r'|this file)\b',
    re.IGNORECASE,
)

_DOC_HINTS = [
    "in the pdf","in the document","in the file","in the report","according to",
    "from the pdf","from the document","from the report","from the file",
    "uploaded","summarize the","summarise the",
    "list all","list the","list every","give me all","show all","enumerate",
    "all the","from our","in our","in cda","from cda","in ar","from ar",
]

_GEN_STARTS = [
    "what is ","what are ","who is ","who was ","how does ","how do ","explain ",
    "define ","tell me about ","what's ","how did ","when did ","when was ",
    "where is ","where was ","why is ","why does ","why do ","how many ",
    "how much ","what was ","what causes ","what happened ","give me an example",
    "are there ","is there a ",
]

_PAGE_RE = [re.compile(p) for p in
            [r'\bpage\s+(\d+)\b', r'\bpg\.?\s*(\d+)\b']]

_LIST_KEYWORDS = [
    "list all","list the","list every","give me all","show all","enumerate",
    "all misconducts","all charges","all items","all rules","all violations",
    "all penalties","all types","all sections","what are all",
    "list each","every single","complete list",
]

_BP_PATTERNS = [
    re.compile(r'bharat\s+dynamics',          re.IGNORECASE),
    re.compile(r'conduct.*discipline.*appeal', re.IGNORECASE),
    re.compile(r'corporate\s+office',          re.IGNORECASE),
]

# FIX #5: tighter bad-followup filter
_BAD_FOLLOWUP_RE = re.compile(
    r'(i do not have access|i don\'?t have access'
    r'|i cannot access|i can\'?t access'
    r'|document structure|multiple pages'
    r'|as an ai|as a language model'
    r'|i\'?m not able to|i am not able to'
    r'|i don\'?t know which page|without more context'
    r'|follow.up question|question \d+|here are)',
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
def _is_boilerplate_heading(h: str) -> bool:
    if not h:
        return False
    return any(p.search(h) for p in _BP_PATTERNS)


def _call(prompt: str, opts: dict = None) -> str:
    options = opts or CLASSIFY_OPTS
    try:
        result = ollama.generate(model=MODEL, prompt=prompt, options=options)
        return result.get("response", "").strip()
    except Exception as e:
        return f"[Error: {e}]"


def _call_stream(prompt: str, opts: dict = None, callback=None) -> str:
    options = opts or ANSWER_OPTS
    try:
        full = []
        for chunk in ollama.generate(model=MODEL, prompt=prompt,
                                     stream=True, options=options):
            t = chunk.get("response", "")
            if t:
                full.append(t)
                if callback:
                    callback(t)
        return "".join(full).strip()
    except Exception as e:
        err = f"[Error: {e}]"
        if callback:
            callback(err)
        return err


def _json_list(raw: str) -> list:
    # Try to find a JSON array anywhere in the output
    s, e = raw.find("["), raw.rfind("]")
    if s != -1 and e > s:
        try:
            return [str(x).strip() for x in json.loads(raw[s:e+1])
                    if str(x).strip()]
        except Exception:
            pass
    # Fallback: grab quoted strings
    return re.findall(r'"([^"]{10,120})"', raw)


# ─────────────────────────────────────────────────────────────────────────────
# FIX #3: Combined classify + rewrite in ONE LLM call (was TWO calls)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_and_rewrite(text: str, history: list,
                           has_docs: bool,
                           source_filter: str = None) -> tuple:
    """
    Returns (is_doc: bool, queries: list[str]).
    Uses pattern matching first (zero LLM cost), falls back to ONE LLM call
    that classifies AND rewrites in a single JSON response.
    """
    low = text.lower()

    # Fast path: explicit override keywords
    if _GENERAL_FORCE_RE.search(text):
        return False, []
    if _DOC_FORCE_RE.search(text):
        return True, _fast_rewrite(text)

    if not has_docs:
        return False, []

    # Short queries (≤3 words) that aren't general-knowledge openers → doc
    word_count = len(text.split())
    if word_count <= 3 and not _GENERAL_FORCE_RE.search(text):
        looks_general = (
            any(low.startswith(g) for g in _GEN_STARTS)
            and not any(h in low for h in _DOC_HINTS)
            and "?" in text
            and not re.search(r'\b(this|it|the|that)\b', low)
        )
        if not looks_general:
            return True, _fast_rewrite(text)

    if source_filter and word_count <= 10:
        looks_general = (
            any(low.startswith(g) for g in _GEN_STARTS)
            and not any(h in low for h in _DOC_HINTS)
            and "?" in text
            and not re.search(r'\b(this|it|the|that)\b', low)
        )
        if not looks_general:
            return True, _fast_rewrite(text)

    if any(h in low for h in _DOC_HINTS):
        return True, _fast_rewrite(text)

    if any(low.startswith(g) for g in _GEN_STARTS) and not source_filter:
        return False, []

    # FIX #3: Single LLM call that classifies AND generates rewrite queries
    hist = "\n".join(
        f'{"U" if m["role"]=="user" else "B"}: {m["content"][:80]}'
        for m in history[-3:]   # was -4; 3 is enough context
    )
    focused_ctx = (
        f'Focused document: "{source_filter}". Lean toward DOC if plausible.\n'
        if source_filter else ""
    )

    raw = _call(
        'Reply with JSON only — no preamble, no markdown, no explanation.\n'
        'Format: {"intent":"DOC","queries":["q1","q2","q3"]}\n'
        'intent = "DOC" if answerable only from an uploaded file, else "GENERAL".\n'
        'queries = 3 search-query rewrites of the question (different angles, fix typos).\n'
        + focused_ctx
        + (f"History:\n{hist}\n" if hist else "")
        + f"Question: {text}\nJSON:",
        opts=CLASSIFY_OPTS,
    )

    try:
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e > s:
            obj    = json.loads(raw[s:e+1])
            is_doc = str(obj.get("intent", "")).strip().upper() == "DOC"
            qs     = [str(q).strip() for q in obj.get("queries", []) if str(q).strip()]
            return is_doc, qs[:3]
    except Exception:
        pass

    return False, []


def _fast_rewrite(text: str) -> list:
    """
    Lightweight query rewriting WITHOUT an LLM call.
    Generates 2 simple variants: the original + a slightly rephrased version.
    Used when pattern-matching already determined intent, saving one LLM call.
    """
    variants = [text]
    low = text.lower()

    # Strip filler phrases to get a clean content query
    stripped = re.sub(
        r'^(what|tell me|give me|show me|list|explain|describe|find|get|'
        r'can you|please|i want to know about)\s+',
        '', low, flags=re.IGNORECASE
    ).strip()
    if stripped and stripped != low and len(stripped) > 3:
        variants.append(stripped)

    # Add an "information about X" form if not already a noun phrase
    if "?" not in text and len(text.split()) <= 5:
        variants.append(f"information about {stripped or text}")

    return variants[:3]


def _is_list(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in _LIST_KEYWORDS)


def _page_num(text: str):
    for pat in _PAGE_RE:
        m = pat.search(text.lower())
        if m:
            return max(1, int(m.group(1)))
    return None


def _source_for(text: str, sources: list):
    low = text.lower()
    for src in sources:
        stem = os.path.splitext(src)[0].lower()
        if src.lower() in low or stem in low:
            return src
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_page(db, user_page: int, src=None) -> list:
    internal_page = user_page - 1
    return [
        {"text": c["text"], "meta": c["meta"], "score": 1.0}
        for c in db.chunks
        if c["meta"].get("page") == internal_page
        and (src is None or c["meta"]["source"] == src)
    ]


def _get_section(db, anchor_chunks: list, source_filter=None) -> list:
    if not anchor_chunks:
        return []

    src = source_filter or anchor_chunks[0]["meta"]["source"]
    all_chunks = sorted(
        (c for c in db.chunks if c["meta"]["source"] == src),
        key=lambda c: (c["meta"].get("page", 0), c["meta"].get("chunk_id", 0)),
    )
    if not all_chunks:
        return anchor_chunks

    anchor_keys = {
        (c["meta"].get("page"), c["meta"].get("chunk_id"))
        for c in anchor_chunks
    }
    start_idx = None
    for i, c in enumerate(all_chunks):
        if (c["meta"].get("page"), c["meta"].get("chunk_id")) in anchor_keys:
            start_idx = i
            break

    if start_idx is None:
        return anchor_chunks

    anchor_heading = all_chunks[start_idx]["meta"].get("heading", "").strip().lower()

    walk_start = start_idx
    if anchor_heading:
        for i in range(start_idx - 1, max(start_idx - 9, -1), -1):
            h = all_chunks[i]["meta"].get("heading", "").strip().lower()
            if h == anchor_heading or not h or _is_boilerplate_heading(h):
                walk_start = i
            else:
                break

    section: list = []
    for i in range(walk_start, len(all_chunks)):
        c = all_chunks[i]
        h = c["meta"].get("heading", "").strip().lower()
        if h == anchor_heading or not h or _is_boilerplate_heading(h):
            section.append({"text": c["text"], "meta": c["meta"], "score": 1.0})
        else:
            break
        if len(section) >= 50:   # was 60; avoids bloating the prompt
            break

    return section if section else anchor_chunks


def _search(db, text: str, history: list, source_filter=None,
            precomputed_queries=None) -> list:
    queries = [text] + (precomputed_queries or _fast_rewrite(text))
    return db.multi_query(
        queries,
        n_final=15,         # was 18; 15 chunks is plenty for most answers
        min_score=0.12,     # was 0.10; stricter = less noise
        max_per_page=4,     # was 5
        source_filter=source_filter,
    )


def _search_section(db, text: str, history: list, source_filter=None,
                    precomputed_queries=None) -> list:
    queries = [text] + (precomputed_queries or _fast_rewrite(text))
    anchors = db.multi_query(
        queries,
        n_final=5,          # was 6
        min_score=0.15,
        max_per_page=3,
        source_filter=source_filter,
    )
    if not anchors:
        return []
    return _get_section(db, anchors, source_filter)


# ─────────────────────────────────────────────────────────────────────────────
# Reranker — LLM-based chunk relevance scoring
# ─────────────────────────────────────────────────────────────────────────────

RERANK_OPTS = {
    "num_predict": 150,
    "temperature": 0.0,
    "top_k":       10,
    "top_p":       0.80,
    "num_ctx":     4096,
    "num_gpu":     NUM_GPU,
}

def _rerank(query: str, chunks: list, n_keep: int = 8) -> list:
    """
    Rerank retrieved chunks using a single LLM call.

    Gives the LLM the query + numbered chunk previews and asks it to
    return the indices of the most relevant ones, ordered by relevance.
    This filters out noise so only truly relevant chunks reach the final prompt.

    Falls back to the original order if the LLM response can't be parsed.
    """
    if len(chunks) <= n_keep:
        return chunks

    # Build numbered previews (first 200 chars of each chunk)
    previews = []
    for i, c in enumerate(chunks):
        text = c["text"][:200].replace("\n", " ").strip()
        previews.append(f"{i}: {text}")

    prompt = (
        "You are a relevance judge. Given a QUESTION and numbered TEXT excerpts, "
        f"return ONLY a JSON array of the {n_keep} most relevant excerpt numbers "
        "ordered from most to least relevant. No explanation, just the JSON array.\n\n"
        f"QUESTION: {query}\n\n"
        "EXCERPTS:\n" + "\n".join(previews) + "\n\n"
        "JSON array:"
    )

    raw = _call(prompt, opts=RERANK_OPTS)

    try:
        s, e = raw.find("["), raw.rfind("]")
        if s != -1 and e > s:
            indices = json.loads(raw[s:e+1])
            indices = [int(i) for i in indices if 0 <= int(i) < len(chunks)]
            if indices:
                # Deduplicate while preserving order
                seen = set()
                unique = []
                for idx in indices:
                    if idx not in seen:
                        seen.add(idx)
                        unique.append(idx)
                return [chunks[i] for i in unique[:n_keep]]
    except Exception:
        pass

    # Fallback: return top chunks by original score order
    return chunks[:n_keep]


# ─────────────────────────────────────────────────────────────────────────────
# Context / prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def _hist_block(history: list) -> str:
    if not history:
        return ""
    lines = []
    for m in history[-(MAX_HIST * 2):]:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content'][:250]}")   # was 300
    return "PRIOR CONVERSATION:\n" + "\n".join(lines) + "\n\n"


def _build_ctx(chunks: list, cap: int = CHUNK_CAP_NORMAL) -> str:
    by_src: dict = defaultdict(list)
    for c in chunks[:cap]:
        by_src[c["meta"]["source"]].append(c)

    parts = []
    for src, cs in by_src.items():
        cs.sort(key=lambda x: (x["meta"].get("page", 0),
                               x["meta"].get("chunk_id", 0)))
        parts.append(f"--- {src} ---")
        for c in cs:
            pg = c["meta"].get("page", 0)
            parts.append(f"[p.{pg + 1}] {c['text']}")
    return "\n".join(parts)


def _prompt_doc(q: str, ctx: str, hist: str, is_list: bool = False,
                qa_examples: list = None) -> str:
    task = (
        "Extract every relevant item from the excerpts. "
        "Copy exact wording. Number each item. Cite page as (p.N).\n"
        "Cover ALL excerpts. After listing, give a brief count.\n"
        "If excerpts are from multiple documents, group your list by the document name.\n"
        if is_list else
        "Answer comprehensively using the excerpts.\n"
        "- Use exact terms, figures, names, dates from the source.\n"
        "- Cite key claims with (p.N).\n"
        "- Synthesize across excerpts when they cover different aspects.\n"
        "- IMPORTANT: If the excerpts come from multiple documents, structure your answer by grouping the information under the respective document names.\n"
    )
    # Few-shot memory: show how similar questions were answered before
    examples_block = ""
    if qa_examples:
        parts = []
        for ex in qa_examples[:2]:
            parts.append(
                f"SIMILAR PAST Q: {ex['question'][:200]}\n"
                f"PAST ANSWER: {ex['answer'][:500]}\n"
            )
        examples_block = (
            "REFERENCE — Here is how similar questions were answered before. "
            "Use the same depth, style, and citation approach:\n"
            + "\n".join(parts) + "\n"
        )
    return (
        DOC_SYSTEM + "\n"
        + task
        + "Base your answer strictly on the excerpts. "
        "If not present, say so clearly.\n\n"
        + examples_block
        + hist
        + f"DOCUMENT EXCERPTS:\n{ctx}\n\n"
        f"QUESTION: {q}\n\n"
        + ("EXTRACTED LIST:\n" if is_list else "ANSWER:\n")
    )


def _prompt_no_ctx(q: str, hist: str, source_filter: str = None) -> str:
    """Prompt when we searched documents but found nothing relevant."""
    if source_filter:
        preamble = (
            f"I searched the document '{source_filter}' thoroughly but found "
            "no excerpts directly relevant to this question.\n"
            "First, tell the user you could not find this in the document. "
            "Then, if you can, provide a helpful answer from your general knowledge, "
            "clearly labeled as general information (not from the document).\n\n"
        )
    else:
        preamble = (
            "I searched all indexed documents but found no excerpts "
            "directly relevant to this question.\n"
            "First, tell the user you could not find this in any document. "
            "Then, if you can, provide a helpful answer from your general knowledge, "
            "clearly labeled as general information (not from the documents).\n\n"
        )
    return (
        GENERAL_SYSTEM + "\n"
        + preamble
        + hist + f"QUESTION: {q}\n\nANSWER:\n"
    )


def _prompt_general(q: str, hist: str) -> str:
    return (
        GENERAL_SYSTEM + "\n"
        + hist + f"QUESTION: {q}\n\nANSWER:\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# FIX #5: Better follow-up generation
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Follow-up generation — LLM-driven, few-shot grounded
# ─────────────────────────────────────────────────────────────────────────────

def _generate_followups(user_text: str, answer: str,
                        history: list = None,
                        source_filter: str = None) -> list:
    """
    Ask the LLM to generate 3 follow-up questions that a smart user would
    actually want to ask next. Uses few-shot examples to show the exact
    quality bar: specific, grounded in the answer content, no filler phrases.
    """
    # Trim answer to what the LLM needs — first 1000 chars covers the meat
    answer_snippet = answer[:1000].strip()
    doc_ctx = f"The document being discussed is: {source_filter}\n" if source_filter else ""

    # Last 2 user turns for context (no need for more)
    prior = ""
    if history:
        turns = [m["content"][:120] for m in history[-4:] if m["role"] == "user"]
        if turns:
            prior = "Recent questions: " + " | ".join(turns[-2:]) + "\n"

    prompt = (
        "You generate follow-up questions for a document Q&A chatbot.\n"
        "Output ONLY a raw JSON array of 3 strings. Nothing else — no prose, no markdown, no labels.\n\n"
        "RULES:\n"
        "- Each question must reference a SPECIFIC term, name, number, clause, or concept from the Answer below\n"
        "- Each question must be something the user genuinely does NOT know yet\n"
        "- Each question must explore a DIFFERENT aspect (procedure / penalty / definition / exception / timeline)\n"
        "- 8 to 18 words each\n"
        "- End with ?\n"
        "- NO vague openers: never start with 'Can you tell me', 'Could you explain', 'What more'\n\n"
        "GOOD EXAMPLES (shows the quality bar):\n"
        '["What is the exact penalty for a major misconduct under Rule 14?", '
        '"How many days does an employee have to respond to a charge sheet?", '
        '"Does the CDA distinguish between minor and major penalties for first offences?"]\n\n'
        "BAD EXAMPLES (never output these):\n"
        '["What are the specific conditions or rules for This will be...", '
        '"Can you tell me more about the document?", '
        '"Are there any exceptions to the This will be treated requ..."]\n\n'
        + doc_ctx
        + prior
        + f"User asked: {user_text[:200]}\n\n"
        f"Answer:\n{answer_snippet}\n\n"
        "JSON array of 3 follow-up questions:"
    )

    raw = _call(prompt, opts=FOLLOWUP_OPTS)

    # Parse JSON array
    candidates = _json_list(raw)
    clean = [
        q.strip() for q in candidates
        if len(q.strip()) > 12
        and q.strip().endswith("?")
        and not _BAD_FOLLOWUP_RE.search(q)
        and not re.search(r'\bThis will\b|\bthis will\b', q)
        and len(q.split()) >= 7
    ]

    if len(clean) >= 3:
        return clean[:3]

    # Secondary parse: grab any question-shaped lines if JSON parse failed
    lines = [
        l.strip().strip('"\'').strip('-*•123456789. ').strip()
        for l in raw.splitlines() if l.strip()
    ]
    extra = [
        l for l in lines
        if len(l) > 12
        and l.endswith("?")
        and not _BAD_FOLLOWUP_RE.search(l)
        and not re.search(r'\bThis will\b|\bthis will\b', l)
        and len(l.split()) >= 7
        and l not in clean
    ]
    clean += extra

    if len(clean) >= 2:
        return clean[:3]

    # Hard fallback: derive questions from the answer's first sentence
    # by asking the model ONE more time with a simpler prompt
    first_sent = re.split(r'(?<=[.!?])\s', answer_snippet)[0][:200]
    retry = _call(
        f"Based on this sentence from a document: \"{first_sent}\"\n"
        f"Write 3 specific follow-up questions a reader would ask. "
        f"Output ONLY a JSON array of 3 strings ending with ?",
        opts=FOLLOWUP_OPTS,
    )
    retry_candidates = _json_list(retry)
    retry_clean = [
        q.strip() for q in retry_candidates
        if len(q.strip()) > 12 and q.strip().endswith("?")
    ]
    clean += [q for q in retry_clean if q not in clean]

    return clean[:3] if clean else []


# ══════════════════════════════════════════════════════════════════════════════
# Chat Session & History persistence
# ══════════════════════════════════════════════════════════════════════════════

class ChatSession:
    def __init__(self, session_id=None, title="New Chat"):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.title      = title
        self.messages   = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.active_doc = None

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.updated_at = datetime.now().isoformat()
        if role == "user" and self.title == "New Chat":
            self.title = content[:50] + ("..." if len(content) > 50 else "")

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "title":      self.title,
            "messages":   self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "active_doc": self.active_doc,
        }

    @staticmethod
    def from_dict(d: dict) -> "ChatSession":
        s = ChatSession(d["session_id"], d["title"])
        s.messages   = d["messages"]
        s.created_at = d.get("created_at", "")
        s.updated_at = d.get("updated_at", "")
        s.active_doc = d.get("active_doc", None)
        return s


def _history_path() -> str:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    return os.path.join(HISTORY_DIR, "sessions.json")


def load_all_sessions() -> list:
    path = _history_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [ChatSession.from_dict(d) for d in data]
    except Exception:
        return []


def save_all_sessions(sessions: list):
    path = _history_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in sessions], f,
                  ensure_ascii=False, indent=2)


def delete_session(sessions: list, session_id: str) -> list:
    sessions = [s for s in sessions if s.session_id != session_id]
    save_all_sessions(sessions)
    return sessions


# ══════════════════════════════════════════════════════════════════════════════
# ChatEngine
# ══════════════════════════════════════════════════════════════════════════════

class ChatEngine:

    def __init__(self):
        self.db = ChatbotDB()

    def get_model_name(self) -> str:
        return MODEL

    def get_chunk_count(self) -> int:
        return self.db.total_chunks()

    def list_documents(self) -> dict:
        return self.db.list_all()

    def upload_file(self, path: str, progress_cb=None) -> tuple:
        try:
            n = self.db.add_file(path, progress_cb=progress_cb)
            return True, f"Indexed {n} chunks from {os.path.basename(path)}"
        except FileNotFoundError:
            return False, f"File not found: {path}"
        except Exception as e:
            return False, f"Failed: {e}"

    def delete_document(self, name: str) -> bool:
        return self.db.delete_file(name)

    def process_image_message(self, b64: str, mime: str, prompt: str,
                              history: list, stream_cb=None) -> str:
        hist = _hist_block(history)
        full_prompt = (
            SYSTEM + "\n"
            + hist
            + "The user has sent an image.\n"
            "Describe only what is DIRECTLY VISIBLE, then answer the question.\n"
            "Never invent text, numbers, or objects not visible.\n\n"
            f"USER QUESTION: {prompt}\n\nANSWER:\n"
        )
        try:
            full = []
            for chunk in ollama.generate(
                model=MODEL, prompt=full_prompt, images=[b64],
                stream=True, options={
                    "num_predict":    1500,
                    "temperature":    0.15,
                    "repeat_penalty": 1.1,
                    "top_k":          25,
                    "top_p":          0.85,
                    "num_ctx":        6144,
                },
            ):
                t = chunk.get("response", "")
                if t:
                    full.append(t)
                    if stream_cb:
                        stream_cb(t)
            return "".join(full).strip()
        except Exception as e:
            err = f"[Note: This model may not support image input. Error: {e}]"
            if stream_cb:
                stream_cb(err)
            return err

    def process_message(self, user_text: str, history: list,
                        stream_cb=None, followup_cb=None,
                        source_filter: str = None) -> str:
        raw = user_text.strip()
        if not raw:
            return ""
        low = raw.lower()

        # ── Greeting fast-path ────────────────────────────────────────────────
        is_casual = _is_pure_greeting(raw)

        if is_casual:
            docs = list(self.db.list_all().keys()) if self.db.total_chunks() > 0 else []
            if docs:
                doc_mention = (
                    f" I have '{docs[0]}' loaded and ready."
                    if len(docs) == 1
                    else f" I have {len(docs)} documents loaded."
                )
            else:
                doc_mention = " Upload a document and I'll help you find answers in it."

            prompt = (
                SYSTEM
                + f'\nThe user said: "{raw}". Reply warmly in 1-2 sentences.'
                + doc_mention
            )
            answer = _call_stream(prompt, opts=CASUAL_OPTS, callback=stream_cb)

            if not answer or len(answer.strip()) < 3:
                fallbacks = {
                    "hi":       "Hi there! I'm your document assistant.",
                    "hello":    "Hello! Ready to help with your documents.",
                    "hey":      "Hey! What can I help you with today?",
                    "thanks":   "Happy to help! Let me know if you need anything else.",
                    "thank you":"You're welcome! Feel free to ask more questions.",
                }
                answer = fallbacks.get(low, "Got it! What would you like to know?")
                if stream_cb:
                    stream_cb(answer)

            if followup_cb:
                has_docs_now = self.db.total_chunks() > 0
                def _casual_followups(has_d=has_docs_now, sf=source_filter):
                    if has_d:
                        docs_list = list(self.db.list_all().keys())
                        name = docs_list[0] if docs_list else "the document"
                        if sf:
                            followup_cb([
                                f"Summarize {sf}",
                                f"What are the key topics in {sf}?",
                                f"List the main sections in {sf}",
                            ])
                        else:
                            followup_cb([
                                f"Summarize {name}",
                                f"What are the key topics in {name}?",
                                "What questions can you answer from the documents?",
                            ])
                    else:
                        followup_cb([
                            "How do I upload a document?",
                            "What kind of questions can you answer?",
                            "What file types do you support?",
                        ])
                threading.Thread(target=_casual_followups, daemon=True).start()
            return answer

        # ── Normal pipeline ───────────────────────────────────────────────────
        has_docs  = self.db.total_chunks() > 0
        sources   = list(self.db.list_all().keys()) if has_docs else []

        page_req  = _page_num(raw) if has_docs else None
        list_mode = _is_list(raw)  if has_docs else False

        if not source_filter and has_docs:
            source_filter = _source_for(raw, sources)

        # ── QA Cache check: reuse previous answers for similar questions ───
        if has_docs and not page_req and not list_mode:
            cached = self.db.find_cached_qa(raw, source_filter)
            if cached:
                answer = cached["answer"]
                if stream_cb:
                    stream_cb(answer)
                if followup_cb:
                    def _cached_followups():
                        followups = _generate_followups(
                            raw, answer, list(history),
                            source_filter=source_filter,
                        )
                        followup_cb(followups)
                    threading.Thread(target=_cached_followups, daemon=True).start()
                return answer

        doc_directed      = False
        rewritten_queries: list = []

        if has_docs:
            # RULE: If a doc is focused, ALWAYS search it — never skip
            if source_filter:
                doc_directed = True
                rewritten_queries = _fast_rewrite(raw)
            elif page_req is not None or list_mode:
                doc_directed = True
                rewritten_queries = _fast_rewrite(raw)
            else:
                # No doc focused: use classifier
                doc_directed, rewritten_queries = _classify_and_rewrite(
                    raw, history, has_docs, source_filter=source_filter)
                # Even if classifier says GENERAL, do a quick doc search
                # to catch cases where docs actually have relevant content
                if not doc_directed:
                    quick = _search(
                        self.db, raw, history, source_filter=None,
                        precomputed_queries=_fast_rewrite(raw),
                    )
                    if quick and quick[0]["score"] > 0.25:
                        doc_directed = True
                        rewritten_queries = _fast_rewrite(raw)

        context: list = []
        if doc_directed:
            if page_req is not None:
                context = _get_page(self.db, page_req, source_filter)
                if not context:
                    msg = f"Nothing found on page {page_req}."
                    if stream_cb:
                        stream_cb(msg)
                    return msg
            elif list_mode:
                context = _search_section(
                    self.db, raw, history, source_filter,
                    precomputed_queries=rewritten_queries,
                )
                # Fallback: if section search fails, try normal search
                if not context:
                    context = _search(
                        self.db, raw, history, source_filter,
                        precomputed_queries=rewritten_queries,
                    )
            else:
                context = _search(
                    self.db, raw, history, source_filter,
                    precomputed_queries=rewritten_queries,
                )
                # Rerank: filter out noise, keep only the most relevant chunks
                if len(context) > 6:
                    context = _rerank(raw, context, n_keep=8)

        hist = _hist_block(history)
        cap  = CHUNK_CAP_LIST if list_mode else CHUNK_CAP_NORMAL

        # Few-shot memory: find similar past Q&A to guide answer style
        qa_examples = []
        if context and doc_directed:
            try:
                qa_examples = self.db.find_qa_examples(
                    raw, source_filter=source_filter, n=2
                )
            except Exception:
                qa_examples = []

        if context:
            ctx    = _build_ctx(context, cap=cap)
            prompt = _prompt_doc(raw, ctx, hist, list_mode,
                                qa_examples=qa_examples)
        elif doc_directed:
            # Searched docs but found nothing — acknowledge it, then general
            prompt = _prompt_no_ctx(raw, hist, source_filter=source_filter)
        else:
            prompt = _prompt_general(raw, hist)

        if stream_cb:
            answer = _call_stream(prompt, opts=ANSWER_OPTS, callback=stream_cb)
        else:
            answer = _call(prompt, opts=ANSWER_OPTS)

        # ── Store answer in QA cache for future reuse ─────────────────────
        if doc_directed and answer and not answer.startswith("[Error"):
            try:
                self.db.cache_qa(raw, answer, source_filter)
            except Exception:
                pass  # Caching is best-effort, never block the response

        # FIX #5: follow-ups run in background thread, don't block the response
        if followup_cb and answer and not answer.startswith("[Error"):
            captured_history  = list(history)
            captured_answer   = answer
            captured_question = raw
            captured_source   = source_filter

            next_page_hint = ""
            if list_mode and context:
                last_page = max(c["meta"].get("page", 0) for c in context[:cap])
                next_page_hint = f"What is on page {last_page + 2} of this section?"

            def _bg_followups():
                followups = _generate_followups(
                    captured_question,
                    captured_answer,
                    captured_history,
                    source_filter=captured_source,
                )
                if list_mode and next_page_hint:
                    followups = (
                        [next_page_hint]
                        + [f for f in followups if next_page_hint not in f]
                    )[:3]
                followup_cb(followups)

            threading.Thread(target=_bg_followups, daemon=True).start()

        return answer