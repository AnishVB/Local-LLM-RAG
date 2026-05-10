"""
chat_engine.py — RAG chat pipeline
CHANGES:
  - Issue 2: Numeric/financial value detection → lower threshold + wider retrieval
  - Issue 3: Follow-ups removed from UI chips; instead appended inline at end of answer
  - Issue 4: Multi-section ambiguity detection → bot asks user to pick a section
  - Issue 5: Vague follow-up detection ("explain in detail" etc.) → anchor to last Q&A only
"""
import os, sys, re, json, threading
import ollama
from dotenv import load_dotenv
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
from database import ChatbotDB

load_dotenv(os.path.join(os.path.dirname(current_dir), ".env"))

MODEL    = os.getenv("MODEL_NAME", "gemma4:e2b")
MAX_HIST = 8
NUM_GPU  = int(os.getenv("NUM_GPU", "1"))  # Set to 0 for CPU-only, 1+ if you have VRAM
NUM_THREAD = int(os.getenv("NUM_THREAD", "4"))  # 4 threads — let mlock/RAM do the heavy lifting, saves CPU for other processes

CHUNK_CAP_NORMAL = 12
CHUNK_CAP_LIST   = 30
HISTORY_DIR = os.path.join(os.path.dirname(current_dir), "chat_history")

_HARD_CASUAL_WORDS = {
    "hi","hello","hey","howdy","sup","yo","hiya","heyy","heyyy",
    "thanks","thank you","thx","ty","cheers","appreciated",
    "ok","okay","cool","alright","got it","sure","yep","yup","yeah","nope",
    "huh","hmm","oh","ah","lol","lmao","haha","bye","goodbye","cya","later","peace",
}
_HARD_CASUAL_PHRASES = {
    "how are you","how are you doing","how's it going","hows it going",
    "what's up","whats up","wassup","wazzup",
    "good morning","good afternoon","good evening","good night",
    "who are you","what are you","what can you do","see you","i see",
}

# ── Issue 5: Vague follow-up patterns that should anchor to last answer only ──
_VAGUE_FOLLOWUP_RE = re.compile(
    r'^(explain(\s+in\s+(more\s+)?(detail|depth|full))?'
    r'|elaborate(\s+on\s+that)?'
    r'|tell\s+me\s+more(\s+about\s+(it|that|this))?'
    r'|more\s+details?(\s+on\s+(it|that|this))?'
    r'|can\s+you\s+expand(\s+on\s+(it|that|this))?'
    r'|expand(\s+on\s+(it|that|this))?'
    r'|what\s+do\s+you\s+mean(\s+by\s+that)?'
    r'|clarify(\s+(that|this|it))?'
    r'|go\s+deeper(\s+(on|into)\s+(it|that|this))?'
    r'|give\s+me\s+more(\s+info(rmation)?)?(\s+on\s+(it|that|this))?'
    r'|i\s+want\s+to\s+know\s+more(\s+about\s+(it|that|this))?'
    r'|what\s+else(\s+can\s+you\s+tell\s+me)?'
    r'|and\s+what\s+about(\s+that)?'
    r'|in\s+(more\s+)?(detail|depth|full(\s+detail)?)'
    r'|full\s+detail(s)?'
    r'|more\s+info(rmation)?'
    r'|go\s+on'
    r'|summarize\s+(it|that|this)?'
    r'|summarise\s+(it|that|this)?'
    r'|give\s+(me\s+)?(a\s+)?summary(\s+of\s+(it|that|this))?'
    r'|tl;?dr'
    r')\??$',
    re.IGNORECASE
)

# ── Detail/depth request detection — user explicitly wants a thorough answer ──
_DETAIL_REQUEST_RE = re.compile(
    r'\b(in\s+(full\s+)?detail|in\s+depth|thoroughly|comprehensive(ly)?|elaborate|'
    r'explain\s+(fully|completely|thoroughly|in\s+depth|in\s+detail)|'
    r'full\s+(explanation|breakdown|detail|answer)|detailed\s+(explanation|answer|breakdown)|'
    r'give\s+me\s+(all|everything|the\s+full)|tell\s+me\s+everything|'
    r'what\s+exactly|break\s+(it\s+)?down|step\s+by\s+step)\b',
    re.IGNORECASE
)

# ── Issue 2: Numeric/financial query detection ────────────────────────────────
_NUMERIC_QUERY_RE = re.compile(
    r'\b(income|revenue|profit|loss|expense|cost|interest|tax|earn|return|rate|ratio'
    r'|amount|value|figure|number|total|sum|balance|asset|liabilit|equity|cash'
    r'|turnover|margin|ebitda|eps|dividend|capital|budget|forecast|target|growth'
    r'|quarter|annual|yearly|monthly|financial|fiscal|rupee|crore|lakh|million|billion'
    r'|percentage|percent|%|₹|\$|how\s+much|what\s+is\s+the\s+(value|amount|figure|number))\b',
    re.IGNORECASE
)

# ── Issue 4: Multi-section ambiguity detection ────────────────────────────────
# Thresholds are intentionally high — annual reports legitimately repeat topics
# across many pages. Only fire when sections are truly about different subjects.
_MIN_SECTIONS_FOR_AMBIGUITY = 8   # distinct non-boilerplate headings
_MIN_PAGES_FOR_AMBIGUITY    = 12  # distinct pages (very high — financial data spans the whole doc)


def _is_pure_greeting(text):
    low = text.strip().lower()
    if low in _HARD_CASUAL_PHRASES: return True
    words = low.split()
    if len(words) == 1 and low in _HARD_CASUAL_WORDS: return True
    if len(words) == 2 and low in _HARD_CASUAL_PHRASES: return True
    return False


def _is_vague_followup(text):
    """Issue 5: detect short vague follow-ups that should anchor to the last answer."""
    stripped = text.strip()
    if len(stripped.split()) > 12:
        return False   # longer questions are specific enough
    return bool(_VAGUE_FOLLOWUP_RE.match(stripped))


def _is_detail_request(text):
    """Detect when user explicitly asks for a detailed/comprehensive answer."""
    return bool(_DETAIL_REQUEST_RE.search(text))


def _is_numeric_query(text):
    """Issue 2: detect queries that are likely asking for numerical/financial values."""
    return bool(_NUMERIC_QUERY_RE.search(text))


def _context_is_ambiguous(context, query):
    """
    Issue 4: return a list of distinct section labels if context covers genuinely
    different topic sections, suggesting the user should pick which one they mean.

    NOT triggered for:
      - Numeric/financial queries (data legitimately appears across the whole doc)
      - Queries where headings are just boilerplate or page numbers
      - Queries where all headings are essentially the same topic
    Returns [] if not ambiguous.
    """
    if not context:
        return []

    # Never ask for clarification on financial/numeric queries — these intentionally
    # aggregate data from across the document (tables, notes, schedules, etc.)
    if _is_numeric_query(query):
        return []

    # Collect only non-boilerplate, substantively different headings
    real_headings = set()
    distinct_pages = set()
    for c in context:
        h = (c["meta"].get("heading") or "").strip()
        if h and not _is_boilerplate_heading(h):
            # Normalise to catch minor variations of the same heading
            real_headings.add(h.lower()[:60])
        distinct_pages.add(c["meta"].get("page", 0))

    # Only trigger if there are many genuinely different section headings
    if (len(real_headings) >= _MIN_SECTIONS_FOR_AMBIGUITY or
            len(distinct_pages) >= _MIN_PAGES_FOR_AMBIGUITY):
        sections = []
        seen = set()
        for c in context:
            h = (c["meta"].get("heading") or "").strip()
            p = c["meta"].get("page", 0) + 1
            # Skip boilerplate headings in the displayed list
            if h and _is_boilerplate_heading(h):
                continue
            label = h if h else f"Page {p}"
            if label not in seen:
                seen.add(label)
                sections.append({"label": label, "page": p})
        if len(sections) >= _MIN_SECTIONS_FOR_AMBIGUITY:
            return sections[:6]
    return []


# ── System prompts ────────────────────────────────────────────────────────────
DOC_SYSTEM = (
    "You are a knowledgeable assistant with access to uploaded documents. "
    "Match your answer length to the complexity of the question: "
    "simple or factual questions get short, direct answers (1-3 sentences); "
    "complex or multi-part questions get thorough, structured answers. "
    "Do NOT pad answers with unnecessary context, definitions, or implications unless asked. "
    "IMPORTANT: Always include ALL specific numbers, figures, percentages, dates, and monetary values "
    "mentioned in the excerpts — never omit numerical data. "
    "Cite pages like (p.N). If excerpts don't fully cover a point, say so, then supplement "
    "with your own knowledge in a section using a '---' divider followed by '## AI SUMMARY' as a heading. "
    "Never refuse to answer.\n"
)

GENERAL_SYSTEM = (
    "You are a helpful, knowledgeable assistant. "
    "Match your answer length to the question: simple questions get concise direct answers; "
    "complex questions get thorough answers with context and examples. "
    "Do NOT over-explain simple things. Be clear and get to the point. "
    "Use bullet points or numbered lists only when they genuinely aid clarity. "
    "Never say 'I cannot', 'I don't have access', or ask for documents to answer general questions.\n"
)

SYSTEM = GENERAL_SYSTEM

# ── LLM options ───────────────────────────────────────────────────────────────
# Lock model in RAM, enable memory-mapped I/O, increase context windows for better answers
_BASE_PERF = {
    "num_gpu": NUM_GPU, 
    "num_thread": NUM_THREAD, 
    "use_mmap": True,      # Memory-mapped I/O for efficient access
    "use_mlock": True,     # Lock model weights into RAM (no paging to disk) — great with 32GB
    "num_keep": 64,        # Keep 64 tokens cached across turns (faster follow-ups)
}

ANSWER_OPTS = {**_BASE_PERF, "num_predict": -1, "temperature": 0.35,
               "repeat_penalty": 1.08, "top_k": 40, "top_p": 0.90, "num_ctx": 120000}

CLASSIFY_OPTS = {**_BASE_PERF, "num_predict": 150, "temperature": 0.0,
                 "top_k": 10, "top_p": 0.80, "num_ctx": 8192}

CASUAL_OPTS   = {**_BASE_PERF, "num_predict": 150, "temperature": 0.55,
                 "top_k": 40, "top_p": 0.92, "num_ctx": 4096}

FOLLOWUP_OPTS = {**_BASE_PERF, "num_predict": 500, "temperature": 0.40,
                 "top_k": 30, "top_p": 0.90, "num_ctx": 8192}

RERANK_OPTS   = {**_BASE_PERF, "num_predict": 120, "temperature": 0.0,
                 "top_k": 10, "top_p": 0.80, "num_ctx": 16384}

# ── Intent patterns ───────────────────────────────────────────────────────────
_GENERAL_FORCE_RE = re.compile(
    r'\b(use your (own |general |training |base )?knowledge|from your training'
    r'|without (the |any )?document|not (from |using )(the |any )?doc'
    r'|general knowledge|on your own|ignore (the |any )?doc'
    r'|don\'?t use (the |any )?doc)\b', re.IGNORECASE)

_DOC_FORCE_RE = re.compile(
    r'\b(in the (pdf|document|file|report|upload)|from the (pdf|document|file|report|upload)'
    r'|according to (the )?(pdf|document|file|report|upload)'
    r'|what does the (pdf|document|file|report) say|is it mentioned'
    r'|does the (pdf|doc) (say|mention|contain|cover)'
    r'|from (our|my) (doc|file|pdf|report|upload)|in (our|my) (doc|file|pdf|report|upload)'
    r'|page \d+|pg\.?\s*\d+|chapter |section '
    r'|summarize (the |this |it\b)|summarise (the |this |it\b)'
    r'|the (pdf|document|doc|file|report)|this (pdf|document|doc|file|report)'
    r'|the title of|what is the title|what\'?s the title|title of (the |this )'
    r'|key topics|main sections|key points|this document|this file)\b', re.IGNORECASE)

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

_PAGE_RE       = [re.compile(p) for p in [r'\bpage\s+(\d+)\b', r'\bpg\.?\s*(\d+)\b']]
_LIST_KEYWORDS = [
    "list all","list the","list every","give me all","show all","enumerate",
    "all misconducts","all charges","all items","all rules","all violations",
    "all penalties","all types","all sections","what are all",
    "list each","every single","complete list",
]
_BP_PATTERNS = [
    re.compile(r'bharat\s+dynamics', re.IGNORECASE),
    re.compile(r'conduct.*discipline.*appeal', re.IGNORECASE),
    re.compile(r'corporate\s+office', re.IGNORECASE),
]
_BAD_FOLLOWUP_RE = re.compile(
    r'(i do not have access|i don\'?t have access|i cannot access|i can\'?t access'
    r'|document structure|multiple pages|as an ai|as a language model'
    r'|i\'?m not able to|i am not able to|i don\'?t know which page'
    r'|without more context|follow.up question|question \d+|here are)', re.IGNORECASE)


def _is_boilerplate_heading(h):
    return h and any(p.search(h) for p in _BP_PATTERNS)

def _ollama_error_msg(e: Exception) -> str:
    msg = str(e).lower()
    if any(kw in msg for kw in ("connection", "connect", "refused", "timeout", "unreachable")):
        return "[Error: Cannot reach Ollama — make sure `ollama serve` is running.]"
    return f"[Error: {e}]"

def _call(prompt, opts=None):
    try:
        return ollama.generate(model=MODEL, prompt=prompt, options=opts or CLASSIFY_OPTS).get("response", "").strip()
    except Exception as e: return _ollama_error_msg(e)

def _call_stream(prompt, opts=None, callback=None):
    try:
        full = []
        for chunk in ollama.generate(model=MODEL, prompt=prompt, stream=True, options=opts or ANSWER_OPTS):
            t = chunk.get("response", "")
            if t: full.append(t); callback and callback(t)
        return "".join(full).strip()
    except Exception as e:
        err = _ollama_error_msg(e); callback and callback(err); return err

def _json_list(raw):
    s, e = raw.find("["), raw.rfind("]")
    if s != -1 and e > s:
        try: return [str(x).strip() for x in json.loads(raw[s:e+1]) if str(x).strip()]
        except Exception: pass
    return re.findall(r'"([^"]{10,120})"', raw)

def _fast_rewrite(text):
    variants = [text]
    low = text.lower()
    stripped = re.sub(r'^(what|tell me|give me|show me|list|explain|describe|find|get|can you|please|i want to know about)\s+',
                      '', low, flags=re.IGNORECASE).strip()
    if stripped and stripped != low and len(stripped) > 3: variants.append(stripped)
    if "?" not in text and len(text.split()) <= 5:
        variants.append(f"information about {stripped or text}")
    return variants[:3]

def _classify_and_rewrite(text, history, has_docs, source_filter=None):
    low = text.lower()
    if _GENERAL_FORCE_RE.search(text): return False, []
    if _DOC_FORCE_RE.search(text):     return True, _fast_rewrite(text)
    if not has_docs:                    return False, []

    word_count = len(text.split())
    if word_count <= 3 and not _GENERAL_FORCE_RE.search(text):
        looks_general = (any(low.startswith(g) for g in _GEN_STARTS)
                         and not any(h in low for h in _DOC_HINTS)
                         and "?" in text and not re.search(r'\b(this|it|the|that)\b', low))
        if not looks_general: return True, _fast_rewrite(text)

    if source_filter and word_count <= 10:
        looks_general = (any(low.startswith(g) for g in _GEN_STARTS)
                         and not any(h in low for h in _DOC_HINTS)
                         and "?" in text and not re.search(r'\b(this|it|the|that)\b', low))
        if not looks_general: return True, _fast_rewrite(text)

    if any(h in low for h in _DOC_HINTS):  return True, _fast_rewrite(text)
    if any(low.startswith(g) for g in _GEN_STARTS) and not source_filter: return False, []

    hist = "\n".join(f'{"U" if m["role"]=="user" else "B"}: {m["content"][:80]}' for m in history[-3:])
    focused_ctx = f'Focused document: "{source_filter}". Lean toward DOC if plausible.\n' if source_filter else ""

    raw = _call(
        'Reply with JSON only — no preamble, no markdown, no explanation.\n'
        'Format: {"intent":"DOC","queries":["q1","q2","q3"]}\n'
        'intent = "DOC" if answerable only from an uploaded file, else "GENERAL".\n'
        'queries = 3 search-query rewrites of the question.\n'
        + focused_ctx + (f"History:\n{hist}\n" if hist else "")
        + f"Question: {text}\nJSON:"
    )
    try:
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e > s:
            obj = json.loads(raw[s:e+1])
            is_doc = str(obj.get("intent", "")).strip().upper() == "DOC"
            qs = [str(q).strip() for q in obj.get("queries", []) if str(q).strip()]
            return is_doc, qs[:3]
    except Exception: pass
    return False, []

def _is_list(text):
    return any(k in text.lower() for k in _LIST_KEYWORDS)

def _page_num(text):
    for pat in _PAGE_RE:
        m = pat.search(text.lower())
        if m: return max(1, int(m.group(1)))
    return None

def _source_for(text, sources):
    low = text.lower()
    for src in sources:
        if src.lower() in low or os.path.splitext(src)[0].lower() in low: return src
    return None

# ── Retrieval helpers ─────────────────────────────────────────────────────────
def _get_page(db, user_page, src=None):
    ip = user_page - 1
    return [{"text": c["text"], "meta": c["meta"], "score": 1.0}
            for c in db.chunks if c["meta"].get("page") == ip and (src is None or c["meta"]["source"] == src)]

def _get_section(db, anchor_chunks, source_filter=None):
    if not anchor_chunks: return []
    src = source_filter or anchor_chunks[0]["meta"]["source"]
    all_chunks = sorted((c for c in db.chunks if c["meta"]["source"] == src),
                        key=lambda c: (c["meta"].get("page", 0), c["meta"].get("chunk_id", 0)))
    if not all_chunks: return anchor_chunks

    anchor_keys = {(c["meta"].get("page"), c["meta"].get("chunk_id")) for c in anchor_chunks}
    start_idx = next((i for i, c in enumerate(all_chunks)
                      if (c["meta"].get("page"), c["meta"].get("chunk_id")) in anchor_keys), None)
    if start_idx is None: return anchor_chunks

    anchor_heading = all_chunks[start_idx]["meta"].get("heading", "").strip().lower()
    walk_start = start_idx
    if anchor_heading:
        for i in range(start_idx - 1, max(start_idx - 9, -1), -1):
            h = all_chunks[i]["meta"].get("heading", "").strip().lower()
            if h == anchor_heading or not h or _is_boilerplate_heading(h): walk_start = i
            else: break

    section = []
    for i in range(walk_start, len(all_chunks)):
        c = all_chunks[i]; h = c["meta"].get("heading", "").strip().lower()
        if h == anchor_heading or not h or _is_boilerplate_heading(h):
            section.append({"text": c["text"], "meta": c["meta"], "score": 1.0})
        else: break
        if len(section) >= 50: break
    return section if section else anchor_chunks

def _search(db, text, history, source_filter=None, precomputed_queries=None, numeric_boost=False):
    queries = [text] + (precomputed_queries or _fast_rewrite(text))
    # Issue 2: for numeric queries, lower min_score and raise max_per_page to catch
    # table rows and short numeric lines that embed with lower cosine similarity
    if numeric_boost:
        return db.multi_query(queries, n_final=20, min_score=0.08, max_per_page=6,
                              source_filter=source_filter)
    return db.multi_query(queries, n_final=15, min_score=0.12, max_per_page=4,
                          source_filter=source_filter)

def _search_section(db, text, history, source_filter=None, precomputed_queries=None):
    queries = [text] + (precomputed_queries or _fast_rewrite(text))
    anchors = db.multi_query(queries, n_final=5, min_score=0.15, max_per_page=3, source_filter=source_filter)
    if not anchors: return []
    return _get_section(db, anchors, source_filter)

def _rerank(query, chunks, n_keep=8):
    if len(chunks) <= n_keep: return chunks
    previews = [f"{i}: {c['text'][:200].replace(chr(10),' ').strip()}" for i, c in enumerate(chunks)]
    raw = _call(
        f"You are a relevance judge. Given a QUESTION and numbered TEXT excerpts, "
        f"return ONLY a JSON array of the {n_keep} most relevant excerpt numbers "
        f"ordered from most to least relevant. No explanation, just the JSON array.\n\n"
        f"QUESTION: {query}\n\nEXCERPTS:\n" + "\n".join(previews) + "\n\nJSON array:",
        opts=RERANK_OPTS,
    )
    try:
        s, e = raw.find("["), raw.rfind("]")
        if s != -1 and e > s:
            indices = [int(i) for i in json.loads(raw[s:e+1]) if 0 <= int(i) < len(chunks)]
            if indices:
                seen, unique = set(), []
                for idx in indices:
                    if idx not in seen: seen.add(idx); unique.append(idx)
                return [chunks[i] for i in unique[:n_keep]]
    except Exception: pass
    return chunks[:n_keep]

# ── Context / prompt builders ─────────────────────────────────────────────────
def _hist_block(history, max_pairs=None):
    if not history: return ""
    limit = (max_pairs * 2) if max_pairs else (MAX_HIST * 2)
    lines = [f'{"User" if m["role"]=="user" else "Assistant"}: {m["content"][:250]}'
             for m in history[-limit:]]
    return "PRIOR CONVERSATION:\n" + "\n".join(lines) + "\n\n"

def _build_ctx(chunks, cap=CHUNK_CAP_NORMAL):
    by_src = defaultdict(list)
    for c in chunks[:cap]: by_src[c["meta"]["source"]].append(c)
    parts = []
    for src, cs in by_src.items():
        cs.sort(key=lambda x: (x["meta"].get("page", 0), x["meta"].get("chunk_id", 0)))
        parts.append(f"--- {src} ---")
        for c in cs: parts.append(f"[p.{c['meta'].get('page', 0) + 1}] {c['text']}")
    return "\n".join(parts)

def _prompt_doc(q, ctx, hist, is_list=False, qa_examples=None, is_numeric=False, is_detail=False):
    numeric_instruction = (
        "CRITICAL: The user is asking about specific numbers, figures, or financial values. "
        "You MUST extract and present ALL numerical data found in the excerpts: "
        "amounts, percentages, dates, ratios, totals — do not skip any. "
        "Present them clearly, ideally in a table or structured list.\n"
        if is_numeric else ""
    )
    if is_list:
        task = (
            "Extract every relevant item from the excerpts. Copy exact wording. Number each item. "
            "Cite page as (p.N). Cover ALL excerpts. After listing, give a brief count. "
            "If excerpts are from multiple documents, group by document name.\n"
        )
    elif is_detail:
        task = (
            "Give a DETAILED and COMPREHENSIVE answer — the user has explicitly asked for depth. Follow this order:\n"
            "1. Use exact terms, figures, names, dates from the excerpts and cite with (p.N).\n"
            "2. Expand on ALL relevant aspects: definitions, procedures, conditions, exceptions, examples, implications.\n"
            "3. If the excerpts cover the topic partially or not at all, say so, then add a section:\n"
            "   \n---\n## AI SUMMARY\n"
            "   (your general knowledge answer here)\n"
            "4. If excerpts come from multiple documents, group information under each document name.\n"
        )
    else:
        task = (
            "Answer the question directly and concisely. Follow this order:\n"
            "1. If the question is simple or factual, answer in 1-3 sentences — do NOT pad.\n"
            "2. If the question is complex, give a structured answer covering all relevant aspects.\n"
            "3. Always include exact terms, figures, names, dates from the excerpts and cite with (p.N).\n"
            "4. If the excerpts don't fully cover the topic, say so, then add a section:\n"
            "   \n---\n## AI SUMMARY\n"
            "   (your general knowledge answer here)\n"
            "5. If excerpts come from multiple documents, group information under each document name.\n"
        )
    examples_block = ""
    if qa_examples:
        parts = [f"SIMILAR PAST Q: {ex['question'][:200]}\nPAST ANSWER: {ex['answer'][:500]}\n" for ex in qa_examples[:2]]
        examples_block = ("REFERENCE — Here is how similar questions were answered before. "
                          "Use the same depth, style, and citation approach:\n" + "\n".join(parts) + "\n")
    return (DOC_SYSTEM + "\n" + numeric_instruction + task + "\n" + examples_block + hist
            + f"DOCUMENT EXCERPTS:\n{ctx}\n\nQUESTION: {q}\n\n"
            + ("EXTRACTED LIST:\n" if is_list else "ANSWER:\n"))

def _prompt_no_ctx(q, hist, source_filter=None):
    preamble = (
        f"I searched the document '{source_filter}' thoroughly but found no directly relevant excerpts.\n"
        if source_filter else
        "I searched all indexed documents but found no directly relevant excerpts.\n"
    )
    return (GENERAL_SYSTEM + "\n" + preamble
            + "Tell the user briefly that this wasn't found in the document(s). "
            "Then answer from your general knowledge under a section:\n"
            "\n---\n## AI SUMMARY\n(your answer here)\n"
            "Keep it concise unless the topic genuinely requires depth.\n\n"
            + hist + f"QUESTION: {q}\n\nANSWER:\n")

def _prompt_general(q, hist):
    return (GENERAL_SYSTEM + "\nAnswer directly. Keep it concise for simple questions; "
            "go deeper only if the question genuinely warrants it.\n\n"
            + hist + f"QUESTION: {q}\n\nANSWER:\n")

def _prompt_vague_followup(q, last_question, last_answer, ctx, source_filter=None):
    """
    Issue 5: For vague follow-ups like 'explain in detail', anchor strictly to
    the previous Q&A pair — do NOT include full conversation history.
    Always gives a detailed expansion since the user is asking for more depth.
    """
    doc_note = f"Document in focus: {source_filter}\n" if source_filter else ""
    ctx_block = f"DOCUMENT EXCERPTS (same as previous answer):\n{ctx}\n\n" if ctx else ""
    return (
        DOC_SYSTEM + "\n"
        + doc_note
        + "The user wants a DETAILED, COMPREHENSIVE expansion of the IMMEDIATELY PREVIOUS answer ONLY.\n"
        "Do NOT re-summarise the conversation. Focus EXCLUSIVELY on the previous Q&A pair below.\n"
        "Expand with depth: add definitions, conditions, exceptions, step-by-step procedures, examples,\n"
        "implications, and any figures or citations from the document excerpts.\n"
        "This is an explicit request for detail — do NOT be brief.\n\n"
        + ctx_block
        + f"PREVIOUS USER QUESTION: {last_question}\n\n"
        f"PREVIOUS ANSWER (expand on this):\n{last_answer[:1500]}\n\n"
        f"USER REQUEST: {q}\n\n"
        "DETAILED EXPANSION:\n"
    )


# ── Issue 3: Inline follow-up suggestion generation ───────────────────────────
def _generate_inline_suggestions(user_text, answer, context, source_filter=None):
    """
    Issue 3: Generate 2-3 contextual follow-up suggestions to append inline
    at the end of the bot's answer. Returns a formatted string to append,
    or empty string if no good suggestions can be made.
    """
    answer_snippet = answer[:800].strip()
    doc_ctx = f"Document: {source_filter}\n" if source_filter else ""

    # Build a richer context hint from retrieved chunks
    chunk_hint = ""
    if context:
        topics = set()
        for c in context[:6]:
            h = (c["meta"].get("heading") or "").strip()
            if h and not _is_boilerplate_heading(h):
                topics.add(h)
        if topics:
            chunk_hint = "Other sections in the document covering related topics: " + ", ".join(list(topics)[:4]) + "\n"

    prompt = (
        "You generate contextual follow-up suggestions for a document Q&A chatbot.\n"
        "Output ONLY a raw JSON object: {\"suggestions\": [\"s1\", \"s2\", \"s3\"]} or {\"suggestions\": []}\n"
        "Rules:\n"
        "- Each suggestion must be a SHORT PHRASE (3-8 words), not a full question\n"
        "- Must reference SPECIFIC data points, numbers, or topics from the Answer\n"
        "- Must explore different angles: comparisons, trends, definitions, breakdowns\n"
        "- If you cannot think of 2 genuinely useful suggestions grounded in the answer, return {\"suggestions\": []}\n"
        "- NEVER suggest vague things like 'tell me more' or 'explain further'\n"
        "- Phrase as things the user might want to know: 'Compare with previous year', 'Break down by segment'\n\n"
        "GOOD EXAMPLES: [\"Compare with previous year\", \"Break down interest income by segment\", "
        "\"View quarterly trend\"]\n"
        "BAD EXAMPLES: [\"Tell me more\", \"Explain in detail\", \"What else?\"]\n\n"
        + doc_ctx + chunk_hint
        + f"User asked: {user_text[:200]}\n\n"
        f"Answer given:\n{answer_snippet}\n\n"
        "JSON:"
    )
    raw = _call(prompt, opts=FOLLOWUP_OPTS)
    try:
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e > s:
            obj = json.loads(raw[s:e+1])
            suggestions = [str(x).strip() for x in obj.get("suggestions", []) if str(x).strip()]
            # Filter out bad ones
            clean = [sg for sg in suggestions
                     if 2 < len(sg.split()) <= 10
                     and not _BAD_FOLLOWUP_RE.search(sg)
                     and not re.search(r'\bmore\b|\bdetail\b|\bexplain\b|\btell me\b', sg, re.IGNORECASE)]
            if len(clean) >= 2:
                return clean[:3]
    except Exception:
        pass
    return []


def _format_inline_suggestions(suggestions):
    """
    Issue 3: Format suggestions as a small appended block at end of answer.
    Returns markdown string.
    """
    if not suggestions:
        return ""
    lines = ["", "---", "**Want to explore further?**"]
    for sg in suggestions:
        lines.append(f"- {sg}")
    return "\n".join(lines)


# ── Issue 4: Multi-section ambiguity message ──────────────────────────────────
def _format_ambiguity_message(query, sections):
    """
    Issue 4: Build a clarification message when the document has many different
    sections covering the same topic.
    """
    lines = [
        f"I found **{len(sections)} different sections** in the document that mention this topic. "
        "Which one would you like me to focus on?\n"
    ]
    for i, sec in enumerate(sections, 1):
        lines.append(f"{i}. **{sec['label']}** (p.{sec['page']})")
    lines.append(
        "\nOr say **\"all of them\"** and I'll summarise across all sections."
    )
    return "\n".join(lines)


# ── Chat Session & History ────────────────────────────────────────────────────
class ChatSession:
    def __init__(self, session_id=None, title="New Chat"):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.title      = title
        self.messages   = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.active_doc = None

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        self.updated_at = datetime.now().isoformat()
        if role == "user" and self.title == "New Chat":
            self.title = content[:50] + ("..." if len(content) > 50 else "")

    def to_dict(self):
        return {"session_id": self.session_id, "title": self.title, "messages": self.messages,
                "created_at": self.created_at, "updated_at": self.updated_at, "active_doc": self.active_doc}

    @staticmethod
    def from_dict(d):
        s = ChatSession(d["session_id"], d["title"])
        s.messages   = d["messages"]
        s.created_at = d.get("created_at", "")
        s.updated_at = d.get("updated_at", "")
        s.active_doc = d.get("active_doc", None)
        return s


# Thread-safe lock for chat history file operations (multi-user)
_history_lock = threading.Lock()

def _history_path():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    return os.path.join(HISTORY_DIR, "sessions.json")

def load_all_sessions():
    path = _history_path()
    with _history_lock:
        if not os.path.exists(path): return []
        try:
            with open(path, "r", encoding="utf-8") as f: data = json.load(f)
            return [ChatSession.from_dict(d) for d in data]
        except Exception: return []

def save_all_sessions(sessions):
    with _history_lock:
        with open(_history_path(), "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in sessions], f, ensure_ascii=False, indent=2)

def delete_session(sessions, session_id):
    sessions = [s for s in sessions if s.session_id != session_id]
    save_all_sessions(sessions); return sessions


def _find_past_answer(query, current_history):
    """
    Cross-session history lookup: search all saved sessions for a question that
    closely matches the current query. Returns the past answer if found, else None.
    Skips messages already in the current conversation to avoid self-matches.
    Uses simple token overlap (Jaccard) — fast, no model call needed.
    """
    try:
        all_sessions = load_all_sessions()
    except Exception:
        return None

    q_tokens = set(re.sub(r'[^\w\s]', '', query.lower()).split())
    if len(q_tokens) < 3:
        return None   # too short to match meaningfully

    # Build set of already-known questions to avoid returning current-session hits
    known_q = {re.sub(r'[^\w\s]', '', m["content"].lower())
               for m in current_history if m["role"] == "user"}

    best_score, best_answer = 0.0, None
    for session in all_sessions:
        msgs = session.messages
        for i, msg in enumerate(msgs):
            if msg["role"] != "user":
                continue
            past_q_raw = msg["content"].strip()
            past_q_norm = re.sub(r'[^\w\s]', '', past_q_raw.lower())
            if past_q_norm in known_q:
                continue   # same session or already seen
            # Find the assistant reply that follows this message
            reply = next((msgs[j]["content"] for j in range(i + 1, len(msgs))
                          if msgs[j]["role"] == "assistant"), None)
            if not reply or len(reply.strip()) < 30:
                continue
            p_tokens = set(past_q_norm.split())
            if not p_tokens:
                continue
            union = q_tokens | p_tokens
            jaccard = len(q_tokens & p_tokens) / len(union) if union else 0.0
            if jaccard > best_score:
                best_score = jaccard
                best_answer = reply

    # Only reuse if overlap is strong enough (≥ 0.65 = very similar question)
    if best_score >= 0.65 and best_answer:
        return best_answer
    return None


# ── Issue 1: Auto-delete empty sessions older than 1 hour ────────────────────
def purge_empty_sessions(sessions):
    """
    Remove sessions that have no messages and were created more than 1 hour ago.
    Returns the cleaned session list.
    """
    cutoff = datetime.now().timestamp() - 3600   # 1 hour in seconds
    cleaned = []
    for s in sessions:
        if s.messages:
            cleaned.append(s)
            continue
        # Parse creation time; if unparseable keep the session to be safe
        try:
            created_ts = datetime.fromisoformat(s.created_at).timestamp()
        except Exception:
            cleaned.append(s)
            continue
        if created_ts > cutoff:
            cleaned.append(s)   # still within grace period
        # else: drop it silently
    return cleaned


# ── ChatEngine ────────────────────────────────────────────────────────────────
class ChatEngine:
    def __init__(self):            self.db = ChatbotDB()
    def get_model_name(self):      return MODEL
    def get_chunk_count(self):     return self.db.total_chunks()
    def list_documents(self):      return self.db.list_all()
    def delete_document(self, n):  return self.db.delete_file(n)

    def upload_file(self, path, progress_cb=None):
        try:
            n = self.db.add_file(path, progress_cb=progress_cb)
            return True, f"Indexed {n} chunks from {os.path.basename(path)}"
        except FileNotFoundError: return False, f"File not found: {path}"
        except Exception as e:    return False, f"Failed: {e}"

    def process_message(self, user_text, history, stream_cb=None, followup_cb=None,
                        source_filter=None, last_answer=""):
        raw = user_text.strip()
        if not raw: return "", []
        low = raw.lower()

        # ── Greeting fast-path ────────────────────────────────────────────────
        if _is_pure_greeting(raw):
            docs = list(self.db.list_all().keys()) if self.db.total_chunks() > 0 else []
            doc_mention = (
                f" I have '{docs[0]}' loaded and ready." if len(docs) == 1
                else f" I have {len(docs)} documents loaded." if docs
                else " Upload a document and I'll help you find answers in it."
            )
            answer = _call_stream(
                SYSTEM + f'\nThe user said: "{raw}". Reply warmly in 1-2 sentences.' + doc_mention,
                opts=CASUAL_OPTS, callback=stream_cb,
            )
            if not answer or len(answer.strip()) < 3:
                fallbacks = {"hi": "Hi there! I'm your document assistant.",
                             "hello": "Hello! Ready to help with your documents.",
                             "hey": "Hey! What can I help you with today?",
                             "thanks": "Happy to help! Let me know if you need anything else.",
                             "thank you": "You're welcome! Feel free to ask more questions."}
                answer = fallbacks.get(low, "Got it! What would you like to know?")
                stream_cb and stream_cb(answer)

            # Issue 3: no follow-up chips for greetings
            followup_cb and followup_cb([])
            return answer, []

        # ── Normal pipeline ───────────────────────────────────────────────────
        has_docs      = self.db.total_chunks() > 0
        sources       = list(self.db.list_all().keys()) if has_docs else []
        page_req      = _page_num(raw) if has_docs else None
        list_mode     = _is_list(raw) if has_docs else False
        is_numeric    = _is_numeric_query(raw)   # Issue 2
        is_detail     = _is_detail_request(raw)  # User explicitly asked for detail

        if not source_filter and has_docs:
            source_filter = _source_for(raw, sources)

        # ── Issue 5: Vague follow-up detection ───────────────────────────────
        # IMPORTANT: check this BEFORE is_detail — short phrases like "in detail"
        # or "full detail" are follow-ups to the previous answer, not standalone
        # detail requests. Vague follow-up always wins when a previous answer exists.
        last_user_q, last_bot_ans = "", last_answer
        for m in reversed(history):
            if m["role"] == "user" and not last_user_q:
                last_user_q = m["content"]
            if m["role"] == "assistant" and not last_bot_ans:
                last_bot_ans = m["content"]
            if last_user_q and last_bot_ans:
                break

        is_vague_fu = _is_vague_followup(raw) and bool(last_bot_ans)
        # If it's a vague follow-up, clear is_detail — user is asking to expand
        # the PREVIOUS answer, not asking a new standalone detailed question
        if is_vague_fu:
            is_detail = False

        if is_vague_fu:
            # Re-search the doc using the PREVIOUS question as the query anchor
            anchor_query = last_user_q or raw
            ctx_chunks = _search(self.db, anchor_query, [], source_filter=source_filter,
                                 precomputed_queries=_fast_rewrite(anchor_query),
                                 numeric_boost=_is_numeric_query(anchor_query))
            ctx = _build_ctx(ctx_chunks) if ctx_chunks else ""
            prompt = _prompt_vague_followup(raw, last_user_q, last_bot_ans, ctx, source_filter)
            answer = _call_stream(prompt, opts=ANSWER_OPTS, callback=stream_cb) if stream_cb else _call(prompt, opts=ANSWER_OPTS)

            # Inline suggestions for vague follow-ups too
            if followup_cb and answer and not answer.startswith("[Error"):
                def _bg_vague_fu():
                    suggestions = _generate_inline_suggestions(anchor_query, answer, ctx_chunks, source_filter)
                    followup_cb(suggestions)
                threading.Thread(target=_bg_vague_fu, daemon=True).start()
            else:
                followup_cb and followup_cb([])

            return answer, ctx_chunks

        # ── QA cache check (current session) ─────────────────────────────────
        if has_docs and not page_req and not list_mode and not is_detail:
            cached = self.db.find_cached_qa(raw, source_filter)
            if cached:
                answer = cached["answer"]
                stream_cb and stream_cb(answer)
                followup_cb and followup_cb([])   # Issue 3: no chips on cache hits
                return answer, []

        # ── Cross-session history lookup ──────────────────────────────────────
        # If the same/similar question was answered in a previous session, reuse it.
        # Skip when user explicitly wants detail (they want a fresh thorough answer).
        if not is_detail and not page_req and not list_mode:
            past_answer = _find_past_answer(raw, history)
            if past_answer:
                stream_cb and stream_cb(past_answer)
                followup_cb and followup_cb([])
                return past_answer, []

        doc_directed, rewritten_queries = False, []

        if has_docs:
            if source_filter or page_req is not None or list_mode or is_numeric:
                doc_directed = True; rewritten_queries = _fast_rewrite(raw)
            else:
                doc_directed, rewritten_queries = _classify_and_rewrite(raw, history, has_docs, source_filter)
                if not doc_directed:
                    quick = _search(self.db, raw, history, source_filter=None,
                                    precomputed_queries=_fast_rewrite(raw))
                    if quick and quick[0]["score"] > 0.25:
                        doc_directed = True; rewritten_queries = _fast_rewrite(raw)

        context = []
        if doc_directed:
            if page_req is not None:
                context = _get_page(self.db, page_req, source_filter)
                if not context:
                    msg = f"Nothing found on page {page_req}."; stream_cb and stream_cb(msg); return msg, []
            elif list_mode:
                context = _search_section(self.db, raw, history, source_filter, rewritten_queries)
                if not context: context = _search(self.db, raw, history, source_filter, rewritten_queries)
            else:
                # Issue 2: pass numeric_boost so financial queries get wider retrieval
                context = _search(self.db, raw, history, source_filter, rewritten_queries,
                                  numeric_boost=is_numeric)
                if len(context) > 6: context = _rerank(raw, context, n_keep=8)

        # ── Issue 4: Multi-section ambiguity check ────────────────────────────
        if doc_directed and context and not page_req and not list_mode:
            ambiguous_sections = _context_is_ambiguous(context, raw)
            if ambiguous_sections:
                clarify_msg = _format_ambiguity_message(raw, ambiguous_sections)
                stream_cb and stream_cb(clarify_msg)
                followup_cb and followup_cb([])
                return clarify_msg, context

        hist = _hist_block(history)
        cap  = CHUNK_CAP_LIST if list_mode else CHUNK_CAP_NORMAL
        qa_examples = []
        if context and doc_directed:
            try: qa_examples = self.db.find_qa_examples(raw, source_filter=source_filter, n=2)
            except Exception: pass

        if context:
            ctx    = _build_ctx(context, cap=cap)
            prompt = _prompt_doc(raw, ctx, hist, list_mode, qa_examples=qa_examples, is_numeric=is_numeric, is_detail=is_detail)
        elif doc_directed:
            prompt = _prompt_no_ctx(raw, hist, source_filter=source_filter)
        else:
            prompt = _prompt_general(raw, hist)

        answer = _call_stream(prompt, opts=ANSWER_OPTS, callback=stream_cb) if stream_cb else _call(prompt, opts=ANSWER_OPTS)

        if doc_directed and answer and not answer.startswith("[Error"):
            try: self.db.cache_qa(raw, answer, source_filter)
            except Exception: pass

        # ── Issue 3: Inline suggestions (background thread) ───────────────────
        # Instead of UI chips, generate suggestions and pass them back via followup_cb
        # The frontend will append them inline to the answer text.
        if followup_cb and answer and not answer.startswith("[Error"):
            captured_answer   = answer
            captured_question = raw
            captured_source   = source_filter
            captured_context  = context
            next_page_hint    = ""
            if list_mode and context:
                last_page = max(c["meta"].get("page", 0) for c in context[:cap])
                next_page_hint = f"Continue to page {last_page + 2}"

            def _bg_suggestions():
                suggestions = _generate_inline_suggestions(
                    captured_question, captured_answer, captured_context,
                    source_filter=captured_source
                )
                if list_mode and next_page_hint:
                    suggestions = ([next_page_hint] + [s for s in suggestions if next_page_hint not in s])[:3]
                followup_cb(suggestions)
            threading.Thread(target=_bg_suggestions, daemon=True).start()
        else:
            followup_cb and followup_cb([])

        return answer, context