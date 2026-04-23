import os, sys, re, json, ollama
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
from database import ChatbotDB

load_dotenv(os.path.join(os.path.dirname(current_dir), ".env"))
MODEL    = os.getenv("MODEL_NAME", "gemma:2b")
console  = Console()
MAX_HIST = 6

CHUNK_CAP_NORMAL = 10
CHUNK_CAP_LIST   = 30

GREETINGS = {"hi","hello","hey","howdy","sup","yo","hiya","good morning",
             "good afternoon","good evening","how are you","what's up",
             "whats up","thanks","thank you","thx","ty","bye","goodbye"}

HELP = f"""
[bold cyan]Commands:[/bold cyan]
  [yellow]upload <path>[/yellow]  upload a PDF or TXT
  [yellow]list[/yellow]           show indexed documents
  [yellow]delete <n>[/yellow]  remove a document
  [yellow]q / quit[/yellow]       exit
[dim]Model: {MODEL} · Type 1/2/3 for follow-up questions[/dim]
"""

# ── LLM ───────────────────────────────────────────────────────────────────────

def _call(prompt: str, tokens=512, temp=0.0, stream=False) -> str:
    opts = {"num_predict": tokens, "temperature": temp}
    try:
        if stream:
            parts = []
            for chunk in ollama.generate(model=MODEL, prompt=prompt, stream=True, options=opts):
                t = chunk["response"]
                console.out(t, end="", highlight=False)
                parts.append(t)
            console.print()
            return "".join(parts).strip()
        return ollama.generate(model=MODEL, prompt=prompt, options=opts)["response"].strip()
    except ollama.ResponseError as e:
        console.print(f"[red]Ollama error:[/red] {e}")
        return ""
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return ""

def _json_list(raw: str) -> list:
    s, e = raw.find("["), raw.rfind("]")
    if s != -1 and e > s:
        try:
            return [str(x).strip() for x in json.loads(raw[s:e+1]) if str(x).strip()]
        except Exception:
            pass
    return []

# ── intent detection ──────────────────────────────────────────────────────────

_DOC_HINTS = [
    "in the pdf","in the document","in the file","in the report","according to",
    "from the pdf","from the document","from the report","from the file",
    "page ","pg ","chapter ","section ","uploaded","summarize the","summarise the",
    "what does the pdf","what does the document","does the pdf","is it mentioned",
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

def _is_doc(text: str, history: list, has_docs: bool) -> bool:
    if not has_docs: return False
    low = text.lower()
    if any(h in low for h in _DOC_HINTS): return True
    if any(low.startswith(g) for g in _GEN_STARTS): return False
    hist = "\n".join(f'{"U" if m["role"]=="user" else "B"}: {m["content"][:100]}'
                     for m in history[-4:])
    r = _call(
        "Reply DOC or GENERAL only. No other words.\n"
        "DOC = specifically about an uploaded file's content.\n"
        "GENERAL = general knowledge any person would know.\n"
        "Default to GENERAL when unsure.\n"
        "Examples: 'mass of TON 618'->GENERAL  'list the charges'->DOC  "
        "'what is an OS'->GENERAL  'what does the report say'->DOC\n"
        + (f"History:\n{hist}\n" if hist else "")
        + f"Question: {text}\nAnswer:", tokens=5)
    return r.strip().upper() == "DOC"

def _is_list(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in [
        "list all","list the","list every","give me all","show all","enumerate",
        "all misconducts","all charges","all items","all rules","all violations",
        "all penalties","all types","all sections","what are the","what are all",
    ])

def _page_num(text: str):
    for pat in [r'\bpage\s+(\d+)\b', r'\bpg\.?\s*(\d+)\b']:
        m = re.search(pat, text.lower())
        if m: return max(0, int(m.group(1)) - 1)
    return None

def _source_for(text: str, sources: list) -> str | None:
    low = text.lower()
    for src in sources:
        stem = os.path.splitext(src)[0].lower()
        if src.lower() in low or stem in low:
            return src
    return None

# ── retrieval ─────────────────────────────────────────────────────────────────

def _get_page(db, page: int, src=None) -> list:
    return [{"text": c["text"], "meta": c["meta"], "score": 1.0}
            for c in db.chunks
            if c["meta"].get("page") == page
            and (src is None or c["meta"]["source"] == src)]

def _is_boilerplate_heading(h: str) -> bool:
    """
    Return True if a heading string looks like a repeating page header rather
    than a genuine section heading.  These must never trigger a section-stop.

    Matches things like:
      "bharat dynamics limited"
      "conduct discipline and appeal rules 2020"
      "corporate office"
    """
    if not h:
        return False
    import re as _re
    _BP = [
        _re.compile(r'bharat\s+dynamics', _re.IGNORECASE),
        _re.compile(r'conduct.*discipline.*appeal', _re.IGNORECASE),
        _re.compile(r'corporate\s+office', _re.IGNORECASE),
    ]
    return any(p.search(h) for p in _BP)


def _get_section(db, anchor_chunks: list, source_filter=None) -> list:
    """
    Given a small set of high-scoring anchor chunks (the ones cosine search
    found), walk forward through the document in reading order and keep pulling
    chunks until the heading clearly changes to a *different real* heading.

    Key fix: repeating page-header boilerplate (e.g. "BHARAT DYNAMICS LIMITED
    (CONDUCT, DISCIPLINE & APPEAL) RULES, 2020") must NOT be treated as a new
    section heading — otherwise the walk stops at every page break.
    """
    if not anchor_chunks:
        return []

    src = source_filter or anchor_chunks[0]["meta"]["source"]

    # Full chunk list for this source, in reading order
    all_chunks = sorted(
        [c for c in db.chunks if c["meta"]["source"] == src],
        key=lambda c: (c["meta"].get("page", 0), c["meta"].get("chunk_id", 0))
    )
    if not all_chunks:
        return anchor_chunks

    # Find the position of the earliest anchor in the reading-order list
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

    # The heading of the section we landed in
    anchor_heading = all_chunks[start_idx]["meta"].get("heading", "").strip().lower()
    anchor_page    = all_chunks[start_idx]["meta"].get("page", 0)

    # Walk backwards up to 5 chunks to catch the section start
    walk_start = start_idx
    if anchor_heading:
        for i in range(start_idx - 1, max(start_idx - 6, -1), -1):
            h = all_chunks[i]["meta"].get("heading", "").strip().lower()
            if h == anchor_heading or not h or _is_boilerplate_heading(h):
                walk_start = i
            else:
                break

    # Walk forward, collecting chunks until a REAL different heading appears
    section = []
    for i in range(walk_start, len(all_chunks)):
        c = all_chunks[i]
        h = c["meta"].get("heading", "").strip().lower()

        # Decide whether to keep this chunk:
        #   1. Heading matches the anchor section → keep
        #   2. No heading (plain continuation text) → keep
        #   3. Boilerplate page header → keep (don't stop, don't count as new section)
        #   4. A genuinely different, non-boilerplate heading → STOP
        if h == anchor_heading or not h or _is_boilerplate_heading(h):
            section.append({"text": c["text"], "meta": c["meta"], "score": 1.0})
        else:
            # Real new section heading detected — stop here
            break

        if len(section) >= 40:
            break

    return section if section else anchor_chunks


def _search(db, text: str, history: list, source_filter=None) -> list:
    """Standard semantic search for normal (non-list) queries."""
    hist = "\n".join(f'{"U" if m["role"]=="user" else "B"}: {m["content"][:120]}'
                     for m in history[-4:])
    raw = _call(
        "Rewrite into 3 search queries covering different angles. Fix spelling. "
        "Output ONLY a JSON array of 3 strings, nothing else.\n"
        + (f"History:\n{hist}\n" if hist else "")
        + f"Question: {text}\nJSON:", tokens=150)
    queries = [text] + _json_list(raw)[:3]
    return db.multi_query(queries, n_final=12, min_score=0.12,
                          max_per_page=4, source_filter=source_filter)


def _search_section(db, text: str, history: list, source_filter=None) -> list:
    """
    For list-mode queries:
    1. Cosine search to pin the section anchor (just needs to find the start)
    2. Walk forward through the document in reading order until heading changes
    Never misses a continuation page. Never pulls random unrelated content.
    """
    hist = "\n".join(f'{"U" if m["role"]=="user" else "B"}: {m["content"][:120]}'
                     for m in history[-4:])
    raw = _call(
        "Rewrite into 3 search queries covering different angles. Fix spelling. "
        "Output ONLY a JSON array of 3 strings, nothing else.\n"
        + (f"History:\n{hist}\n" if hist else "")
        + f"Question: {text}\nJSON:", tokens=150)
    queries = [text] + _json_list(raw)[:3]

    anchors = db.multi_query(queries, n_final=6, min_score=0.15,
                             max_per_page=3, source_filter=source_filter)
    if not anchors:
        return []

    return _get_section(db, anchors, source_filter)

# ── context builder ───────────────────────────────────────────────────────────

def _build_ctx(chunks: list, cap: int = CHUNK_CAP_NORMAL) -> str:
    by_src = defaultdict(list)
    for c in chunks[:cap]:
        by_src[c["meta"]["source"]].append(c)
    parts = []
    for src, cs in by_src.items():
        cs.sort(key=lambda x: (x["meta"].get("page", 0), x["meta"].get("chunk_id", 0)))
        parts.append(f"--- {src} ---")
        for c in cs:
            p = c["meta"].get("page")
            parts.append(f"[p.{p+1 if p is not None else '?'}] {c['text']}")
    return "\n".join(parts)

def _hist_block(history: list) -> str:
    if not history: return ""
    return "PRIOR CONVERSATION:\n" + "\n".join(
        f'{"User" if m["role"]=="user" else "Bot"}: {m["content"]}'
        for m in history[-(MAX_HIST * 2):]
    ) + "\n\n"

# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM = (
    "You are an office document assistant. You help users find and understand "
    "information from uploaded documents.\n"
    "You are direct, accurate, and professional.\n"
)

def _prompt_doc(q: str, ctx: str, hist: str, is_list: bool) -> str:
    task = (
        "Extract every relevant item from the document excerpts below. "
        "Copy the exact wording. Number each item. Cite the page at the end of each item like (p.N).\n"
        "Work through ALL excerpts provided — do not stop early.\n"
        "After listing all items, look at the last item and the last page number in the excerpts. "
        "If the section numbering is still going (e.g. ends on item 5 of what looks like a longer list), "
        "or if there is no closing note/end marker, add this exact warning on a new line:\n"
        "'Note: This section may continue beyond the excerpts shown. The next page may have more items.'\n"
        if is_list else
        "Answer the question fully using the document excerpts below. "
        "Be specific and thorough. Cite the page after each key point like (filename p.N).\n"
    )
    return (
        SYSTEM + "\n"
        + task
        + "IMPORTANT: Use ONLY the excerpts. Do not use outside knowledge. "
        "If the answer is not present, say so clearly.\n\n"
        + hist
        + f"DOCUMENT EXCERPTS:\n{ctx}\n\n"
        f"QUESTION: {q}\n\n"
        + ("EXTRACTED LIST:\n" if is_list else "ANSWER:\n")
    )

def _prompt_no_ctx(q: str, hist: str) -> str:
    return (
        SYSTEM + "\n"
        "The document search did not find relevant content for this question.\n"
        "Tell the user in one sentence, then answer from general knowledge if you can.\n\n"
        + hist + f"QUESTION: {q}\n\nANSWER:\n"
    )

def _prompt_general(q: str, hist: str) -> str:
    return (
        SYSTEM + "\n"
        "Answer this general knowledge question clearly and helpfully. "
        "Do not mention documents.\n\n"
        + hist + f"QUESTION: {q}\n\nANSWER:\n"
    )

# ── upload ────────────────────────────────────────────────────────────────────

def _upload(db, path: str):
    path = path.strip().strip('"\'')
    name, tid = os.path.basename(path), None
    with Progress(TextColumn("[blue]{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}"), TimeRemainingColumn(),
                  console=console, transient=True) as prog:
        def cb(done, total):
            nonlocal tid
            if tid is None:
                tid = prog.add_task(f"Indexing {name}…", total=total)
            prog.update(tid, completed=done)
        try:
            n = db.add_file(path, progress_cb=cb)
            console.print(f"[green]✓[/green] {n} chunks from [bold]{name}[/bold]")
        except FileNotFoundError:
            console.print(f"[red]Not found:[/red] {path}")
        except Exception as e:
            console.print(f"[red]Failed:[/red] {e}")

# ── main ──────────────────────────────────────────────────────────────────────

def run_chat():
    db, history, followups = ChatbotDB(), [], []

    console.print(Panel(
        f"[bold cyan]Document Assistant[/bold cyan]\n"
        f"[dim]Model: {MODEL} · {db.total_chunks()} chunks loaded · type help[/dim]",
        border_style="cyan"))

    while True:
        try:
            raw = console.input("\n[bold yellow]You:[/bold yellow] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]"); break
        if not raw: continue
        low = raw.lower()

        # ── commands ──────────────────────────────────────────────────────────
        if low in ("q","quit","exit"):
            console.print("[dim]Goodbye![/dim]"); break
        if low == "help":
            console.print(HELP); continue
        if low.startswith("upload "):
            _upload(db, raw[7:]); continue
        if low == "list":
            data = db.list_all()
            if not data:
                console.print("[yellow]Library is empty.[/yellow]")
            else:
                t = Table(title="Knowledge Base", border_style="cyan")
                t.add_column("Document", style="bold")
                t.add_column("Chunks", justify="right", style="cyan")
                for src, cnt in sorted(data.items()):
                    t.add_row(src, str(cnt))
                console.print(t)
            continue
        if low.startswith("delete "):
            tgt = raw[7:].strip()
            console.print(f"[green]Deleted[/green] {tgt}." if db.delete_file(tgt)
                          else f"[red]'{tgt}' not found.[/red]")
            continue

        # ── follow-up shortcut ────────────────────────────────────────────────
        if raw in ("1","2","3") and followups:
            idx = int(raw) - 1
            if idx < len(followups):
                raw, low = followups[idx], followups[idx].lower()
                console.print(f"[dim]→ {raw}[/dim]")
            followups = []

        # ── greetings ─────────────────────────────────────────────────────────
        if low in GREETINGS:
            console.print("\n[bold cyan]Bot:[/bold cyan]")
            _call(f'{SYSTEM}\nReply to this greeting warmly in one sentence: "{raw}"',
                  tokens=50, temp=0.7, stream=True)
            console.print("[dim]" + "─"*40 + "[/dim]")
            continue

        # ── intent ────────────────────────────────────────────────────────────
        has_docs      = db.total_chunks() > 0
        sources       = list(db.list_all().keys()) if has_docs else []
        page_req      = _page_num(raw) if has_docs else None
        list_mode     = _is_list(raw) if has_docs else False
        source_filter = _source_for(raw, sources) if has_docs else None

        doc_directed = False
        if has_docs:
            if page_req is not None or source_filter:
                doc_directed = True
            elif list_mode:
                doc_directed = True
            else:
                with console.status("[dim]Classifying…[/dim]", spinner="dots"):
                    doc_directed = _is_doc(raw, history, has_docs)

        # ── retrieval ─────────────────────────────────────────────────────────
        context = []
        if doc_directed:
            if page_req is not None:
                context = _get_page(db, page_req, source_filter)
                if not context:
                    console.print(f"[yellow]Nothing on page {page_req+1}.[/yellow]")
                    continue

            elif list_mode:
                # Find the section with cosine, then walk forward in reading
                # order until the heading changes — catches continuation pages
                # without pulling random unrelated content from the PDF.
                with console.status("[dim]Finding section…[/dim]", spinner="dots"):
                    context = _search_section(db, raw, history, source_filter)
                if not context:
                    console.print("[yellow]Could not locate that section.[/yellow]")
                    continue

            else:
                with console.status("[dim]Searching…[/dim]", spinner="dots"):
                    context = _search(db, raw, history, source_filter)

        # ── build prompt ──────────────────────────────────────────────────────
        hist = _hist_block(history)
        cap  = CHUNK_CAP_LIST if list_mode else CHUNK_CAP_NORMAL

        if context:
            ctx    = _build_ctx(context, cap=cap)
            prompt = _prompt_doc(raw, ctx, hist, list_mode)
        elif doc_directed:
            prompt = _prompt_no_ctx(raw, hist)
        else:
            prompt = _prompt_general(raw, hist)

        # ── answer ────────────────────────────────────────────────────────────
        console.print("\n[bold cyan]Bot:[/bold cyan]")
        answer = _call(prompt, tokens=2048, temp=0.1, stream=True)
        if not answer: continue

        # sources
        if context:
            srcs = sorted({
                f'{c["meta"]["source"]} p.{c["meta"].get("page",0)+1}'
                for c in context[:cap]
            })
            console.print(f"[dim]Sources: {' · '.join(srcs)}[/dim]")

        # follow-ups
        # For list-mode, always inject a "check next page" suggestion if the
        # answer sources end on a page that likely continues.
        next_page_hint = ""
        if list_mode and context:
            last_page = max(c["meta"].get("page", 0) for c in context[:cap]) + 1
            next_page_hint = f"What is on page {last_page + 1} of this section?"

        fu_raw = _call(
            "Generate 3 short follow-up questions the user might ask next. "
            "Output ONLY a JSON array of 3 strings.\n"
            f"User asked: {raw}\nAnswer: {answer[:300]}\nJSON:",
            tokens=200, temp=0.3)
        followups = _json_list(fu_raw)

        # Ensure "check next page" is always the first suggestion for list queries
        if list_mode and next_page_hint:
            followups = [next_page_hint] + [f for f in followups if next_page_hint not in f]
            followups = followups[:3]

        if followups:
            console.print("\n[dim]You might also ask:[/dim]")
            for i, q in enumerate(followups, 1):
                console.print(f"[dim]  {i}. {q}[/dim]")
            console.print("[dim]  Type 1, 2, or 3[/dim]")

        history.append({"role": "user",      "content": raw})
        history.append({"role": "assistant", "content": answer})
        if len(history) > MAX_HIST * 2:
            history = history[-(MAX_HIST * 2):]
        console.print("[dim]" + "─"*40 + "[/dim]")

if __name__ == "__main__":
    run_chat()