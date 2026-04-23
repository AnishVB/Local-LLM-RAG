import os
import re
import json
import math
import ollama
import pdfplumber
from pathlib import Path

DB_PATH     = "./librarian_db"
CHUNKS_FILE = os.path.join(DB_PATH, "chunks.json")
EMBED_FILE  = os.path.join(DB_PATH, "embeddings.json")

EMBED_MODEL   = "nomic-embed-text"
EMBED_LIMIT   = 2000         # nomic-embed-text supports 8192 tokens (~4000 chars)
CHUNK_CHARS   = 1200         # nomic handles much larger chunks
OVERLAP_CHARS = 60
MAX_RETRIEVE  = 200


# ─── persistence ──────────────────────────────────────────────────────────────

def _load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def _save_json(path, data):
    os.makedirs(DB_PATH, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# ─── math ─────────────────────────────────────────────────────────────────────

def _cosine(a, b):
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return 0.0 if (mag_a == 0 or mag_b == 0) else dot / (mag_a * mag_b)


# ─── text cleaning ────────────────────────────────────────────────────────────

# Patterns that appear on every page header/footer and must never be mistaken
# for a section heading.  Add any document-specific boilerplate here.
_BOILERPLATE_PATTERNS = [
    re.compile(r'bharat\s+dynamics\s+limited', re.IGNORECASE),
    re.compile(r'conduct.*discipline.*appeal', re.IGNORECASE),
    re.compile(r'corporate\s+office', re.IGNORECASE),
    re.compile(r'issue\s+date', re.IGNORECASE),
    re.compile(r'^page\s+\d+\s+of\s+\d+$', re.IGNORECASE),
    re.compile(r'^\d+\s+of\s+\d+$'),
]

def _is_boilerplate(line: str) -> bool:
    """Return True if the line looks like a repeating page header/footer."""
    s = line.strip()
    return any(p.search(s) for p in _BOILERPLATE_PATTERNS)

def _clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E]', ' ', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _detect_heading(lines: list) -> str:
    for line in lines[:8]:
        line = line.strip()
        if not (4 < len(line) < 100):
            continue
        # Skip repeating page-header boilerplate — these are NOT section headings
        if _is_boilerplate(line):
            continue
        if line.isupper():
            return line
        words = line.split()
        if not words:
            continue
        cap = sum(1 for w in words if w and w[0].isupper())
        if len(words) <= 10 and cap / len(words) >= 0.6:
            return line
    return ""


# ─── PDF extraction ───────────────────────────────────────────────────────────

def _extract_pdf(file_path: str) -> list:
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            try:
                raw = page.extract_text(layout=False) or ""
            except Exception:
                continue
            raw = _clean(raw)
            if not raw.strip():
                continue
            lines   = [l for l in raw.splitlines() if l.strip()]
            heading = _detect_heading(lines)
            pages.append({"page": page_num, "text": raw, "heading": heading})
    return pages


# ─── chunking ─────────────────────────────────────────────────────────────────

def _chunk_page(page_text: str, source: str, page_num: int, heading: str) -> list:
    if not page_text or not page_text.strip():
        return []

    paragraphs = [p.strip() for p in re.split(r'\n{2,}', page_text) if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [l.strip() for l in page_text.splitlines() if l.strip()]
    if not paragraphs:
        return []

    chunks   = []
    buf      = ""
    chunk_id = 0

    def _flush(text: str, cid: int):
        text = text.strip()
        if not text:
            return None
        label = f"[{heading}] " if (cid == 0 and heading) else ""
        full  = (label + text).strip()[:CHUNK_CHARS]   # hard cap
        return {
            "text": full,
            "meta": {"source": source, "page": page_num,
                     "chunk_id": cid, "heading": heading}
        }

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > CHUNK_CHARS:
            if buf.strip():
                c = _flush(buf, chunk_id)
                if c:
                    chunks.append(c)
                chunk_id += 1
                buf = ""
            words, sub = para.split(), ""
            for word in words:
                trial = (sub + " " + word).strip() if sub else word
                if len(trial) > CHUNK_CHARS and sub:
                    c = _flush(sub, chunk_id)
                    if c:
                        chunks.append(c)
                    chunk_id += 1
                    sub = word
                else:
                    sub = trial
            buf = sub
            continue

        trial = (buf + "\n" + para).strip() if buf else para
        if len(trial) > CHUNK_CHARS and buf:
            c = _flush(buf, chunk_id)
            if c:
                chunks.append(c)
            chunk_id += 1
            tail = buf[-OVERLAP_CHARS:].strip()
            buf  = (tail + "\n" + para).strip() if tail else para
        else:
            buf = trial

    if buf.strip():
        c = _flush(buf, chunk_id)
        if c:
            chunks.append(c)

    return chunks


# ─── embedding ────────────────────────────────────────────────────────────────

def _embed(text: str) -> list:
    """Truncate hard at EMBED_LIMIT chars before sending to the model."""
    safe = text.strip()[:EMBED_LIMIT]
    if not safe:
        safe = " "
    try:
        return ollama.embeddings(model=EMBED_MODEL, prompt=safe)["embedding"]
    except Exception as e:
        short = safe[:200]
        return ollama.embeddings(model=EMBED_MODEL, prompt=short)["embedding"]


# ─── DB class ─────────────────────────────────────────────────────────────────

class ChatbotDB:
    def __init__(self):
        self.chunks:     list = _load_json(CHUNKS_FILE)
        self.embeddings: list = _load_json(EMBED_FILE)
        # Re-sync if a previous upload crashed mid-way and left them mismatched
        min_len = min(len(self.chunks), len(self.embeddings))
        if len(self.chunks) != len(self.embeddings):
            print(f"[WARNING] chunks/embeddings mismatch ({len(self.chunks)} vs "
                  f"{len(self.embeddings)}). Auto-trimming to {min_len}.")
        self.chunks     = self.chunks[:min_len]
        self.embeddings = self.embeddings[:min_len]

    def _save(self):
        _save_json(CHUNKS_FILE, self.chunks)
        _save_json(EMBED_FILE,  self.embeddings)

    def add_file(self, file_path: str, progress_cb=None) -> int:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext  = Path(file_path).suffix.lower()
        name = Path(file_path).name
        raw_chunks = []

        if ext == ".pdf":
            pages = _extract_pdf(file_path)
            if not pages:
                raise ValueError(
                    "PDF has no extractable text. "
                    "It may be a scanned/image-only PDF which needs OCR first."
                )
            for p in pages:
                raw_chunks.extend(
                    _chunk_page(p["text"], name, p["page"], p["heading"])
                )

        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = _clean(f.read())
            raw_chunks = _chunk_page(text, name, page_num=0, heading="")

        else:
            raise ValueError("Unsupported format — use .pdf or .txt")

        raw_chunks = [c for c in raw_chunks if c and c.get("text", "").strip()]
        if not raw_chunks:
            raise ValueError("File is empty or has no extractable text.")

        # Remove old chunks for this file
        keep = [i for i, c in enumerate(self.chunks) if c["meta"]["source"] != name]
        self.chunks     = [self.chunks[i]     for i in keep]
        self.embeddings = [self.embeddings[i] for i in keep]

        total = len(raw_chunks)
        for i, chunk in enumerate(raw_chunks):
            try:
                emb = _embed(chunk["text"])
            except Exception as e:
                print(f"[WARNING] Skipping chunk {i} due to embed error: {e}")
                continue
            self.chunks.append(chunk)
            self.embeddings.append(emb)
            # Save incrementally every 50 chunks so a crash doesn't lose everything
            if (i + 1) % 50 == 0:
                self._save()
            if progress_cb:
                progress_cb(i + 1, total)

        self._save()
        return len([c for c in self.chunks if c["meta"]["source"] == name])

    # ── scoring ───────────────────────────────────────────────────────────────

    def _score_all(self, queries: list, source_filter: str = None) -> dict:
        """
        Score every chunk against all queries and keep the best score per chunk.
        If source_filter is set, only chunks from that source are scored.
        """
        best = {}
        n = min(len(self.chunks), len(self.embeddings))
        for qt in queries:
            q = _embed(qt)
            for i in range(n):
                if source_filter and self.chunks[i]["meta"]["source"] != source_filter:
                    continue
                s = _cosine(q, self.embeddings[i])
                if s > best.get(i, 0.0):
                    best[i] = s
        return best

    # ── standard search ───────────────────────────────────────────────────────

    def multi_query(self, queries: list, n_final: int = 20,
                    min_score: float = 0.10, max_per_page: int = 5,
                    source_filter: str = None) -> list:
        """
        Semantic search with per-page deduplication.
        Raised default max_per_page to 5 (was 4) so dense pages aren't truncated.
        """
        if not self.chunks:
            return []
        ranked = sorted(self._score_all(queries, source_filter).items(),
                        key=lambda x: x[1], reverse=True)
        seen, results = {}, []
        for i, score in ranked[:MAX_RETRIEVE * 2]:
            if score < min_score:
                break
            chunk = self.chunks[i]
            key   = (chunk["meta"]["source"], chunk["meta"].get("page"))
            if seen.get(key, 0) >= max_per_page:
                continue
            results.append({
                "text":  chunk["text"],
                "meta":  chunk["meta"],
                "score": round(score, 4)
            })
            seen[key] = seen.get(key, 0) + 1
            if len(results) >= n_final:
                break
        return results

    # ── exhaustive search (list-all / full-section) ───────────────────────────

    def exhaustive_query(self, queries: list, n_final: int = 200,
                         source_filter: str = None) -> list:
        """
        For "list all X" style queries we cannot afford to miss chunks because
        their cosine score happened to be low.

        Strategy
        --------
        • If source_filter is set → return ALL chunks from that source in
          reading order.  No cosine threshold, no page cap.  The LLM receives
          the full document and can find every item in the list regardless of
          which page it appears on.

        • If no source_filter → fall back to a scored search with a very low
          threshold (0.06) and a generous per-page cap (8) so we still capture
          continuation pages that are only loosely similar to the query.

        In both cases results are sorted into document reading order so the LLM
        sees pages 1 → 2 → 3 … rather than a cosine-ranked jumble.
        """
        if not self.chunks:
            return []

        if source_filter:
            # Return every chunk from this source, no filtering whatsoever
            source_chunks = [
                {"text": c["text"], "meta": c["meta"], "score": 1.0}
                for c in self.chunks
                if c["meta"]["source"] == source_filter
            ]
            source_chunks.sort(
                key=lambda x: (x["meta"].get("page", 0), x["meta"].get("chunk_id", 0))
            )
            return source_chunks  # ALL chunks — nothing is missed

        # No source filter: scored search with very permissive settings
        ranked = sorted(self._score_all(queries, source_filter).items(),
                        key=lambda x: x[1], reverse=True)
        seen, results = {}, []
        for i, score in ranked[:MAX_RETRIEVE * 2]:
            if score < 0.06:
                break
            chunk = self.chunks[i]
            key   = (chunk["meta"]["source"], chunk["meta"].get("page"))
            if seen.get(key, 0) >= 8:   # allow up to 8 chunks per page for lists
                continue
            results.append({
                "text":  chunk["text"],
                "meta":  chunk["meta"],
                "score": round(score, 4)
            })
            seen[key] = seen.get(key, 0) + 1
            if len(results) >= n_final:
                break

        # Sort into reading order so continuation pages aren't scattered
        results.sort(
            key=lambda x: (x["meta"].get("page", 0), x["meta"].get("chunk_id", 0))
        )
        return results

    def query(self, text: str, n: int = 5) -> list:
        return self.multi_query([text], n_final=n)

    def list_all(self) -> dict:
        counts = {}
        for c in self.chunks:
            src = c["meta"]["source"]
            counts[src] = counts.get(src, 0) + 1
        return counts

    def delete_file(self, name: str) -> bool:
        keep = [i for i, c in enumerate(self.chunks) if c["meta"]["source"] != name]
        if len(keep) == len(self.chunks):
            return False
        self.chunks     = [self.chunks[i] for i in keep]
        self.embeddings = [self.embeddings[i] for i in keep]
        self._save()
        return True

    def total_chunks(self) -> int:
        return len(self.chunks)