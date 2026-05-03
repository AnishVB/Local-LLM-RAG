"""
database.py — Vector database for document chunks (LanceDB + PyMuPDF edition)

Handles:
  - PDF and TXT ingestion (extraction → cleaning → chunking)
  - PDF extraction via PyMuPDF (fitz) — faster and more accurate than pdfplumber
  - Embedding via Ollama (nomic-embed-text)
  - LanceDB for persistent vector storage with hybrid search
  - Full-text (BM25) + semantic (cosine) hybrid retrieval
  - Source-filtered retrieval for doc-pinning mode

FIXES (v2):
  3. Space/time: removed redundant re-builds, reuse source index; lazy matrix build

PAGE NUMBER CONVENTION (important):
  All "page" values stored in chunk metadata are 0-indexed integers
  matching PyMuPDF's page numbering (first page = 0).
  The display layer (frontend + chat_engine) adds 1 when showing to users.
  This file never adds or subtracts 1 — it only stores what PyMuPDF gives.
"""

import os
import re
import json
import numpy as np
import ollama
import fitz  # PyMuPDF
import lancedb
import pyarrow as pa
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# ── Storage paths ─────────────────────────────────────────────────────────────
DB_PATH       = "./document_db"
LANCE_DB_PATH = os.path.join(DB_PATH, "lancedb")

# ── Tuning constants ──────────────────────────────────────────────────────────
EMBED_MODEL   = "nomic-embed-text"   # Ollama model used for embedding
EMBED_DIM     = 768                  # Dimension of nomic-embed-text vectors
EMBED_LIMIT   = 2000   # Max characters fed to the embedding model per chunk
CHUNK_CHARS   = 1200   # Target chunk size in characters
OVERLAP_CHARS = 60     # Characters of overlap between consecutive chunks
MAX_RETRIEVE  = 200    # Hard cap on candidates before scoring/filtering

# LanceDB table names
TABLE_NAME    = "documents"
QA_TABLE_NAME = "qa_cache"

# ── LanceDB Schemas ──────────────────────────────────────────────────────────
LANCE_SCHEMA = pa.schema([
    pa.field("text",     pa.utf8()),
    pa.field("source",   pa.utf8()),
    pa.field("page",     pa.int32()),
    pa.field("chunk_id", pa.int32()),
    pa.field("heading",  pa.utf8()),
    pa.field("vector",   pa.list_(pa.float32(), EMBED_DIM)),
])

QA_SCHEMA = pa.schema([
    pa.field("question",  pa.utf8()),
    pa.field("answer",    pa.utf8()),
    pa.field("source",    pa.utf8()),      # source_filter used, or "" for none
    pa.field("timestamp", pa.utf8()),
    pa.field("vector",    pa.list_(pa.float32(), EMBED_DIM)),
])

# QA cache similarity threshold: how close a question must be to reuse a cached answer
QA_CACHE_THRESHOLD = 0.93


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that match document-specific boilerplate lines (headers/footers
# that repeat on every page and add no informational value)
_BOILERPLATE_PATTERNS = [
    re.compile(r'bharat\s+dynamics\s+limited', re.IGNORECASE),
    re.compile(r'conduct.*discipline.*appeal',  re.IGNORECASE),
    re.compile(r'corporate\s+office',           re.IGNORECASE),
    re.compile(r'issue\s+date',                 re.IGNORECASE),
    re.compile(r'^page\s+\d+\s+of\s+\d+$',     re.IGNORECASE),  # "Page 3 of 20"
    re.compile(r'^\d+\s+of\s+\d+$'),                             # "3 of 20"
]

def _is_boilerplate(line: str) -> bool:
    """Return True if the line matches any boilerplate pattern."""
    s = line.strip()
    return any(p.search(s) for p in _BOILERPLATE_PATTERNS)

def _clean(text: str) -> str:
    """
    Normalise extracted text:
      - Strip non-printable / non-ASCII characters
      - Collapse multiple spaces/tabs into one
      - Limit consecutive blank lines to two
    """
    if not text:
        return ""
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E]', ' ', text)  # non-printable → space
    text = re.sub(r'[ \t]{2,}', ' ', text)                   # multi-space → single
    text = re.sub(r'\n{3,}', '\n\n', text)                   # triple newline → double
    return text.strip()

def _detect_heading(lines: list) -> str:
    """
    Heuristic: look at the first 8 lines of a page and return the most
    likely section heading.  Criteria:
      - Between 5 and 100 characters
      - Not boilerplate
      - Either ALL CAPS, or mostly Title Case (≥60% capitalised words)
    Returns empty string if no candidate found.
    """
    for line in lines[:8]:
        line = line.strip()
        if not (4 < len(line) < 100):
            continue
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


# ─────────────────────────────────────────────────────────────────────────────
# PDF extraction — PyMuPDF (replaces pdfplumber for speed + accuracy)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pdf(file_path: str) -> list:
    """
    Extract text from every page of a PDF using PyMuPDF (fitz).

    Advantages over pdfplumber:
      - 5-10x faster extraction
      - Better handling of multi-column layouts
      - More accurate reading order
      - Better Unicode support

    Returns a list of dicts:
      {"page": int (0-indexed), "text": str, "heading": str}

    Pages that produce no text (e.g. scanned images) are silently skipped.
    """
    pages = []
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise ValueError(f"Cannot open PDF: {e}")

    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            # Use "text" mode with sorting for proper reading order
            raw = page.get_text("text", sort=True) or ""
        except Exception:
            continue   # Skip pages that throw (corrupt / image-only)
        raw = _clean(raw)
        if not raw.strip():
            continue
        lines   = [l for l in raw.splitlines() if l.strip()]
        heading = _detect_heading(lines)
        # page_num is 0-indexed here; display layer will add 1 for users
        pages.append({"page": page_num, "text": raw, "heading": heading})

    doc.close()
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_page(page_text: str, source: str, page_num: int, heading: str) -> list:
    """
    Split a single page's text into overlapping chunks of ≤CHUNK_CHARS.

    Algorithm:
      1. Split by double newlines (paragraph boundaries).
      2. Fall back to line-by-line if only one paragraph.
      3. Accumulate paragraphs into a buffer until the next one would overflow.
      4. On overflow, flush the buffer as a chunk, carry the last OVERLAP_CHARS
         as the start of the next chunk.
      5. Extra-long paragraphs (> CHUNK_CHARS) are split word-by-word.

    Each chunk dict:
      {"text": str, "meta": {"source": str, "page": int, "chunk_id": int, "heading": str}}
    """
    if not page_text or not page_text.strip():
        return []

    # Split into paragraphs; fall back to lines if the page is one big block
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', page_text) if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [l.strip() for l in page_text.splitlines() if l.strip()]
    if not paragraphs:
        return []

    chunks   = []
    buf      = ""
    chunk_id = 0

    def _flush(text: str, cid: int):
        """Package the accumulated buffer as a chunk dict, or return None if empty."""
        text = text.strip()
        if not text:
            return None
        # Prepend the heading to the first chunk of each page for retrieval context
        label = f"[{heading}] " if (cid == 0 and heading) else ""
        full  = (label + text).strip()[:CHUNK_CHARS]
        return {
            "text": full,
            "meta": {
                "source":   source,
                "page":     page_num,   # 0-indexed; display layer adds 1
                "chunk_id": cid,
                "heading":  heading,
            }
        }

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Handle paragraphs that exceed chunk size on their own
        if len(para) > CHUNK_CHARS:
            if buf.strip():
                c = _flush(buf, chunk_id)
                if c:
                    chunks.append(c)
                chunk_id += 1
                buf = ""
            # Split the long paragraph word-by-word
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

        # Check if adding this paragraph would overflow the buffer
        trial = (buf + "\n" + para).strip() if buf else para
        if len(trial) > CHUNK_CHARS and buf:
            c = _flush(buf, chunk_id)
            if c:
                chunks.append(c)
            chunk_id += 1
            # Keep a short overlap for context continuity
            tail = buf[-OVERLAP_CHARS:].strip()
            buf  = (tail + "\n" + para).strip() if tail else para
        else:
            buf = trial

    # Flush any remaining text
    if buf.strip():
        c = _flush(buf, chunk_id)
        if c:
            chunks.append(c)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=256)
def _embed_cached(text: str) -> tuple:
    """
    Embed text using Ollama's nomic-embed-text model.
    Returns a tuple (hashable, so it can be cached by lru_cache).
    Falls back to a shorter prompt if the full text fails.
    """
    safe = text.strip()[:EMBED_LIMIT] or " "
    try:
        return tuple(ollama.embeddings(model=EMBED_MODEL, prompt=safe)["embedding"])
    except Exception:
        # Retry with truncated text in case the full length caused an error
        return tuple(ollama.embeddings(model=EMBED_MODEL, prompt=safe[:200])["embedding"])

def _embed(text: str) -> list:
    """Convenience wrapper: embed text and return as a plain list."""
    return list(_embed_cached(text))


# ─────────────────────────────────────────────────────────────────────────────
# ChatbotDB — main vector store class (LanceDB backend)
# ─────────────────────────────────────────────────────────────────────────────

class ChatbotDB:
    """
    Vector store backed by LanceDB with hybrid search capabilities.

    Key design points:
      - Documents are stored as rows in a LanceDB table with text, metadata,
        and embedding vectors.
      - Hybrid search combines semantic (cosine) and full-text (BM25) retrieval
        using LanceDB's built-in FTS support via Tantivy.
      - Source filtering uses LanceDB's native WHERE clause for efficiency.
      - The .chunks property provides backward-compatible access for
        chat_engine.py's section-walk and page-retrieval logic.
    """

    def __init__(self):
        os.makedirs(DB_PATH, exist_ok=True)
        self._db = lancedb.connect(LANCE_DB_PATH)
        self._table = None
        self._qa_table = None
        self._chunks_cache = None   # Lazy cache for .chunks property
        self._open_or_create_table()
        self._open_or_create_qa_table()

    def _open_or_create_table(self):
        """Open the existing LanceDB table or create an empty one."""
        try:
            if TABLE_NAME in self._db.table_names():
                self._table = self._db.open_table(TABLE_NAME)
            else:
                self._table = None
        except Exception:
            self._table = None

    def _create_table(self, data: list):
        """Create the LanceDB table from a list of row dicts."""
        self._table = self._db.create_table(
            TABLE_NAME, data=data, schema=LANCE_SCHEMA, mode="overwrite"
        )
        # Create full-text search index for hybrid search
        try:
            self._table.create_fts_index("text", replace=True)
        except Exception as e:
            print(f"[INFO] FTS index creation: {e}")

    def _rebuild_fts(self):
        """Rebuild the full-text search index after data changes."""
        if self._table is not None:
            try:
                self._table.create_fts_index("text", replace=True)
            except Exception:
                pass  # FTS is optional; vector search always works

    def _invalidate_cache(self):
        """Clear the chunks cache after any write operation."""
        self._chunks_cache = None

    @property
    def chunks(self) -> list:
        """
        Backward-compatible property: returns all chunks as a list of dicts
        matching the old format: [{"text": str, "meta": {...}}, ...]

        Used by chat_engine.py for _get_page() and _get_section() operations.
        Results are cached and invalidated on writes.
        """
        if self._chunks_cache is not None:
            return self._chunks_cache

        if self._table is None:
            return []
        try:
            df = self._table.to_pandas()
            result = []
            for _, row in df.iterrows():
                result.append({
                    "text": row["text"],
                    "meta": {
                        "source":   row["source"],
                        "page":     int(row["page"]),
                        "chunk_id": int(row["chunk_id"]),
                        "heading":  row["heading"],
                    }
                })
            self._chunks_cache = result
            return result
        except Exception:
            return []

    def add_file(self, file_path: str, progress_cb=None) -> int:
        """
        Ingest a new document (PDF or TXT):
          1. Extract text page by page (PyMuPDF for PDFs).
          2. Chunk each page.
          3. Embed each chunk (with progress reporting).
          4. Remove any old version of the same file.
          5. Add new rows to LanceDB and rebuild the FTS index.

        Returns the number of chunks added.
        progress_cb(done, total) is called every 50 chunks.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext  = Path(file_path).suffix.lower()
        name = Path(file_path).name

        # Extract text based on file type
        if ext == ".pdf":
            pages = _extract_pdf(file_path)
            if not pages:
                raise ValueError("PDF has no extractable text.")
            raw_chunks = []
            for p in pages:
                raw_chunks.extend(
                    _chunk_page(p["text"], name, p["page"], p["heading"])
                )
        elif ext in (".txt", ".md"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_chunks = _chunk_page(_clean(f.read()), name, 0, "")
        else:
            raise ValueError("Unsupported format — use .pdf, .txt, or .md")

        # Filter out any empty chunks that slipped through
        raw_chunks = [c for c in raw_chunks if c and c.get("text", "").strip()]
        if not raw_chunks:
            raise ValueError("File is empty or has no extractable text.")

        # Remove the old version of this file
        self.delete_file(name)

        # Embed each chunk and build rows for LanceDB
        total = len(raw_chunks)
        rows = []
        for i, chunk in enumerate(raw_chunks):
            try:
                emb = _embed(chunk["text"])
            except Exception as e:
                print(f"[WARNING] skipping chunk {i}: {e}")
                continue
            rows.append({
                "text":     chunk["text"],
                "source":   chunk["meta"]["source"],
                "page":     chunk["meta"]["page"],
                "chunk_id": chunk["meta"]["chunk_id"],
                "heading":  chunk["meta"]["heading"],
                "vector":   emb,
            })
            if progress_cb and (i + 1) % 50 == 0:
                progress_cb(i + 1, total)

        if progress_cb:
            progress_cb(total, total)

        if not rows:
            raise ValueError("No chunks could be embedded.")

        # Clear QA cache since document content changed
        self.clear_qa_cache()

        # Add rows to LanceDB
        if self._table is None:
            self._create_table(rows)
        else:
            self._table.add(rows)
            self._rebuild_fts()

        self._invalidate_cache()
        return len(rows)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _hybrid_search(self, query_text: str, query_vector: list,
                       n: int = 20, source_filter: str = None) -> list:
        """
        Hybrid search combining vector similarity and full-text BM25.

        LanceDB's hybrid_search does Reciprocal Rank Fusion (RRF) by default,
        combining the strengths of both search types.
        """
        if self._table is None:
            return []

        where_clause = f"source = '{source_filter}'" if source_filter else None

        try:
            # Try hybrid search first (vector + FTS)
            # Syntax: search(query_type="hybrid").vector(vec).text(text)
            builder = self._table.search(query_type="hybrid", vector_column_name="vector")\
                                 .vector(query_vector)\
                                 .text(query_text)

            if where_clause:
                builder = builder.where(where_clause)

            results = builder.limit(n).to_pandas()
        except Exception:
            # Fallback to pure vector search if FTS isn't available
            return self._vector_search(query_vector, n=n, source_filter=source_filter)

        if results.empty:
            return []

        output = []
        for _, row in results.iterrows():
            score = 1.0
            if "_distance" in row:
                # If metric was cosine, sim = 1 - distance
                score = max(0.0, 1.0 - float(row["_distance"]))
            elif "_relevance_score" in row:
                # RRF scores are typically very small (e.g., 0.01 to 0.033)
                # Multiply by 30 to bring them into the 0.0 - 1.0 range
                # so they pass the min_score threshold in multi_query
                score = min(1.0, float(row["_relevance_score"]) * 30.0)

            output.append({
                "text": row["text"],
                "meta": {
                    "source":   row["source"],
                    "page":     int(row["page"]),
                    "chunk_id": int(row["chunk_id"]),
                    "heading":  row["heading"],
                },
                "score": round(score, 4),
            })
        return output

    def _vector_search(self, query_vector: list, n: int = 20,
                       source_filter: str = None) -> list:
        """Pure vector (cosine) search — used as a building block."""
        if self._table is None:
            return []

        where_clause = f"source = '{source_filter}'" if source_filter else None

        try:
            builder = self._table.search(
                query_vector, query_type="vector", vector_column_name="vector"
            ).metric("cosine")
            if where_clause:
                builder = builder.where(where_clause)
            results = builder.limit(n).to_pandas()
        except Exception:
            return []

        if results.empty:
            return []

        output = []
        for _, row in results.iterrows():
            score = 1.0
            if "_distance" in row:
                score = max(0.0, 1.0 - float(row["_distance"]))
            output.append({
                "text": row["text"],
                "meta": {
                    "source":   row["source"],
                    "page":     int(row["page"]),
                    "chunk_id": int(row["chunk_id"]),
                    "heading":  row["heading"],
                },
                "score": round(score, 4),
            })
        return output

    # ── Public query methods ──────────────────────────────────────────────────

    def multi_query(self, queries: list, n_final: int = 20,
                    min_score: float = 0.10, max_per_page: int = 5,
                    source_filter: str = None) -> list:
        """
        Standard retrieval: run hybrid search with multiple query variants,
        then merge and deduplicate results.

        Returns a list of {"text", "meta", "score"} dicts, sorted by score.
        """
        if self._table is None:
            return []

        # Embed all query variants in parallel
        with ThreadPoolExecutor(max_workers=min(4, len(queries))) as ex:
            vecs = list(ex.map(_embed, queries))

        # Run hybrid search for each query and merge results
        all_results = {}
        for query_text, query_vec in zip(queries, vecs):
            hits = self._hybrid_search(
                query_text, query_vec,
                n=n_final * 2,
                source_filter=source_filter,
            )
            for hit in hits:
                key = (hit["meta"]["source"], hit["meta"]["page"],
                       hit["meta"]["chunk_id"])
                if key not in all_results or hit["score"] > all_results[key]["score"]:
                    all_results[key] = hit

        # Sort by score descending, apply min_score and max_per_page
        ranked = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)

        seen, results = {}, []
        for hit in ranked:
            if hit["score"] < min_score:
                break
            key = (hit["meta"]["source"], hit["meta"].get("page"))
            if seen.get(key, 0) >= max_per_page:
                continue
            results.append(hit)
            seen[key] = seen.get(key, 0) + 1
            if len(results) >= n_final:
                break
        return results

    def exhaustive_query(self, queries: list, n_final: int = 200,
                         source_filter: str = None) -> list:
        """
        Exhaustive retrieval for when we need the entire document or a very
        large portion of it (e.g. full-document summary).
        For source-filtered queries: returns all chunks in reading order.
        For unfiltered: returns top-scored chunks down to score 0.06.
        """
        if self._table is None:
            return []

        if source_filter:
            # Just return all chunks from the source in order
            try:
                df = self._table.search().where(
                    f"source = '{source_filter}'"
                ).limit(n_final).to_pandas()

                result = []
                for _, row in df.iterrows():
                    result.append({
                        "text": row["text"],
                        "meta": {
                            "source":   row["source"],
                            "page":     int(row["page"]),
                            "chunk_id": int(row["chunk_id"]),
                            "heading":  row["heading"],
                        },
                        "score": 1.0,
                    })
                result.sort(key=lambda x: (
                    x["meta"].get("page", 0),
                    x["meta"].get("chunk_id", 0),
                ))
                return result
            except Exception:
                # Fallback: filter from all chunks
                all_c = self.chunks
                result = [
                    {"text": c["text"], "meta": c["meta"], "score": 1.0}
                    for c in all_c if c["meta"]["source"] == source_filter
                ]
                result.sort(key=lambda x: (
                    x["meta"].get("page", 0),
                    x["meta"].get("chunk_id", 0),
                ))
                return result

        # Scored retrieval for multi-source queries
        with ThreadPoolExecutor(max_workers=min(4, len(queries))) as ex:
            vecs = list(ex.map(_embed, queries))

        all_results = {}
        for query_text, query_vec in zip(queries, vecs):
            hits = self._hybrid_search(
                query_text, query_vec,
                n=n_final,
                source_filter=source_filter,
            )
            for hit in hits:
                key = (hit["meta"]["source"], hit["meta"]["page"],
                       hit["meta"]["chunk_id"])
                if key not in all_results or hit["score"] > all_results[key]["score"]:
                    all_results[key] = hit

        ranked = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)

        seen, results = {}, []
        for hit in ranked:
            if hit["score"] < 0.06:
                break
            key = (hit["meta"]["source"], hit["meta"].get("page"))
            if seen.get(key, 0) >= 8:
                continue
            results.append(hit)
            seen[key] = seen.get(key, 0) + 1
            if len(results) >= n_final:
                break

        results.sort(key=lambda x: (
            x["meta"].get("page", 0),
            x["meta"].get("chunk_id", 0),
        ))
        return results

    def query(self, text: str, n: int = 5) -> list:
        """Simple single-query convenience method."""
        return self.multi_query([text], n_final=n)

    def list_all(self) -> dict:
        """Return {source_filename: chunk_count} for all indexed documents."""
        if self._table is None:
            return {}
        try:
            df = self._table.to_pandas()
            if df.empty:
                return {}
            return dict(df.groupby("source").size())
        except Exception:
            return {}

    def delete_file(self, name: str) -> bool:
        """
        Remove all chunks for the named file.
        Uses LanceDB's native delete for efficiency.
        Returns True if the file was found, False otherwise.
        """
        if self._table is None:
            return False
        try:
            # Check if file exists first
            existing = self.list_all()
            if name not in existing:
                return False

            self._table.delete(f"source = '{name}'")
            self._rebuild_fts()
            self._invalidate_cache()
            self.clear_qa_cache()   # Answers may reference deleted content

            # If table is now empty, clean up
            remaining = self.list_all()
            if not remaining:
                try:
                    self._db.drop_table(TABLE_NAME)
                except Exception:
                    pass
                self._table = None

            return True
        except Exception as e:
            print(f"[WARNING] delete_file error: {e}")
            return False

    def total_chunks(self) -> int:
        """Return the total number of indexed text chunks."""
        if self._table is None:
            return 0
        try:
            return self._table.count_rows()
        except Exception:
            return 0

    # ── QA Answer Cache ───────────────────────────────────────────────────────

    def _open_or_create_qa_table(self):
        """Open the QA cache table if it exists."""
        try:
            if QA_TABLE_NAME in self._db.table_names():
                self._qa_table = self._db.open_table(QA_TABLE_NAME)
            else:
                self._qa_table = None
        except Exception:
            self._qa_table = None

    def cache_qa(self, question: str, answer: str, source_filter: str = None):
        """
        Store a question-answer pair in the cache with the question's embedding.
        Future similar questions will get this answer instantly.
        """
        if not question.strip() or not answer.strip():
            return
        try:
            from datetime import datetime
            emb = _embed(question)
            row = {
                "question":  question.strip(),
                "answer":    answer.strip(),
                "source":    source_filter or "",
                "timestamp": datetime.now().isoformat(),
                "vector":    emb,
            }
            if self._qa_table is None:
                self._qa_table = self._db.create_table(
                    QA_TABLE_NAME, data=[row], schema=QA_SCHEMA, mode="overwrite"
                )
            else:
                self._qa_table.add([row])
        except Exception as e:
            print(f"[INFO] QA cache store skipped: {e}")

    def find_cached_qa(self, question: str,
                       source_filter: str = None) -> dict | None:
        """
        Search the cache for a previously answered similar question.

        Returns {"question": str, "answer": str} if a match is found
        with similarity >= QA_CACHE_THRESHOLD, else None.

        Also matches on source_filter to avoid returning answers from
        a different document context.
        """
        if self._qa_table is None:
            return None
        try:
            q_emb = _embed(question)
            src = source_filter or ""

            builder = self._qa_table.search(
                q_emb, query_type="vector", vector_column_name="vector"
            )
            # Filter by source context
            builder = builder.where(f"source = '{src}'")
            results = builder.limit(1).to_pandas()

            if results.empty:
                return None

            row = results.iloc[0]
            # Check similarity threshold
            if "_distance" in row:
                similarity = max(0.0, 1.0 - float(row["_distance"]))
                if similarity >= QA_CACHE_THRESHOLD:
                    return {
                        "question": row["question"],
                        "answer":   row["answer"],
                    }
            return None
        except Exception:
            return None

    def clear_qa_cache(self):
        """Clear the entire QA cache. Called when documents change."""
        try:
            if self._qa_table is not None:
                self._db.drop_table(QA_TABLE_NAME)
                self._qa_table = None
        except Exception:
            self._qa_table = None

    def find_qa_examples(self, question: str, source_filter: str = None,
                         n: int = 2, min_similarity: float = 0.70) -> list:
        """
        Find similar past Q&A pairs to use as few-shot examples in the prompt.

        Unlike find_cached_qa (which returns exact matches for reuse),
        this returns SIMILAR past answers to guide the LLM's style and depth.
        Uses a lower threshold (0.70 vs 0.93) since these are reference
        examples, not direct cache hits.

        Returns a list of {"question": str, "answer": str} dicts.
        """
        if self._qa_table is None:
            return []
        try:
            q_emb = _embed(question)
            src = source_filter or ""

            builder = self._qa_table.search(
                q_emb, query_type="vector", vector_column_name="vector"
            )
            builder = builder.where(f"source = '{src}'")
            results = builder.limit(n + 1).to_pandas()  # +1 to skip exact match

            if results.empty:
                return []

            examples = []
            for _, row in results.iterrows():
                if "_distance" in row:
                    similarity = max(0.0, 1.0 - float(row["_distance"]))
                else:
                    similarity = 0.0

                # Skip exact cache hits (those go through find_cached_qa)
                if similarity >= QA_CACHE_THRESHOLD:
                    continue
                # Only include if similar enough to be useful
                if similarity >= min_similarity:
                    examples.append({
                        "question": row["question"],
                        "answer":   row["answer"],
                    })
                if len(examples) >= n:
                    break

            return examples
        except Exception:
            return []