

"""
research_paper.py

Fetch arXiv papers and generate:
- 2-line LLM overview
- 3–5 bullet key points per paper

Supports:
- Topic search
- Keyword search
- Author filtering (post-process)
"""

# ================== IMPORTS ==================

import os
import json
import re
import sys
import logging
from typing import List, Dict, Any
from datetime import datetime
from html import unescape
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.retrievers import ArxivRetriever

# ================== CONSTANTS ==================

DEFAULT_LOAD_MAX_DOCS = 8
DEFAULT_PER_TOPIC = 5
DEFAULT_TOP_K = 10

DEFAULT_TOPICS = [
    "machine learning",
    "deep learning",
    "generative ai",
    "transformers",
    "computer vision",
    "natural language processing",
    "reinforcement learning",
    "retrieval-augmented generation",
    "agentic AI",
    "model context protocol (MCP)",
]

# ================== LOGGING ==================

logger = logging.getLogger("research_paper")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ================== REGEX ==================

ARXIV_ABS_RE = re.compile(r"https?://arxiv\.org/abs/([\w\.\-]+)", re.I)
ARXIV_PDF_RE = re.compile(r"https?://arxiv\.org/pdf/([\w\.\-]+)\.pdf", re.I)

# ================== LLM SETUP ==================

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0.2,
)

# ================== HELPERS ==================

def parse_date(meta: Dict[str, Any]) -> datetime | None:
    for k in ("published", "updated", "created"):
        if k in meta:
            try:
                return datetime.fromisoformat(str(meta[k]))
            except Exception:
                pass
    return None


def prepare_abstract_for_llm(text: str, max_words: int = 120) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return " ".join(text.split()[:max_words])


# ================== LLM SUMMARY ==================

def generate_llm_summary_for_paper(title: str, abstract: str) -> Dict[str, Any]:
    if not abstract or len(abstract.split()) < 15:
        return {"overview": "", "key_points": []}

    prompt = f"""
You are an expert machine learning researcher.

TASK:
1. Write EXACTLY TWO LINES explaining what this paper is about.
2. Write 3–5 bullet points describing the main contributions.

RULES:
- Do NOT copy the abstract
- Do NOT add explanations
- Output ONLY valid JSON

FORMAT:
{{
  "overview": "line 1\\nline 2",
  "key_points": ["point 1", "point 2", "point 3"]
}}

ABSTRACT:
\"\"\"{abstract}\"\"\"
"""

    try:
        resp = llm.invoke(prompt)
        if not resp or not resp.content:
            return {"overview": "", "key_points": []}

        text = re.sub(r"```json|```", "", resp.content).strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {"overview": "", "key_points": []}

        data = json.loads(m.group(0))
        overview_lines = [l.strip() for l in data.get("overview", "").splitlines() if l.strip()]
        overview = "\n".join(overview_lines[:2])

        points = [p.strip() for p in data.get("key_points", []) if p.strip()][:5]

        return {"overview": overview, "key_points": points}

    except Exception as e:
        logger.warning("LLM summary failed: %s", e)
        return {"overview": "", "key_points": []}


# ================== PAPER PARSING ==================

def make_paper_record_from_doc(doc) -> Dict[str, Any]:
    meta = doc.metadata or {}

    # title = meta.get("title") or "No title"
    title = (
    meta.get("title")
    or meta.get("Title")
    or (doc.page_content.split("\n")[0] if doc.page_content else "No title")
)

    authors = meta.get("authors") or []
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]

    summary = doc.page_content or ""
    clean_abstract = prepare_abstract_for_llm(summary)

    url = None
    pdf = None
    for v in meta.values():
        s = unescape(str(v))
        if not url and (m := ARXIV_ABS_RE.search(s)):
            url = f"https://arxiv.org/abs/{m.group(1)}"
        if not pdf and (m := ARXIV_PDF_RE.search(s)):
            pdf = f"https://arxiv.org/pdf/{m.group(1)}.pdf"

    date = parse_date(meta)
    llm_out = generate_llm_summary_for_paper(title, clean_abstract)

    return {
        "title": title,
        "authors": authors,
        "url": url,
        "pdf": pdf,
        "date": date,
        "raw_summary": summary,
        "snippet": clean_abstract,
        "llm_overview": llm_out["overview"],
        "llm_key_points": llm_out["key_points"],
    }


# ================== AUTHOR FILTER ==================

def filter_papers_by_author(papers: List[Dict[str, Any]], author_query: str) -> List[Dict[str, Any]]:
    aq = author_query.lower().strip()
    return [
        p for p in papers
        if any(aq in a.lower() for a in (p.get("authors") or []))
    ]


# ================== MAIN API (USED BY UI) ==================

def get_research_papers(
    query: str,
    mode: str = "topic",
    load_max_docs: int = DEFAULT_LOAD_MAX_DOCS,
    per_topic: int = DEFAULT_PER_TOPIC,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:

    retriever = ArxivRetriever(
        load_max_docs=load_max_docs,
        load_all_available_meta=True,
    )

    docs = []

    if mode == "author":
        docs = retriever.invoke(f'au:"{query}"') or []
    else:
        docs = retriever.invoke(query) or []

    docs = docs[:load_max_docs]

    papers = []
    for d in docs:
        try:
            papers.append(make_paper_record_from_doc(d))
        except Exception:
            pass

    # keyword filtering only
    if mode == "keywords":
        q = query.lower()
        papers = [
            p for p in papers
            if q in p.get("title", "").lower()
            or q in p.get("raw_summary", "").lower()
        ]

    papers.sort(key=lambda p: p["date"] or datetime.min, reverse=True)
    return papers[:top_k]



# ================== CLI TEST ==================

if __name__ == "__main__":
    res = get_research_papers("Yann LeCun", mode="author")
    for r in res:
        print(r["title"], "-", r["authors"])


# =========================cursor code====================================


# """
# research_paper.py

# Interactive topic selection + improved URL extraction for arXiv results.

# Provides:
#  - DEFAULT_TOPICS
#  - get_trending_papers_for_topic(topic, ...)
#  - CLI entrypoint (if run as script)

# This module is import-safe (it will not raise on import if the Arxiv retriever is unavailable).

# """
# import os
# from dotenv import load_dotenv
# load_dotenv()

# from langchain_groq import ChatGroq
# # model=os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
# from langchain_groq import ChatGroq
# import os

# llm = ChatGroq(
#     api_key=os.environ["GROQ_API_KEY"],
#     model="qwen/qwen3-32b",
#     temperature=0.2
# )


# from typing import List, Dict, Any
# from collections import OrderedDict
# from datetime import datetime
# import argparse
# import json
# import sys
# import time
# import logging
# import re
# from html import unescape

# try:
#     from langchain_community.retrievers import ArxivRetriever
#     _ARXIV_AVAILABLE = True
# except Exception:
#     ArxivRetriever = None
#     _ARXIV_AVAILABLE = False

# DEFAULT_TOPICS = [
#     "machine learning",
#     "deep learning",
#     "generative ai",
#     "transformers",
#     "computer vision",
#     "natural language processing",
#     "reinforcement learning",
#     "retrieval-augmented generation",
#     "agentic AI",
#     "model context protocol (MCP)",
#     "data engineering / lakehouses",
#     "statistics for data science",
# ]

# DEFAULT_LOAD_MAX_DOCS = 8
# DEFAULT_PER_TOPIC = 5
# DEFAULT_TOP_K = 25
# RETRIEVE_RETRIES = 3
# RETRIEVE_BACKOFF = 1.0

# logger = logging.getLogger("research_paper")
# handler = logging.StreamHandler(sys.stderr)
# formatter = logging.Formatter("[%(levelname)s] %(message)s")
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.setLevel(logging.INFO)
# ARXIV_ABS_RE = re.compile(r"https?://(?:www\.)?arxiv\.org/abs/([\w\-\./]+)", re.IGNORECASE)
# ARXIV_PDF_RE = re.compile(r"https?://(?:www\.)?arxiv\.org/(?:pdf|pdf/)?([\w\-\./]+)(?:\.pdf)?", re.IGNORECASE)
# ARXIV_ID_RE_1 = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b")
# ARXIV_ID_RE_2 = re.compile(r"\b([a-zA-Z\-]+\/\d{7}(?:v\d+)?)\b")
# ARXIV_ID_RE_3 = re.compile(r"\b(\d{7}(?:v\d+)?)\b")


# def parse_date_from_metadata(meta: Dict[str, Any]) -> datetime:
#     for key in ("Updated", "UpdatedDate", "updated", "Published", "published", "PublishedDate", "created", "created_at"):
#         if key in meta and meta[key]:
#             val = meta[key]
#             try:
#                 return datetime.fromisoformat(str(val))
#             except Exception:
#                 for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d %b %Y", "%d %B %Y"):
#                     try:
#                         return datetime.strptime(str(val), fmt)
#                     except Exception:
#                         continue
#     return None


# def _iter_metadata_values(meta: Dict[str, Any]):
#     for k, v in meta.items():
#         if v is None:
#             continue
#         if isinstance(v, (list, tuple, set)):
#             for item in v:
#                 if item is None:
#                     continue
#                 yield str(item)
#         elif isinstance(v, dict):
#             for nested in v.values():
#                 if nested is not None:
#                     yield str(nested)
#         else:
#             yield str(v)


# def find_arxiv_links_and_ids(meta: Dict[str, Any], doc_text: str = "") -> Dict[str, str]:
#     found_abs = None
#     found_pdf = None

#     for val in _iter_metadata_values(meta):
#         s = unescape(val)
#         m_abs = ARXIV_ABS_RE.search(s)
#         if m_abs:
#             found_abs = f"https://arxiv.org/abs/{m_abs.group(1)}"
#         m_pdf = ARXIV_PDF_RE.search(s)
#         if m_pdf:
#             g = m_pdf.group(1)
#             if s.strip().lower().endswith(".pdf"):
#                 found_pdf = m_pdf.group(0)
#             else:
#                 found_pdf = f"https://arxiv.org/pdf/{g}.pdf"
#         if not found_abs:
#             m_id = ARXIV_ID_RE_1.search(s)
#             if m_id:
#                 iid = m_id.group(1)
#                 found_abs = f"https://arxiv.org/abs/{iid}"
#                 if not found_pdf:
#                     found_pdf = f"https://arxiv.org/pdf/{iid}.pdf"
#         if not found_abs:
#             m_id2 = ARXIV_ID_RE_2.search(s)
#             if m_id2:
#                 iid = m_id2.group(1)
#                 found_abs = f"https://arxiv.org/abs/{iid}"
#                 if not found_pdf:
#                     found_pdf = f"https://arxiv.org/pdf/{iid}.pdf"
#         if found_abs and found_pdf:
#             break

#     if (not found_abs or not found_pdf) and doc_text:
#         txt = unescape(doc_text)
#         m_abs = ARXIV_ABS_RE.search(txt)
#         if m_abs and not found_abs:
#             found_abs = f"https://arxiv.org/abs/{m_abs.group(1)}"
#         m_pdf = ARXIV_PDF_RE.search(txt)
#         if m_pdf and not found_pdf:
#             g = m_pdf.group(1)
#             found_pdf = m_pdf.group(0) if txt.strip().lower().endswith(".pdf") else f"https://arxiv.org/pdf/{g}.pdf"
#         if not found_abs:
#             m_id = ARXIV_ID_RE_1.search(txt)
#             if m_id:
#                 iid = m_id.group(1)
#                 found_abs = found_abs or f"https://arxiv.org/abs/{iid}"
#                 found_pdf = found_pdf or f"https://arxiv.org/pdf/{iid}.pdf"
#         if not found_abs:
#             m_id2 = ARXIV_ID_RE_2.search(txt)
#             if m_id2:
#                 iid = m_id2.group(1)
#                 found_abs = found_abs or f"https://arxiv.org/abs/{iid}"
#                 found_pdf = found_pdf or f"https://arxiv.org/pdf/{iid}.pdf"
#         if not found_abs:
#             m_id3 = ARXIV_ID_RE_3.search(txt)
#             if m_id3:
#                 iid = m_id3.group(1)
#                 found_abs = found_abs or f"https://arxiv.org/abs/{iid}"
#                 found_pdf = found_pdf or f"https://arxiv.org/pdf/{iid}.pdf"

#     return {"abs": found_abs, "pdf": found_pdf}


# def make_paper_record_from_doc(doc) -> Dict[str, Any]:
#     meta = getattr(doc, "metadata", {}) or {}
#     title = meta.get("Title") or meta.get("title") or meta.get("title_txt") or (
#         doc.page_content.split("\n", 1)[0] if getattr(doc, "page_content", None) else "No title"
#     )

#     authors = meta.get("Authors") or meta.get("authors") or meta.get("Author") or []
#     if isinstance(authors, str):
#         authors_list = [a.strip() for a in authors.replace("\n", ",").split(",") if a.strip()]
#     elif isinstance(authors, (list, tuple)):
#         authors_list = [str(a) for a in authors]
#     else:
#         authors_list = []

#     raw_pdf = meta.get("Pdf_url") or meta.get("pdf_url") or meta.get("pdf") or meta.get("link") or meta.get("url")
#     entry_id = meta.get("EntryId") or meta.get("id") or meta.get("entry_id") or raw_pdf or title

#     summary = getattr(doc, "page_content", "") or ""
#     snippet = (summary[:700] + "...") if len(summary) > 700 else summary

#     date = parse_date_from_metadata(meta)

#     categories_raw = meta.get("Categories") or meta.get("categories") or meta.get("Category") or meta.get("tags") or []
#     if isinstance(categories_raw, (list, tuple)):
#         categories = ", ".join([str(c) for c in categories_raw])
#     else:
#         categories = str(categories_raw) if categories_raw else ""

#     urls = find_arxiv_links_and_ids(meta, doc_text=summary)

#     url = urls.get("abs") or None
#     pdf = urls.get("pdf") or (raw_pdf if (isinstance(raw_pdf, str) and raw_pdf.startswith("http")) else None)

#     if not url and entry_id:
#         eid = str(entry_id).strip()
#         eid = re.sub(r"^arXiv[:\s]*", "", eid, flags=re.IGNORECASE)
#         if ARXIV_ID_RE_1.match(eid) or ARXIV_ID_RE_2.match(eid) or ARXIV_ID_RE_3.match(eid):
#             url = f"https://arxiv.org/abs/{eid}"
#             if not pdf:
#                 pdf = f"https://arxiv.org/pdf/{eid}.pdf"

#     if not url and not pdf:
#         logger.debug("No arXiv URL/pdf found for title: %s; metadata keys: %s", title, list(meta.keys()))

#     return {
#         "entry_id": entry_id,
#         "title": title,
#         "authors": authors_list,
#         "pdf": pdf,
#         "url": url,
#         "snippet": snippet,
#         "raw_summary": summary,
#         "date": date,
#         "metadata": meta,
#         "categories": categories,
#     }
# def prepare_abstract_for_llm(text: str, max_words: int = 250) -> str:
#     if not text:
#         return ""

#     # Remove extra spaces and newlines
#     text = " ".join(text.split())

#     # Keep only first N words
#     words = text.split()
#     if len(words) > max_words:
#         text = " ".join(words[:max_words])

#     return text.strip()

# def generate_llm_summary_for_paper(
#     title: str,
#     abstract: str,
#     model: str = "qwen/qwen3-32b",
# ) -> Dict[str, Any]:
#     """
#     Uses LLM to generate:
#     - 2-line paper overview
#     - 3–5 bullet points
#     """

#     if not abstract or len(abstract.split()) < 40:
#         return {"overview": "", "key_points": []}

#     prompt = f"""
# You are an expert research analyst.

# TASK:
# Given the research paper abstract below, do the following:

# 1. Write EXACTLY TWO LINES explaining what the paper is about.
#    - Do NOT copy sentences from the abstract
#    - Explain in simple technical language
#    - Focus on the main idea of the paper

# 2. Provide 3 to 5 bullet points describing:
#    - Main contributions
#    - Key methods or ideas
#    - Why the paper is useful

# STRICT RULES:
# - Output MUST be valid JSON
# - Do NOT include extra text
# - Do NOT include headings
# - Do NOT repeat the abstract
# - Keep the overview to exactly 2 lines

# JSON FORMAT (MANDATORY):
# {{
#   "overview": "line 1\\nline 2",
#   "key_points": [
#     "point 1",
#     "point 2",
#     "point 3"
#   ]
# }}

# ABSTRACT:
# \"\"\"
# {abstract}
# \"\"\"
# """
    

#     llm = ChatGroq(
#         api_key=os.getenv("GROQ_API_KEY"),
#         model=model,
#         temperature=0.2
#     )

#     response = llm.invoke(prompt)
#     text = response.content if hasattr(response, "content") else str(response)

#     try:
#         return json.loads(text)
#     except Exception:
#         # Safe fallback
#         return {"overview": "", "key_points": []}

# def get_trending_papers_for_topic(
#     topic: str,
#     load_max_docs: int = DEFAULT_LOAD_MAX_DOCS,
#     per_topic: int = DEFAULT_PER_TOPIC,
#     top_k: int = DEFAULT_TOP_K,
# ) -> List[Dict[str, Any]]:
#     """
#     Fetch trending arXiv papers for a topic.

#     Raises:
#         RuntimeError if the ArxivRetriever dependency is not available.
#     """
#     if not _ARXIV_AVAILABLE:
#         raise RuntimeError(
#             "ArxivRetriever (langchain_community) is not available. "
#             "Install it via `pip install langchain-community` or ensure the import path is correct."
#         )

#     retriever = ArxivRetriever(
#         load_max_docs=load_max_docs,
#         load_all_available_meta=True,
#     )

#     aggregated: "OrderedDict[str, Dict]" = OrderedDict()
#     docs = None
#     for attempt in range(1, RETRIEVE_RETRIES + 1):
#         try:
#             docs = retriever.invoke(topic)
#             break
#         except Exception as e:
#             logger.warning("ArXiv query error for '%s' (attempt %d/%d): %s", topic, attempt, RETRIEVE_RETRIES, e)
#             if attempt < RETRIEVE_RETRIES:
#                 backoff = RETRIEVE_BACKOFF * (2 ** (attempt - 1))
#                 time.sleep(backoff)
#             else:
#                 logger.error("Giving up on topic '%s' after %d attempts.", topic, RETRIEVE_RETRIES)
#     if not docs:
#         return []

#     docs_slice = docs[:per_topic] if docs else []
#     for doc in docs_slice:
#         # rec = make_paper_record_from_doc(doc)
#         rec = make_paper_record_from_doc(doc)

#         llm_result = generate_llm_summary_for_paper(
#         title=rec.get("title", ""),
#         abstract=rec.get("raw_summary", "")
# )

#         rec["llm_overview"] = llm_result.get("overview", "")
#         rec["llm_key_points"] = llm_result.get("key_points", [])

#         key = str(rec.get("entry_id") or rec.get("pdf") or rec.get("title"))
#         if key not in aggregated:
#             aggregated[key] = rec
#         else:
#             existing = aggregated[key]
#             if len(rec.get("authors", [])) > len(existing.get("authors", [])):
#                 existing["authors"] = rec["authors"]
#             if not existing.get("snippet") and rec.get("snippet"):
#                 existing["snippet"] = rec["snippet"]
#             if rec.get("date") and (not existing.get("date") or rec["date"] > existing.get("date")):
#                 existing["date"] = rec["date"]
#             if not existing.get("categories") and rec.get("categories"):
#                 existing["categories"] = rec["categories"]
#             if not existing.get("url") and rec.get("url"):
#                 existing["url"] = rec["url"]
#             if not existing.get("pdf") and rec.get("pdf"):
#                 existing["pdf"] = rec["pdf"]

#     papers = list(aggregated.values())
#     papers.sort(key=lambda p: p["date"] or datetime.min, reverse=True)
#     return papers[:top_k]


# def print_papers_grouped_by_topic(grouped: Dict[str, List[Dict[str, Any]]], out_stream=sys.stdout):
#     for topic, papers in grouped.items():
#         print(f"\n\n## Topic: {topic}\n", file=out_stream)
#         if not papers:
#             print("_No papers found for this topic._\n", file=out_stream)
#             continue
#         for i, p in enumerate(papers, start=1):
#             print(f"### {i}. {p.get('title', 'No title')}", file=out_stream)
#             if p.get("authors"):
#                 print(f"**Authors:** {', '.join(p.get('authors', []))}", file=out_stream)
#             if p.get("date"):
#                 try:
#                     print(f"**Published/Updated:** {p['date'].date().isoformat()}", file=out_stream)
#                 except Exception:
#                     print(f"**Published/Updated:** {p['date']}", file=out_stream)
#             if p.get("categories"):
#                 print(f"**Categories:** {p.get('categories')}", file=out_stream)
#             if p.get("url"):
#                 print(f"[Abstract]({p.get('url')})", file=out_stream)
#             else:
#                 print(f"_No abstract URL found_", file=out_stream)
#             if p.get("pdf"):
#                 print(f"[PDF]({p.get('pdf')})", file=out_stream)
#             else:
#                 print(f"_No PDF URL found_", file=out_stream)
#             if p.get("snippet"):
#                 print(f"> {p.get('snippet')}", file=out_stream)
#             print("\n---\n", file=out_stream)


# def prompt_select_topics(default_topics: List[str]) -> List[str]:
#     """
#     Interactive console prompt to choose topics from default_topics.
#     Returns the chosen list of topic strings.
#     """
#     print("Select topics by entering comma-separated numbers (e.g. 1,3,5) or type 'all' to select all topics.")
#     print("Available topics:")
#     for idx, t in enumerate(default_topics, start=1):
#         print(f"  {idx}. {t}")
#     try:
#         selection = input("Your selection: ").strip()
#     except Exception:
#         # if input() fails for some reason (non-interactive), default to all
#         logger.info("No interactive input available; defaulting to all topics.")
#         return default_topics

#     if not selection:
#         logger.info("Empty selection; defaulting to all topics.")
#         return default_topics

#     if selection.lower() in ("all", "a", "*"):
#         return default_topics

#     chosen = []
#     parts = [p.strip() for p in selection.split(",") if p.strip()]
#     for part in parts:
#         try:
#             n = int(part)
#             if 1 <= n <= len(default_topics):
#                 chosen.append(default_topics[n - 1])
#             else:
#                 logger.warning("Ignored invalid topic number: %s", part)
#         except ValueError:
#             logger.warning("Ignored invalid selection token: %s", part)

#     if not chosen:
#         logger.info("No valid selections; defaulting to all topics.")
#         return default_topics

#     return chosen


# def main():
#     parser = argparse.ArgumentParser(description="Fetch trending arXiv papers for chosen topics (prints to console).")
#     parser.add_argument("--topics", type=str, help="Comma-separated list of topics (overrides interactive selection).")
#     parser.add_argument("--topics-file", type=str, help="Path to a newline-separated topics file.")
#     parser.add_argument("--per-topic", type=int, default=DEFAULT_PER_TOPIC, help="Per-topic hits to keep.")
#     parser.add_argument("--load-max-docs", type=int, default=DEFAULT_LOAD_MAX_DOCS, help="ArXivRetriever load_max_docs.")
#     parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Max papers to show per topic.")
#     parser.add_argument("--output", type=str, help="Optional output file path (if provided, results will be written there in markdown).")
#     parser.add_argument("--json", action="store_true", help="Also output JSON to stdout (after markdown).")
#     parser.add_argument("--debug-missing-urls", action="store_true", help="Print debugging info for records missing URLs.")

#     args, _unknown = parser.parse_known_args()

#     # Determine topics: CLI overrides interactive prompt
#     if args.topics_file:
#         try:
#             with open(args.topics_file, "r", encoding="utf-8") as f:
#                 topics = [line.strip() for line in f if line.strip()]
#         except Exception as e:
#             logger.error("Failed to read topics file: %s", e)
#             return
#     elif args.topics:
#         topics = [t.strip() for t in args.topics.split(",") if t.strip()]
#     else:
#         # Interactive selection from default topics
#         topics = prompt_select_topics(DEFAULT_TOPICS)

#     logger.info("Selected topics: %s", topics)
#     logger.info("per_topic=%d, load_max_docs=%d, top_k=%d", args.per_topic, args.load_max_docs, args.top_k)

#     grouped_results: Dict[str, List[Dict[str, Any]]] = OrderedDict()
#     for t in topics:
#         logger.info("Fetching papers for topic: %s", t)
#         try:
#             papers = get_trending_papers_for_topic(
#                 topic=t,
#                 load_max_docs=int(args.load_max_docs),
#                 per_topic=int(args.per_topic),
#                 top_k=int(args.top_k),
#             )
#         except Exception as e:
#             logger.error("Error fetching papers for topic %s: %s", t, e)
#             papers = []
#         grouped_results[t] = papers

#     # Output to file if requested (single markdown file with grouped sections)
#     if args.output:
#         try:
#             with open(args.output, "w", encoding="utf-8") as f:
#                 print_papers_grouped_by_topic(grouped_results, out_stream=f)
#             logger.info("Results written to %s", args.output)
#         except Exception as e:
#             logger.error("Failed to write output file: %s", e)

#     # Always print to stdout
#     print_papers_grouped_by_topic(grouped_results, out_stream=sys.stdout)

#     # JSON dump if requested (grouped)
#     if args.json:
#         serializable = {}
#         for topic, papers in grouped_results.items():
#             serializable[topic] = []
#             for p in papers:
#                 d = dict(p)
#                 if isinstance(d.get("date"), datetime):
#                     d["date"] = d["date"].isoformat()
#                 serializable[topic].append(d)
#         print("\n\nJSON OUTPUT:\n")
#         print(json.dumps(serializable, indent=2, ensure_ascii=False))

#     # debug missing urls
#     if args.debug_missing_urls:
#         missing = []
#         for topic, papers in grouped_results.items():
#             for p in papers:
#                 if not p.get("url") and not p.get("pdf"):
#                     missing.append((topic, p))
#         if missing:
#             logger.info("Records missing any URL/pdf: %d", len(missing))
#             for topic, m in missing:
#                 logger.info("Topic: %s | Title: %s | entry_id: %s | metadata keys: %s", topic, m.get("title"), m.get("entry_id"), list(m.get("metadata", {}).keys()))
#         else:
#             logger.info("All records have at least one URL or PDF link.")


# if __name__ == "__main__":
#     main()




# """
# research_paper.py

# Fetch trending arXiv papers and generate:
# - 2-line LLM overview
# - 3–5 bullet key points per paper
# """

# ================== ENV & LLM SETUP ==================

# import os
# import json
# import time
# import re
# import sys
# import logging
# from typing import List, Dict, Any
# from collections import OrderedDict
# from datetime import datetime
# from html import unescape
# from dotenv import load_dotenv

# load_dotenv()

# from langchain_groq import ChatGroq

# GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
# GROQ_API_KEY = os.environ["GROQ_API_KEY"]

# # ================== LOGGING ==================

# logger = logging.getLogger("research_paper")
# handler = logging.StreamHandler(sys.stderr)
# handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
# logger.addHandler(handler)
# logger.setLevel(logging.INFO)

# # ================== ARXIV SETUP ==================

# try:
#     from langchain_community.retrievers import ArxivRetriever
#     _ARXIV_AVAILABLE = True
# except Exception:
#     ArxivRetriever = None
#     _ARXIV_AVAILABLE = False

# DEFAULT_TOPICS = [
#     "machine learning",
#     "deep learning",
#     "generative ai",
#     "transformers",
#     "computer vision",
#     "natural language processing",
#     "reinforcement learning",
#     "retrieval-augmented generation",
#     "agentic AI",
#     "model context protocol (MCP)",
# ]

# DEFAULT_LOAD_MAX_DOCS = 8
# DEFAULT_PER_TOPIC = 5
# DEFAULT_TOP_K = 10

# # ================== REGEX ==================

# ARXIV_ABS_RE = re.compile(r"https?://arxiv\.org/abs/([\w\.\-]+)", re.I)
# ARXIV_PDF_RE = re.compile(r"https?://arxiv\.org/pdf/([\w\.\-]+)\.pdf", re.I)

# # ================== HELPERS ==================

# def parse_date(meta: Dict[str, Any]) -> datetime | None:
#     for k in ("published", "updated", "created"):
#         if k in meta:
#             try:
#                 return datetime.fromisoformat(str(meta[k]))
#             except Exception:
#                 pass
#     return None


# def prepare_abstract_for_llm(text: str, max_words: int = 220) -> str:
#     if not text:
#         return ""
#     text = " ".join(text.split())
#     words = text.split()
#     return " ".join(words[:max_words])


# # ================== LLM SUMMARY ==================

# def generate_llm_summary_for_paper(title: str, abstract: str) -> Dict[str, Any]:
#     if not abstract or len(abstract.split()) < 15:
#         return {"overview": "", "key_points": []}

#     prompt = f"""
# You are an expert machine learning researcher.

# TASK:
# 1. Write EXACTLY TWO LINES explaining what this paper is about.
# 2. Write 3–5 bullet points describing the main contributions.

# RULES:
# - Do NOT copy the abstract
# - Do NOT add explanations
# - Output ONLY valid JSON

# FORMAT:
# {{
#   "overview": "line 1\\nline 2",
#   "key_points": ["point 1", "point 2", "point 3"]
# }}

# ABSTRACT:
# \"\"\"
# {abstract}
# \"\"\"
# """

#     llm = ChatGroq(
#         api_key=GROQ_API_KEY,
#         model=GROQ_MODEL,
#         temperature=0.2
#     )

#     try:
#         resp = llm.invoke(prompt)
#         data = json.loads(resp.content)

#         # enforce structure
#         lines = [l.strip() for l in data.get("overview", "").splitlines() if l.strip()]
#         overview = "\n".join(lines[:2])

#         points = [p.strip() for p in data.get("key_points", []) if p.strip()][:5]

#         return {
#             "overview": overview,
#             "key_points": points
#         }
#     except Exception as e:
#         logger.warning("LLM summary failed: %s", e)
#         return {"overview": "", "key_points": []}


# # ================== PAPER EXTRACTION ==================

# def make_paper_record_from_doc(doc) -> Dict[str, Any]:
#     meta = doc.metadata or {}

#     title = meta.get("title") or meta.get("Title") or "No title"
#     authors = meta.get("authors") or meta.get("Authors") or []
#     if isinstance(authors, str):
#         authors = [a.strip() for a in authors.split(",") if a.strip()]

#     summary = doc.page_content or ""
#     clean_abstract = prepare_abstract_for_llm(summary)

#     url = None
#     pdf = None

#     for val in meta.values():
#         s = unescape(str(val))
#         if not url:
#             m = ARXIV_ABS_RE.search(s)
#             if m:
#                 url = f"https://arxiv.org/abs/{m.group(1)}"
#         if not pdf:
#             m = ARXIV_PDF_RE.search(s)
#             if m:
#                 pdf = f"https://arxiv.org/pdf/{m.group(1)}.pdf"

#     date = parse_date(meta)

#     llm_out = generate_llm_summary_for_paper(title, clean_abstract)

#     return {
#         "title": title,
#         "authors": authors,
#         "url": url,
#         "pdf": pdf,
#         "date": date,
#         "raw_summary": summary,
#         "snippet": clean_abstract,
#         "llm_overview": llm_out["overview"],
#         "llm_key_points": llm_out["key_points"],
#     }


# # ================== MAIN FETCH FUNCTION ==================

# def get_trending_papers_for_topic(
#     topic: str,
#     load_max_docs: int = DEFAULT_LOAD_MAX_DOCS,
#     per_topic: int = DEFAULT_PER_TOPIC,
#     top_k: int = DEFAULT_TOP_K,
# ) -> List[Dict[str, Any]]:

#     if not _ARXIV_AVAILABLE:
#         raise RuntimeError("ArxivRetriever not available")

#     retriever = ArxivRetriever(
#         load_max_docs=load_max_docs,
#         load_all_available_meta=True,
#     )

#     docs = retriever.invoke(topic) or []
#     docs = docs[:per_topic]

#     papers = []
#     for doc in docs:
#         papers.append(make_paper_record_from_doc(doc))

#     papers.sort(key=lambda p: p["date"] or datetime.min, reverse=True)
#     return papers[:top_k]


# # ================== CLI (OPTIONAL) ==================

# if __name__ == "__main__":
#     topic = DEFAULT_TOPICS[0]
#     results = get_trending_papers_for_topic(topic)

#     for i, p in enumerate(results, 1):
#         print(f"\n{i}. {p['title']}")
#         print(p["llm_overview"])
#         for b in p["llm_key_points"]:
#             print("-", b)

