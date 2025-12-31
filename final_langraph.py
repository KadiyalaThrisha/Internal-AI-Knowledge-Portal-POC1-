
#!/usr/bin/env python3

import os
import csv
import re
import sys
import time
import json
import traceback
import inspect
import asyncio
import feedparser
import calendar
from html import unescape
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Callable, Optional, Tuple

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_classic.vectorstores import FAISS
# from langchain_classic.embeddings import HuggingFaceEmbeddings
from langchain_classic.docstore.document import Document

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader




import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from streamlit import title
from tqdm import tqdm
# from langchain_classic.document_loaders import UnstructuredURLLoader
# from langchain_community.document_loaders import UnstructuredURLLoader
from dotenv import load_dotenv
load_dotenv()




CSV_PATH = "ai_rss_results.csv"
RSS_FAISS_DIR = "faiss_index_local"
# BLOG_FAISS_DIR = "./kdnuggets_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"



# ================= BLOG CONFIG (SINGLE SOURCE) =================

BLOG_START_URL = "https://www.kdnuggets.com/tag/artificial-intelligence"
BLOG_FAISS_DIR = "./kdnuggets_faiss"
BLOG_REQUEST_DELAY = 1.0
BLOG_MAX_PAGES = 25
blog_faiss_store: Optional[FAISS] = None

# ================= ANALYTICS VIDHYA BLOG CONFIG =================

AV_START_URL = "https://www.analyticsvidhya.com/blog/category/artificial-intelligence/"
AV_FAISS_DIR = "./faiss_blogs/analyticsvidhya"
AV_MAX_PAGES = 25

# ================= MACHINE LEARNING MASTERY CONFIG =================

MLM_START_URLS = [
    "https://machinelearningmastery.com/start-here/"
]

MLM_FAISS_DIR = "./faiss_blogs/ml_mastery"
MLM_MAX_PAGES = 150




def extract_av_article_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        if href.startswith("/"):
            href = urljoin(base_url, href)

        if "analyticsvidhya.com/blog/" not in href:
            continue

        # Filter out category, tag, author pages
        if "/category/" in href or "/tag/" in href or "/author/" in href:
            continue

        # Keep only real articles
        if re.search(r"/\d{4}/", href):
            links.add(href.split("?")[0].rstrip("/"))

    return sorted(links)


def crawl_analytics_vidhya(start_url, max_pages=AV_MAX_PAGES):
    to_visit = [start_url]
    visited = set()
    article_urls = set()

    print(f"[AV crawl] START → {start_url}")

    for page_no in range(max_pages):
        if not to_visit:
            break

        url = to_visit.pop(0)
        if url in visited:
            continue

        print(f"[AV crawl] ({page_no+1}/{max_pages}) Fetching: {url}")

        try:
            html = fetch_html(url)
        except Exception as e:
            print(f"[AV crawl][WARN] failed: {e}")
            visited.add(url)
            continue

        visited.add(url)

        found = extract_av_article_links(html, url)
        article_urls.update(found)

        print(f"[AV crawl] Found {len(found)} articles (total={len(article_urls)})")

        # Pagination: older posts
        soup = BeautifulSoup(html, "html.parser")
        next_link = soup.find("a", class_="next")
        if next_link and next_link.get("href"):
            next_url = urljoin(url, next_link["href"])
            if next_url not in visited:
                to_visit.append(next_url)

        time.sleep(BLOG_REQUEST_DELAY)

    print(f"[AV crawl] DONE. Total articles: {len(article_urls)}")
    return sorted(article_urls)





import time

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ================= BLOG CONFIG ================

# BLOG_CONFIG = {
#     "KDnuggets": {
#         "faiss_dir": "./kdnuggets_faiss",
#         "start_url": "https://www.kdnuggets.com/tag/artificial-intelligence",
#         "mode": "crawl",
#         "max_pages": 25,
#     },
# }




CSV_META_MAP: Dict[str, Dict[str, str]] = {}
rss_faiss_store: Optional[FAISS] = None
# blog_faiss_store: Optional[FAISS] = None
from pydantic import BaseModel

class WorkflowState(BaseModel):
    request_id: str
    query: str

    selected_tool: str = "rss"
  

    status: str = "initialized"
    retrieved_docs: List[Dict[str, Any]] = []
    retrieved_text: str = ""
    generated_post: Any = ""
    sources: List[Dict[str, str]] = []
    error: Optional[str] = None


def build_analytics_vidhya_index(state: WorkflowState) -> WorkflowState:
    state.status = "indexing_blog"
    try:
        print("🚀 Building Analytics Vidhya blog index")

        urls = crawl_analytics_vidhya(
            start_url=AV_START_URL,
            max_pages=AV_MAX_PAGES
        )

        if not urls:
            raise RuntimeError("No Analytics Vidhya URLs found")

        docs, failed = load_articles_to_docs(urls)

        if not docs:
            raise RuntimeError("No Analytics Vidhya documents loaded")

        build_faiss_vectorstore_for_blog(
            docs,
            persist_directory=AV_FAISS_DIR
        )

        state.status = "indexed_blog"
        print("✅ Analytics Vidhya FAISS built successfully")
        return state

    except Exception as e:
        state.status = "error"
        state.error = str(e)
        return state



DAYS_WINDOW = 7
MIN_PER_FEED = 5

AI_FEEDS = [
    "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml",
    "https://venturebeat.com/category/ai/feed/",
    "https://www.analyticsvidhya.com/blog/category/artificial-intelligence/feed/",
    "https://news.google.com/rss/search?q=artificial+intelligence&hl=en-IN&gl=IN&ceid=IN:en",
    "https://techcrunch.com/feed/",
    "https://yourstory.com/feed",
    "https://economictimes.indiatimes.com/tech/rssfeeds/13357270.cms",
    "https://blog.langchain.dev/rss/",
    "https://www.llamaindex.ai/blog/rss.xml",
    "https://openai.com/blog/rss/",
    "https://huggingface.co/blog/feed.xml",
    "https://mistral.ai/news/feed.xml",
    "https://www.anthropic.com/news/feed.xml",
    "https://vercel.com/changelog/rss",
    "https://weaviate.io/blog/rss.xml",
    "https://www.pinecone.io/rss.xml",
    "https://milvus.io/blog/index.xml",
    "https://news.ycombinator.com/rss",
]

def rebuild_rss_faiss_always(state: WorkflowState) -> WorkflowState:
    """
    ALWAYS rebuild RSS FAISS from latest RSS feeds.
    This ensures no stale news.
    """

    state.status = "rebuilding_rss"

    try:
        print("[RSS] Collecting latest RSS feeds → CSV")
        collect_rss_and_write_csv(
            output_csv=CSV_PATH,
            days_window=DAYS_WINDOW,
            min_per_feed=MIN_PER_FEED
        )

        print("[RSS] Loading pages + cleaning")
        links, meta_map = read_csv_metadata(CSV_PATH)

        docs, failed = load_and_clean_pages(links, meta_map)
        if not docs:
            raise RuntimeError("No RSS documents loaded")

        print("[RSS] Rebuilding FAISS index (overwrite)")
        global rss_faiss_store
        rss_faiss_store = build_faiss_from_docs_and_save(
            docs,
            RSS_FAISS_DIR,
            EMBEDDING_MODEL_NAME
        )

        state.status = "rss_rebuilt"
        return state

    except Exception as e:
        state.status = "error"
        state.error = f"RSS rebuild failed: {repr(e)}\n{traceback.format_exc()}"
        return state

def extract_mlm_article_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        if href.startswith("/"):
            href = urljoin(base_url, href)

        if "machinelearningmastery.com" not in href:
            continue

        # Filter category pages
        if any(x in href for x in [
            "/category/", "/tag/", "/author/", "/about", "/contact"
        ]):
            continue

        # Keep real articles (long slugs)
        if href.count("/") >= 4:
            links.add(href.split("?")[0].rstrip("/"))

    return sorted(links)


from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

def crawl_machine_learning_mastery(start_urls, max_pages=300):
    visited = set()
    collected_urls = []

    for start_url in start_urls:
        print(f"[MLM] Crawling hub: {start_url}")

        resp = requests.get(start_url, timeout=20)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # ✅ Extract only MachineLearningMastery article links
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(start_url, href)

            parsed = urlparse(full_url)

            if (
                "machinelearningmastery.com" in parsed.netloc
                and full_url.startswith("https://machinelearningmastery.com/")
                and full_url not in visited
                and not any(x in full_url for x in [
                    "/category/",
                    "/tag/",
                    "/author/",
                    "/page/",
                    "#",
                ])
            ):
                links.append(full_url)

        print(f"[MLM] Found {len(links)} candidate article links")

        for url in links:
            if len(collected_urls) >= max_pages:
                break
            if url not in visited:
                visited.add(url)
                collected_urls.append(url)

    print(f"[MLM] Total articles selected: {len(collected_urls)}")
    return collected_urls


def build_machine_learning_mastery_index(state: WorkflowState) -> WorkflowState:
    state.status = "indexing_blog"
    try:
        print("🚀 Building Machine Learning Mastery index")

        urls = crawl_machine_learning_mastery(
            start_urls=MLM_START_URLS,
            max_pages=MLM_MAX_PAGES
        )

        if not urls:
            raise RuntimeError("No Machine Learning Mastery URLs found")

        docs, failed = load_articles_to_docs(urls)

        if not docs:
            raise RuntimeError("No Machine Learning Mastery docs loaded")

        # ✅ IMPORTANT: use blog_name instead of chunk params
        build_faiss_vectorstore_for_blog(
            docs=docs,
            persist_directory=MLM_FAISS_DIR,
        
        )

        # ✅ HARD VERIFICATION (this fixes your issue)
        index_path = os.path.join(MLM_FAISS_DIR, "index.faiss")
        if not os.path.exists(index_path):
            raise RuntimeError("FAISS index.faiss not created")

        state.status = "indexed_blog"
        print("✅ Machine Learning Mastery FAISS built successfully")
        return state

    except Exception as e:
        state.status = "error"
        state.error = str(e)
        print("❌ MLM indexing failed:", e)
        return state



# def load_blog_urls_from_rss(rss_url: str, limit: int = 30) -> list[str]:
#     feed = feedparser.parse(rss_url)
#     urls = []

#     for entry in feed.entries[:limit]:
#         link = entry.get("link")
#         if link:
#             urls.append(link)

#     return list(dict.fromkeys(urls))  # dedupe



def entry_published_dt(entry):
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    try:
        ts = int(calendar.timegm(parsed))
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        return None

def to_iso(dt: datetime):
    if not dt:
        return ""
    return dt.astimezone(timezone.utc).isoformat()

def clean_html_summary(raw_html: str) -> str:
    if not raw_html:
        return ""
    s = unescape(raw_html)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def collect_rss_and_write_csv(output_csv: str = CSV_PATH,
                              days_window: int = DAYS_WINDOW,
                              min_per_feed: int = MIN_PER_FEED) -> List[Dict[str, Any]]:
    feeds = list(AI_FEEDS)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days_window)

    all_articles = []
    seen_links = set()

    for feed_url in feeds:
        parsed_feed = feedparser.parse(feed_url)
        feed_title = parsed_feed.feed.get("title", feed_url)

        items = []
        for idx, entry in enumerate(parsed_feed.entries):
            link = (entry.get("link") or entry.get("id") or "").strip()
            if not link:
                continue
            pub_dt = entry_published_dt(entry)
            pub_ts = int(pub_dt.timestamp()) if pub_dt else 0
            items.append({
                "entry": entry,
                "link": link,
                "published_dt": pub_dt,
                "published_ts": pub_ts,
                "index": idx
            })

        recent = [it for it in items if it["published_dt"] and it["published_dt"] >= cutoff]
        recent.sort(key=lambda x: x["published_ts"], reverse=True)

        if len(recent) < min_per_feed:
            recent_links = {it["link"] for it in recent}
            candidates = [it for it in items if it["link"] not in recent_links]
            candidates.sort(key=lambda x: (x["published_ts"], -x["index"]), reverse=True)
            needed = min_per_feed - len(recent)
            for c in candidates:
                if needed <= 0:
                    break
                recent.append(c)
                needed -= 1

        selected = recent[:max(min_per_feed, len(recent))]

        for sel in selected:
            link = sel["link"]
            if link in seen_links:
                continue
            entry = sel["entry"]
            pub_dt = sel["published_dt"]
            pub_ts = sel["published_ts"] if sel["published_dt"] else 0

            published_str = entry.get("published", "") or entry.get("updated", "") or ""

            raw_summary = entry.get("summary", "") or entry.get("description", "") or ""
            summary = clean_html_summary(raw_summary)

            title = (entry.get("title") or "No title").strip()

            all_articles.append({
                "feed": feed_title,
                "title": title,
                "link": link,
                "published_iso": to_iso(pub_dt),
                "published_ts": pub_ts,
                "published_raw": published_str,
                "summary": summary
            })
            seen_links.add(link)

    all_articles.sort(key=lambda x: x["published_ts"], reverse=True)

    fieldnames = ["feed", "title", "link", "published_iso", "published_ts", "published_raw", "summary"]
    with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_articles:
            writer.writerow(row)

    return all_articles

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)





def remove_diagram_lines(text: str, min_words_keep: int = 4, punct_ratio_thresh: float = 0.4) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    cleaned_lines = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        if re.search(r'\bfig(?:ure)?\.?\s*\d+\b', low):
            continue
        if any(kw in low for kw in [
            "figure", "fig.", "diagram", "flowchart", "flow chart", "chart:", "table:", "image:",
            "illustration", "infographic", "caption:", "click to view", "download pdf", "view image",
            "see figure", "see diagram", "open image", "related image", "share this article",
            "embed", "powered by", "svg", "png", "jpg", "gif"
        ]):
            if len(s.split()) < 20:
                continue
        if re.fullmatch(r'[\W_]{3,}', s):
            continue
        words = s.split()
        if len(words) <= min_words_keep:
            non_alnum = re.sub(r'\w', '', s)
            punct_ratio = len(non_alnum) / max(1, len(s))
            if punct_ratio > punct_ratio_thresh:
                continue
            if re.search(r'^[\w\-\s\|,]{0,60}$', s) and ('|' in s or ',' in s or s.isupper()):
                continue
            cap_frac = sum(1 for w in words if w and w[0].isupper()) / max(1, len(words))
            if cap_frac > 0.6 and len(words) <= 6:
                continue
        if re.search(r'\b(share|twitter|facebook|linkedin|email|subscribe|subscribe to)\b', low) and len(words) <= 8:
            continue
        cleaned_lines.append(s)
    return "\n".join(cleaned_lines)

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    noisy_block_patterns = [
        r'(?is)table of contents.*',
        r'(?is)you might also like[:]?.*',
        r'(?is)related posts[:]?.*',
        r'(?is)more from.*',
    ]
    for pat in noisy_block_patterns:
        text = re.sub(pat, '', text)
    text = remove_diagram_lines(text)
    s = re.sub(r'\s+', ' ', text).strip()
    s = re.sub(r'^[^\w]+', '', s)
    s = re.sub(r'[^\w]+$', '', s)
    return s


def read_csv_metadata(csv_path: str) -> Tuple[List[str], Dict[str, Dict]]:
    links = []
    meta_map = {}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            link = (row.get("link") or "").strip()
            title = (row.get("title") or "").strip()
            rss_summary = (row.get("summary") or "").strip()
            if not link:
                continue
            links.append(link)
            meta_map[link] = {"title": title, "rss_summary": rss_summary}
    return links, meta_map

def load_and_clean_pages(links: List[str], meta_map: Dict[str, Dict]) -> Tuple[List[Document], List[Tuple[str,str]]]:
    docs = []
    failed = []
    for url in links:
        try:
            loader = WebBaseLoader(url)
            

            loaded = loader.load()
            if not loaded:
                failed.append((url, "empty"))
                continue
            doc = loaded[0]
            clean_content = clean_text(doc.page_content) if hasattr(doc, "page_content") else clean_text(str(doc))
            doc.page_content = clean_content
            csv_meta = meta_map.get(url, {})
            title = csv_meta.get("title") or doc.metadata.get("title") or ""
            rss_summary = csv_meta.get("rss_summary") or ""
            if getattr(doc.metadata, "source", "") == "Towards Data Science":
                final_summary = clean_text(clean_content[:600])
            else:
                if rss_summary and len(rss_summary.strip()) > 20:
                    final_summary = clean_text(rss_summary)
                else:
                    final_summary = clean_content[:600].strip()

            doc.metadata["title"] = title.strip()
            doc.metadata["link"] = url
            doc.metadata["summary"] = final_summary.strip()
            docs.append(doc)
        except Exception as e:
            failed.append((url, str(e)))
    return docs, failed

def build_faiss_from_docs_and_save(documents: List[Document], index_dir: str, model_name: str) -> FAISS:
    split_docs = text_splitter.split_documents(documents)
    if not split_docs:
        raise RuntimeError("No split documents to index.")
    for d in split_docs:
        d.metadata["title"] = (d.metadata.get("title") or "").strip()
        d.metadata["link"] = (d.metadata.get("link") or "").strip()
        if not d.metadata.get("summary"):
            d.metadata["summary"] = (clean_text(d.page_content)[:300]).strip()
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    store = FAISS.from_documents(split_docs, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    store.save_local(index_dir)
    return store

def build_rss_index_node(state: WorkflowState) -> WorkflowState:
    state.status = "indexing_rss"
    try:
        links, meta_map = read_csv_metadata(CSV_PATH)
        global CSV_META_MAP
        CSV_META_MAP = {}
        CSV_META_MAP.update(meta_map)
        docs, failed = load_and_clean_pages(links, meta_map)
        if not docs:
            raise RuntimeError("No docs loaded for indexing.")
        global rss_faiss_store
        rss_faiss_store = build_faiss_from_docs_and_save(docs, RSS_FAISS_DIR, EMBEDDING_MODEL_NAME)
        state.status = "indexed_rss"
    except Exception as e:
        state.status = "error"
        state.error = f"build_rss_index_node failed: {repr(e)}\n{traceback.format_exc()}"
    return state

# START_URL = BLOG_START_URL
REQUEST_DELAY = BLOG_REQUEST_DELAY
MAX_PAGES = BLOG_MAX_PAGES
HUGGINGFACE_EMBEDDING_MODEL = EMBEDDING_MODEL_NAME
# FAISS_PERSIST_DIR = BLOG_FAISS_DIR

def fetch_html(url):
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text

def extract_article_links_from_tag_page(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a", href=True)
    links = set()
    for a in anchors:
        href = a["href"].strip()
        href = urljoin(base_url, href)
        parsed = urlparse(href)
        if "kdnuggets.com" not in parsed.netloc:
            continue
        if re.search(r"/\d{4}/\d{2}/|/20\d{2}/|-[a-z0-9\-]+$", href):
            links.add(href.split("?")[0].rstrip("/"))
        else:
            if "/blog" in href or (len(parsed.path.split("/")) > 2 and "-" in parsed.path):
                links.add(href.split("?")[0].rstrip("/"))
    return sorted(links)

def find_pagination_next(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    for text in ["Next", "Older", "Older Posts", "Next »", "→", "older posts"]:
        el = soup.find("a", string=lambda s: s and text.lower() in s.lower())
        if el and el.get("href"):
            return urljoin(base_url, el["href"])
    el = soup.find("link", rel="next")
    if el and el.get("href"):
        return urljoin(base_url, el["href"])
    return None
def crawl_tag(start_url, max_pages=MAX_PAGES):
    to_visit = [start_url]
    visited = set()
    article_urls = set()

    print(f"[crawl] START url={start_url}, max_pages={max_pages}")

    for page_no in range(max_pages):
        if not to_visit:
            break

        url = to_visit.pop(0)
        if url in visited:
            continue

        print(f"[crawl] ({page_no+1}/{max_pages}) Fetching: {url}")

        try:
            html = fetch_html(url)
        except Exception as e:
            print(f"[crawl][WARN] failed to fetch {url}: {e}")
            visited.add(url)
            continue

        visited.add(url)

        found = extract_article_links_from_tag_page(html, url)
        article_urls.update(found)

        print(f"[crawl] Found {len(found)} links (total={len(article_urls)})")

        next_page = find_pagination_next(html, url)
        if next_page and next_page not in visited:
            to_visit.append(next_page)

        time.sleep(REQUEST_DELAY)

    print(f"[crawl] DONE. Total article URLs: {len(article_urls)}")
    return sorted(article_urls)



def load_articles_to_docs(urls, use_unstructured=True):
    docs = []
    failed = []
    # if use_unstructured:
    #     try:
    #         loader = UnstructuredURLLoader(urls=urls)
    #         docs = loader.load()
    #         return docs, failed
    #     except Exception as e:
    #         print(f"[info] Unstructured loader failed, falling back to WebBaseLoader: {e}")
    for url in tqdm(urls, desc="Loading URLs"):
        try:
            loader = WebBaseLoader(url)

            d = loader.load()
            docs.extend(d)
        except Exception as e:
            failed.append((url, str(e)))
        time.sleep(REQUEST_DELAY)
    return docs, failed

def build_faiss_vectorstore_for_blog(
    docs,
    persist_directory: str,
    chunk_size: int = 800,

    chunk_overlap: int = 100,
):
    """
    Build FAISS index for blogs using UNIFORM chunking.
    Matches old (stable) blog logic exactly.
    """

    os.makedirs(persist_directory, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    split_docs = []
    for d in docs:
        text = getattr(d, "page_content", None) or getattr(d, "content", "")
        metadata = getattr(d, "metadata", {}) or {}
        chunks = splitter.create_documents([text], metadatas=[metadata])
        split_docs.extend(chunks)

    embeddings = HuggingFaceEmbeddings(
        model_name=HUGGINGFACE_EMBEDDING_MODEL
    )

    vectordb = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings
    )

    vectordb.save_local(persist_directory)
    return vectordb




    
def load_blog_faiss() -> FAISS:
    return load_faiss_local(
        BLOG_FAISS_DIR,
        EMBEDDING_MODEL_NAME
    )


def blog_retriever_kdnuggets(query: str, state: WorkflowState) -> WorkflowState:
    try:
        blog_store = load_faiss_local(BLOG_FAISS_DIR, EMBEDDING_MODEL_NAME)
        retriever = make_faiss_retriever_tool_from_store(blog_store, k=6)
        return retriever(query, state)
    except Exception as e:
        state.status = "error"
        state.error = f"KDnuggets retriever error: {e}"
        return state


def extract_title_from_doc(doc) -> str:
    # 1️⃣ Metadata title
    meta_title = (doc.metadata or {}).get("title")
    if meta_title and len(meta_title.strip()) > 5:
        return meta_title.strip()

    text = (doc.page_content or "").strip()
    if not text:
        return "Untitled Article"

    # 2️⃣ First line heuristic (often h1)
    first_line = text.split("\n")[0].strip()
    if 5 < len(first_line) < 120:
        return first_line

    # 3️⃣ First sentence fallback
    sentence = text.split(".")[0].strip()
    if 5 < len(sentence) < 120:
        return sentence

    return "Untitled Article"



def load_faiss_local(persist_dir: str, model_name: str = EMBEDDING_MODEL_NAME) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    if not os.path.isdir(persist_dir):
        raise FileNotFoundError(f"FAISS persist dir not found: {persist_dir}")
    # allow dangerous deserialization to support some older stores
    vectordb = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    return vectordb



def make_faiss_retriever_tool_from_store(
    faiss_store_local: FAISS,
    k: int = 10,
    max_unique: int = 10,
    fetch_k: int = 50,
    use_mmr: bool = True,
) -> Callable[[str, WorkflowState], WorkflowState]:
    """
    Build a retriever_node(query, state) that:
      - calls the FAISS retriever (robust to method signatures)
      - deduplicates results by link/title keeping the best chunk per source
      - populates state.retrieved_docs (title, link, optional summary) up to max_unique
      - deliberately DOES NOT populate state.retrieved_text
    """

    # ------------------ Create retriever ------------------
    try:
        if use_mmr:
            retriever = faiss_store_local.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.7},
            )
        else:
            retriever = faiss_store_local.as_retriever(search_kwargs={"k": k})
    except Exception:
        retriever = faiss_store_local.as_retriever(search_kwargs={"k": k})

    # ------------------ Robust caller ------------------
    def call_get_relevant_documents(q: str):
    # ✅ Most reliable for current LangChain
        try:
            return retriever.invoke(q)
        except Exception:
            pass

    # ✅ Older versions
        try:
            return retriever.get_relevant_documents(q)
        except Exception:
            pass

    # ❌ If both fail, surface error
        raise RuntimeError("FAISS retriever returned no documents")


    # ------------------ Retriever node ------------------

    def retriever_node(query: str, state: WorkflowState) -> WorkflowState:
        try:
            docs = call_get_relevant_documents(query)

            print("[DEBUG] docs returned:", len(docs))

            state.retrieved_docs = []

            for d in docs[:max_unique]:
                meta = getattr(d, "metadata", {}) or {}

                title = extract_title_from_doc(d)

                link = (
        meta.get("link")
        or meta.get("source")
        or meta.get("url")
        or meta.get("source_url")
        or ""
    ).strip()

                summary = (meta.get("summary") or d.page_content or "")[:300]

                state.retrieved_docs.append({
        "title": title,
        "link": link,
        "summary": summary,
    })

            state.retrieved_text = ""
            state.status = "retrieved"

        except Exception as e:
            state.status = "error"
            state.error = str(e)

        return state

    # def retriever_node(query: str, state: WorkflowState) -> WorkflowState:
    #     query_lc = query.lower()

    #     def keyword_match(title: str, snippet: str) -> bool:
    #         text = f"{title} {snippet}".lower()
    #         return any(q in text for q in query_lc.split())

    #     state.status = f"retrieving_{getattr(state, 'selected_tool', 'rss')}"

    #     try:
    #         docs = call_get_relevant_documents(query)
    #         per_key_best = {}

    #         for idx, d in enumerate(docs or []):
    #             meta = getattr(d, "metadata", {}) or {}

    #             link = (
    #                 meta.get("link")
    #                 or meta.get("source")
    #                 or meta.get("url")
    #                 or meta.get("source_url")
    #                 or ""
    #             ).strip()

    #             title = (meta.get("title") or "").strip()
    #             snippet = (
    #                 meta.get("summary")
    #                 or meta.get("snippet")
    #                 or (getattr(d, "page_content", "") or "")[:400]
    #             )

    #             # ==================================================
    #             # ✅ CHANGE 1 — SOURCE-AWARE RELEVANCE LOGIC
    #             # ==================================================
            
    #             is_relevant = True
    #             # ==================================================

    #             if not is_relevant:
    #                 continue

    #             key = link or (title[:200] if title else f"chunk-{idx}")

    #             # Score extraction
    #             score = None
    #             for score_key in ("score", "similarity", "dist", "distance"):
    #                 if score_key in meta:
    #                     try:
    #                         score = float(meta[score_key])
    #                     except Exception:
    #                         score = None

    #             if score is None:
    #                 score = 1.0 / (1 + idx)

    #             per_key_best[key] = {
    #                 "score": score,
    #                 "title": title or snippet[:120],
    #                 "snippet": snippet,
    #                 "link": link,
    #             }

    #         # ------------------ Sort + dedupe ------------------
    #         sorted_items = sorted(
    #             per_key_best.values(),
    #             key=lambda x: x["score"],
    #             reverse=True,
    #         )

    #         state.retrieved_docs = []
    #         seen = set()

    #         for it in sorted_items:
    #             if len(state.retrieved_docs) >= max_unique:
    #                 break

    #             dedupe_key = (it["link"] or it["title"]).strip()
    #             if not dedupe_key or dedupe_key in seen:
    #                 continue

    #             seen.add(dedupe_key)
    #             state.retrieved_docs.append({
    #                 "title": it["title"],
    #                 "link": it["link"],
    #                 "summary": it["snippet"][:300] if it.get("snippet") else "",
    #             })

    #         state.retrieved_text = ""
    #         state.status = f"retrieved_{getattr(state, 'selected_tool', 'rss')}"

    #     except Exception as e:
    #         state.status = "error"
    #         state.error = (
    #             f"{getattr(state, 'selected_tool', 'retriever')} retriever error: "
    #             f"{repr(e)}\n{traceback.format_exc()}"
    #         )

    #     return state

    return retriever_node



import re

def normalize_title(title: str) -> str:
    """
    Remove markdown links and extra brackets from blog titles.
    Example:
    [[MCP Servers]](https://...)  → MCP Servers
    """
    if not title:
        return ""
    title = re.sub(r"\[+(.+?)\]+\([^)]+\)", r"\1", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def normalize_summary(summary: str) -> str:
    """
    Clean summary text for LLM consumption.
    """
    if not summary:
        return ""
    summary = re.sub(r"\s+", " ", summary)
    return summary.strip()



def make_react_agent_node_with_tool(
    retriever_call_fn: Callable[[str, WorkflowState], WorkflowState],
    llm_model: str = "llama-3.1-70b-versatile"
):
    """
    FINAL AGENT
    Output per article:
    title
    link
    key_points
    """

    import json
    import re
    from langchain_groq import ChatGroq

    def node(state: WorkflowState) -> WorkflowState:
        state.status = "generating"

        try:
            sources = state.retrieved_docs or []

            if not sources:
                state.generated_post = []
                state.status = "done"
                return state
            source_payload = []

            for s in sources:
                clean_title = clean_blog_title(s.get("title", ""))

                clean_summary = normalize_summary(s.get("summary", ""))

                source_payload.append({
        "title": clean_title,
        "link": s.get("link", ""),
        "summary": clean_summary
    })


            

            prompt = f"""
You are an expert tech news and blog summarizer.
STRICT RULES (MANDATORY):
- Title must be clean, professional mainly for blogs
- Do NOT include conversational phrases like “Here’s the response”
- Use ONLY the information explicitly present in SOURCES
- If information is missing, say so
- Do NOT infer, guess, or add future announcements
- Do NOT mention model versions unless present in SOURCES
- Do NOT summarize beyond retrieved content
- If sources are insufficient, return an empty list

TASK:
For EACH source below:
- Understand the topic
STRICT RULE:
- Use the title EXACTLY as provided for your undertsing
- rephrase, summarize, or rewrite titles just small title
- Do NOT prepend numbers or headings to titles

- Generate key bullet points covering the main aspects

STRICT RULES:
- Output must be a SINGLE valid JSON ARRAY
- ONE object per source
- Each object MUST contain:
  - title
  - link
  - key_points
- key_points must contain 3–5 concise bullets
- Do NOT merge sources
- Do NOT add extra text

JSON FORMAT:
[
  {{
    "title": "<title>",
    "link": "<url>",
    "key_points": ["point 1", "point 2", "point 3"]
    
  }}
]

SOURCES:
{json.dumps(source_payload, indent=2)}
"""

            llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=llm_model,
                temperature=0.2
            )

            response = llm.invoke(prompt)
            raw_text = response.content if hasattr(response, "content") else str(response)

            try:
                data = json.loads(raw_text)
            except Exception:
                match = re.search(r"\[.*\]", raw_text, re.DOTALL)
                if not match:
                    raise ValueError("Invalid JSON from LLM")
                data = json.loads(match.group(0))

            state.generated_post = [
                {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "key_points": [k for k in item.get("key_points", []) if k]
                }
                for item in data
            ]

            state.status = "done"
            return state

        except Exception as e:
            state.status = "error"
            state.error = f"agent error: {repr(e)}\n{traceback.format_exc()}"
            return state

    return node




class NodeSpec(BaseModel):
    id: str
    func: Callable[..., WorkflowState]
    run_with_query: bool = False

class StateGraph:
    def __init__(self, nodes: List[NodeSpec], edges: List[tuple], start_node: str):
        self.nodes_map = {n.id: n for n in nodes}
        self.edges = edges
        self.start_node = start_node

    def run(self, state: WorkflowState):
        current = self.start_node
        visited = set()
        while current:
            if current in visited:
                state.status = "error"
                state.error = f"cycle detected at node {current}"
                return state
            visited.add(current)
            node = self.nodes_map.get(current)
            if node is None:
                state.status = "error"
                state.error = f"node {current} not found"
                return state
            try:
                if node.run_with_query:
                    state = node.func(state.query, state)
                else:
                    state = node.func(state)
            except Exception as e:
                state.status = "error"
                state.error = f"node execution error: {repr(e)}\n{traceback.format_exc()}"
                return state

            outgoing = [t for (f, t) in self.edges if f == current]
            if not outgoing:
                break
            current = outgoing[0]
        return state


import re
def clean_blog_title(title: str) -> str:
    if not title:
        return ""

    t = title.strip()

    # Remove newlines
    t = t.replace("\n", " ")

    # Remove numbering like "1.", "2)", "1. 8"
    t = re.sub(r"^\s*\d+[\.\)\-]?\s*", "", t)

    # Cut off after separators
    for sep in [" – ", " - ", "|", " — "]:
        if sep in t:
            t = t.split(sep)[0]

    # Limit length (blog titles get noisy)
    t = t[:120]

    return " ".join(t.split())

def build_blog_index_node(state: WorkflowState) -> WorkflowState:
    state.status = "indexing_blog"
    try:
        print("🚀 Building KDnuggets blog index using HTML crawl")

        urls = crawl_tag(
            start_url=BLOG_START_URL,
            max_pages=BLOG_MAX_PAGES
        )

        if not urls:
            raise RuntimeError("No KDnuggets blog URLs found")

        docs, failed = load_articles_to_docs(
            urls,
            use_unstructured=True
        )

        if not docs:
            raise RuntimeError("No blog documents loaded")

        build_faiss_vectorstore_for_blog(
            docs,
            persist_directory=BLOG_FAISS_DIR
        )

        state.status = "indexed_blog"
        print("✅ KDnuggets blog FAISS built successfully")
        return state

    except Exception as e:
        state.status = "error"
        state.error = str(e)
        return state


BLOG_SOURCES = {
    "KDnuggets": {
        "faiss_dir": "./kdnuggets_faiss",
        "build_fn": build_blog_index_node,
    },
    "Analytics Vidhya": {
        "faiss_dir": "./faiss_blogs/analyticsvidhya",
        "build_fn": build_analytics_vidhya_index,
    },
    "Machine Learning Mastery": {
        "faiss_dir": "./faiss_blogs/ml_mastery",
        "build_fn": build_machine_learning_mastery_index,
    },
    
}



def run_full_workflow_example(
    query: str,
    selected_tool: str = "rss",
    selected_blog: Optional[str] = None,
    build_if_missing: bool = False,
):
    """
    FINAL CLEAN WORKFLOW

    - RSS  → searches ONLY rss FAISS
    - BLOG → searches ONLY blog FAISS (KDnuggets HTML crawl)
    - Builds index ONLY if missing + explicitly requested
    """

    global rss_faiss_store, blog_faiss_store

    # ---------------------------------------------------
    # STEP 1: RSS MODE
    # ---------------------------------------------------
    if selected_tool == "rss":
        print("[main] RSS mode → LOAD existing FAISS only")

        if rss_faiss_store is None:
            rss_faiss_store = load_faiss_local(
                RSS_FAISS_DIR,
                EMBEDDING_MODEL_NAME
            )

        retriever_node = make_faiss_retriever_tool_from_store(
            rss_faiss_store,
            k=6
        )

    # ---------------------------------------------------
    # STEP 2: BLOG MODE (KDnuggets)
    # ---------------------------------------------------
    elif selected_tool == "blog":
        if not selected_blog or selected_blog not in BLOG_SOURCES:
            raise ValueError("Invalid or missing blog selection")

        blog_cfg = BLOG_SOURCES[selected_blog]
        blog_dir = blog_cfg["faiss_dir"]
        build_fn = blog_cfg["build_fn"]

        print(f"[main] BLOG mode → {selected_blog}")

        try:
            blog_faiss_store = load_faiss_local(blog_dir, EMBEDDING_MODEL_NAME)
            print(f"[main] Loaded {selected_blog} FAISS")

        except Exception:
            if not build_if_missing:
                raise RuntimeError(
                    f"{selected_blog} FAISS missing. Run with build_if_missing=True"
            )

            print(f"[main] Building {selected_blog} FAISS")
            build_state = WorkflowState(
            request_id=f"build-{selected_blog}",
            query="",
            selected_tool="blog",
            selected_blog=selected_blog,
        )

            build_fn(build_state)

            if build_state.status == "error":
                raise RuntimeError(build_state.error)

            blog_faiss_store = load_faiss_local(blog_dir, EMBEDDING_MODEL_NAME)

        retriever_node = make_faiss_retriever_tool_from_store(
        blog_faiss_store,
        k=6
    )


    else:
        raise ValueError(f"Unknown selected_tool: {selected_tool}")

    # ---------------------------------------------------
    # STEP 3: Agent
    # ---------------------------------------------------
    agent_node = make_react_agent_node_with_tool(
        retriever_call_fn=retriever_node,
        llm_model="openai/gpt-oss-120b",
    )

    # ---------------------------------------------------
    # STEP 4: Graph (FAST PATH)
    # ---------------------------------------------------
    nodes = [
        NodeSpec(id="start", func=lambda s: s),
        NodeSpec(id="retriever", func=retriever_node, run_with_query=True),
        NodeSpec(id="agent", func=agent_node),
        NodeSpec(id="end", func=lambda s: s),
    ]

    edges = [
        ("start", "retriever"),
        ("retriever", "agent"),
        ("agent", "end"),
    ]

    graph = StateGraph(nodes=nodes, edges=edges, start_node="start")

    init_state = WorkflowState(
        request_id="req-001",
        query=query,
        selected_tool=selected_tool,
        selected_blog=selected_blog,
    )

    return graph.run(init_state)


if __name__ == "__main__":
    print("🔄 Refreshing AI RSS → CSV → FAISS")

    refresh_state = WorkflowState(
        request_id="rss-refresh",
        query=""
    )

    refresh_state = rebuild_rss_faiss_always(refresh_state)

    if refresh_state.status == "error":
        print("❌ RSS refresh failed")
        print(refresh_state.error)
        sys.exit(1)

    print("✅ RSS refresh completed")

    print("🚀 Running query on latest AI news")

    result = run_full_workflow_example(
        query="latest ai news",
        selected_tool="rss"
    )

    print("\n✅ FINAL RESULT STATUS:", result.status)

    if result.error:
        print("❌ ERROR:", result.error)
    else:
        print("\n📄 RETRIEVED DOCS:")
        for d in result.retrieved_docs:
            print("-", d["title"], "=>", d["link"])








#!/usr/bin/env python3

# import os
# import csv
# import re
# import sys
# import time
# import json
# import traceback
# import inspect
# import asyncio
# import feedparser
# import calendar
# from html import unescape
# from datetime import datetime, timezone, timedelta
# from typing import List, Dict, Any, Callable, Optional, Tuple

# from sentence_transformers import SentenceTransformer
# import numpy as np
# import faiss

# from langchain_community.document_loaders import WebBaseLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_classic.vectorstores import FAISS
# from langchain_classic.embeddings import HuggingFaceEmbeddings
# from langchain_classic.docstore.document import Document

# import requests
# from urllib.parse import urljoin, urlparse
# from bs4 import BeautifulSoup
# from tqdm import tqdm
# from langchain_classic.document_loaders import UnstructuredURLLoader
# from dotenv import load_dotenv
# load_dotenv()

# CSV_PATH = "ai_rss_results.csv"
# RSS_FAISS_DIR = "faiss_index_local"
# BLOG_FAISS_DIR = "./kdnuggets_faiss"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# BLOG_START_URL = "https://www.kdnuggets.com/tag/artificial-intelligence"
# BLOG_REQUEST_DELAY = 1.0
# BLOG_MAX_PAGES = 50

# CSV_META_MAP: Dict[str, Dict[str, str]] = {}
# rss_faiss_store: Optional[FAISS] = None
# blog_faiss_store: Optional[FAISS] = None
# from pydantic import BaseModel

# class WorkflowState(BaseModel):
#     request_id: str
#     query: str
#     selected_tool: str = "rss"   # "rss" or "blog" (default rss)
#     status: str = "initialized"
#     retrieved_docs: List[Dict[str, Any]] = []
#     retrieved_text: str = ""
#     generated_post: str = ""
#     sources: List[Dict[str, str]] = []
#     error: Optional[str] = None

# DAYS_WINDOW = 7
# MIN_PER_FEED = 5

# AI_FEEDS = [
#     "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml",
#     "https://venturebeat.com/category/ai/feed/",
#     "https://www.analyticsvidhya.com/blog/category/artificial-intelligence/feed/",
#     "https://news.google.com/rss/search?q=artificial+intelligence&hl=en-IN&gl=IN&ceid=IN:en",
#     "https://techcrunch.com/feed/",
#     "https://yourstory.com/feed",
#     "https://economictimes.indiatimes.com/tech/rssfeeds/13357270.cms",
#     "https://blog.langchain.dev/rss/",
#     "https://www.llamaindex.ai/blog/rss.xml",
#     "https://openai.com/blog/rss/",
#     "https://huggingface.co/blog/feed.xml",
#     "https://mistral.ai/news/feed.xml",
#     "https://www.anthropic.com/news/feed.xml",
#     "https://vercel.com/changelog/rss",
#     "https://weaviate.io/blog/rss.xml",
#     "https://www.pinecone.io/rss.xml",
#     "https://milvus.io/blog/index.xml",
#     "https://news.ycombinator.com/rss",
# ]

# def entry_published_dt(entry):
#     parsed = entry.get("published_parsed") or entry.get("updated_parsed")
#     if not parsed:
#         return None
#     try:
#         ts = int(calendar.timegm(parsed))
#         return datetime.fromtimestamp(ts, tz=timezone.utc)
#     except Exception:
#         return None

# def to_iso(dt: datetime):
#     if not dt:
#         return ""
#     return dt.astimezone(timezone.utc).isoformat()

# def clean_html_summary(raw_html: str) -> str:
#     if not raw_html:
#         return ""
#     s = unescape(raw_html)
#     s = re.sub(r"<[^>]+>", "", s)
#     s = re.sub(r"\s+", " ", s)
#     return s.strip()

# def collect_rss_and_write_csv(output_csv: str = CSV_PATH,
#                               days_window: int = DAYS_WINDOW,
#                               min_per_feed: int = MIN_PER_FEED) -> List[Dict[str, Any]]:
#     feeds = list(AI_FEEDS)
#     now = datetime.now(timezone.utc)
#     cutoff = now - timedelta(days=days_window)

#     all_articles = []
#     seen_links = set()

#     for feed_url in feeds:
#         parsed_feed = feedparser.parse(feed_url)
#         feed_title = parsed_feed.feed.get("title", feed_url)

#         items = []
#         for idx, entry in enumerate(parsed_feed.entries):
#             link = (entry.get("link") or entry.get("id") or "").strip()
#             if not link:
#                 continue
#             pub_dt = entry_published_dt(entry)
#             pub_ts = int(pub_dt.timestamp()) if pub_dt else 0
#             items.append({
#                 "entry": entry,
#                 "link": link,
#                 "published_dt": pub_dt,
#                 "published_ts": pub_ts,
#                 "index": idx
#             })

#         recent = [it for it in items if it["published_dt"] and it["published_dt"] >= cutoff]
#         recent.sort(key=lambda x: x["published_ts"], reverse=True)

#         if len(recent) < min_per_feed:
#             recent_links = {it["link"] for it in recent}
#             candidates = [it for it in items if it["link"] not in recent_links]
#             candidates.sort(key=lambda x: (x["published_ts"], -x["index"]), reverse=True)
#             needed = min_per_feed - len(recent)
#             for c in candidates:
#                 if needed <= 0:
#                     break
#                 recent.append(c)
#                 needed -= 1

#         selected = recent[:max(min_per_feed, len(recent))]

#         for sel in selected:
#             link = sel["link"]
#             if link in seen_links:
#                 continue
#             entry = sel["entry"]
#             pub_dt = sel["published_dt"]
#             pub_ts = sel["published_ts"] if sel["published_dt"] else 0

#             published_str = entry.get("published", "") or entry.get("updated", "") or ""

#             raw_summary = entry.get("summary", "") or entry.get("description", "") or ""
#             summary = clean_html_summary(raw_summary)

#             title = (entry.get("title") or "No title").strip()

#             all_articles.append({
#                 "feed": feed_title,
#                 "title": title,
#                 "link": link,
#                 "published_iso": to_iso(pub_dt),
#                 "published_ts": pub_ts,
#                 "published_raw": published_str,
#                 "summary": summary
#             })
#             seen_links.add(link)

#     all_articles.sort(key=lambda x: x["published_ts"], reverse=True)

#     fieldnames = ["feed", "title", "link", "published_iso", "published_ts", "published_raw", "summary"]
#     with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for row in all_articles:
#             writer.writerow(row)

#     return all_articles

# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# def remove_diagram_lines(text: str, min_words_keep: int = 4, punct_ratio_thresh: float = 0.4) -> str:
#     if not text:
#         return ""
#     lines = text.splitlines()
#     cleaned_lines = []
#     for ln in lines:
#         s = ln.strip()
#         if not s:
#             continue
#         low = s.lower()
#         if re.search(r'\bfig(?:ure)?\.?\s*\d+\b', low):
#             continue
#         if any(kw in low for kw in [
#             "figure", "fig.", "diagram", "flowchart", "flow chart", "chart:", "table:", "image:",
#             "illustration", "infographic", "caption:", "click to view", "download pdf", "view image",
#             "see figure", "see diagram", "open image", "related image", "share this article",
#             "embed", "powered by", "svg", "png", "jpg", "gif"
#         ]):
#             if len(s.split()) < 20:
#                 continue
#         if re.fullmatch(r'[\W_]{3,}', s):
#             continue
#         words = s.split()
#         if len(words) <= min_words_keep:
#             non_alnum = re.sub(r'\w', '', s)
#             punct_ratio = len(non_alnum) / max(1, len(s))
#             if punct_ratio > punct_ratio_thresh:
#                 continue
#             if re.search(r'^[\w\-\s\|,]{0,60}$', s) and ('|' in s or ',' in s or s.isupper()):
#                 continue
#             cap_frac = sum(1 for w in words if w and w[0].isupper()) / max(1, len(words))
#             if cap_frac > 0.6 and len(words) <= 6:
#                 continue
#         if re.search(r'\b(share|twitter|facebook|linkedin|email|subscribe|subscribe to)\b', low) and len(words) <= 8:
#             continue
#         cleaned_lines.append(s)
#     return "\n".join(cleaned_lines)

# def clean_text(text: str) -> str:
#     if not text:
#         return ""
#     text = text.replace('\r\n', '\n').replace('\r', '\n')
#     noisy_block_patterns = [
#         r'(?is)table of contents.*',
#         r'(?is)you might also like[:]?.*',
#         r'(?is)related posts[:]?.*',
#         r'(?is)more from.*',
#     ]
#     for pat in noisy_block_patterns:
#         text = re.sub(pat, '', text)
#     text = remove_diagram_lines(text)
#     s = re.sub(r'\s+', ' ', text).strip()
#     s = re.sub(r'^[^\w]+', '', s)
#     s = re.sub(r'[^\w]+$', '', s)
#     return s


# def read_csv_metadata(csv_path: str) -> Tuple[List[str], Dict[str, Dict]]:
#     links = []
#     meta_map = {}
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f"{csv_path} not found.")
#     with open(csv_path, newline="", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             link = (row.get("link") or "").strip()
#             title = (row.get("title") or "").strip()
#             rss_summary = (row.get("summary") or "").strip()
#             if not link:
#                 continue
#             links.append(link)
#             meta_map[link] = {"title": title, "rss_summary": rss_summary}
#     return links, meta_map

# def load_and_clean_pages(links: List[str], meta_map: Dict[str, Dict]) -> Tuple[List[Document], List[Tuple[str,str]]]:
#     docs = []
#     failed = []
#     for url in links:
#         try:
#             loader = WebBaseLoader(url)
#             loaded = loader.load()
#             if not loaded:
#                 failed.append((url, "empty"))
#                 continue
#             doc = loaded[0]
#             clean_content = clean_text(doc.page_content) if hasattr(doc, "page_content") else clean_text(str(doc))
#             doc.page_content = clean_content
#             csv_meta = meta_map.get(url, {})
#             title = csv_meta.get("title") or doc.metadata.get("title") or ""
#             rss_summary = csv_meta.get("rss_summary") or ""
#             if rss_summary and len(rss_summary.strip()) > 20:
#                 final_summary = clean_text(rss_summary)
#             else:
#                 snippet = clean_content[:600]
#                 final_summary = snippet.strip()
#             doc.metadata["title"] = title.strip()
#             doc.metadata["link"] = url
#             doc.metadata["summary"] = final_summary.strip()
#             docs.append(doc)
#         except Exception as e:
#             failed.append((url, str(e)))
#     return docs, failed

# def build_faiss_from_docs_and_save(documents: List[Document], index_dir: str, model_name: str) -> FAISS:
#     split_docs = text_splitter.split_documents(documents)
#     if not split_docs:
#         raise RuntimeError("No split documents to index.")
#     for d in split_docs:
#         d.metadata["title"] = (d.metadata.get("title") or "").strip()
#         d.metadata["link"] = (d.metadata.get("link") or "").strip()
#         if not d.metadata.get("summary"):
#             d.metadata["summary"] = (clean_text(d.page_content)[:300]).strip()
#     embeddings = HuggingFaceEmbeddings(model_name=model_name)
#     store = FAISS.from_documents(split_docs, embeddings)
#     os.makedirs(index_dir, exist_ok=True)
#     store.save_local(index_dir)
#     return store

# def build_rss_index_node(state: WorkflowState) -> WorkflowState:
#     state.status = "indexing_rss"
#     try:
#         links, meta_map = read_csv_metadata(CSV_PATH)
#         global CSV_META_MAP
#         CSV_META_MAP = {}
#         CSV_META_MAP.update(meta_map)
#         docs, failed = load_and_clean_pages(links, meta_map)
#         if not docs:
#             raise RuntimeError("No docs loaded for indexing.")
#         global rss_faiss_store
#         rss_faiss_store = build_faiss_from_docs_and_save(docs, RSS_FAISS_DIR, EMBEDDING_MODEL_NAME)
#         state.status = "indexed_rss"
#     except Exception as e:
#         state.status = "error"
#         state.error = f"build_rss_index_node failed: {repr(e)}\n{traceback.format_exc()}"
#     return state

# START_URL = BLOG_START_URL
# REQUEST_DELAY = BLOG_REQUEST_DELAY
# MAX_PAGES = BLOG_MAX_PAGES
# HUGGINGFACE_EMBEDDING_MODEL = EMBEDDING_MODEL_NAME
# FAISS_PERSIST_DIR = BLOG_FAISS_DIR

# def fetch_html(url):
#     r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
#     r.raise_for_status()
#     return r.text

# def extract_article_links_from_tag_page(html, base_url):
#     soup = BeautifulSoup(html, "html.parser")
#     anchors = soup.find_all("a", href=True)
#     links = set()
#     for a in anchors:
#         href = a["href"].strip()
#         href = urljoin(base_url, href)
#         parsed = urlparse(href)
#         if "kdnuggets.com" not in parsed.netloc:
#             continue
#         if re.search(r"/\d{4}/\d{2}/|/20\d{2}/|-[a-z0-9\-]+$", href):
#             links.add(href.split("?")[0].rstrip("/"))
#         else:
#             if "/blog" in href or (len(parsed.path.split("/")) > 2 and "-" in parsed.path):
#                 links.add(href.split("?")[0].rstrip("/"))
#     return sorted(links)

# def find_pagination_next(html, base_url):
#     soup = BeautifulSoup(html, "html.parser")
#     for text in ["Next", "Older", "Older Posts", "Next »", "→", "older posts"]:
#         el = soup.find("a", string=lambda s: s and text.lower() in s.lower())
#         if el and el.get("href"):
#             return urljoin(base_url, el["href"])
#     el = soup.find("link", rel="next")
#     if el and el.get("href"):
#         return urljoin(base_url, el["href"])
#     return None

# def crawl_tag(start_url, max_pages=MAX_PAGES):
#     to_visit = [start_url]
#     visited = set()
#     article_urls = set()
#     for _ in range(max_pages):
#         if not to_visit:
#             break
#         url = to_visit.pop(0)
#         if url in visited:
#             continue
#         try:
#             html = fetch_html(url)
#         except Exception as e:
#             print(f"[warn] failed to fetch {url}: {e}")
#             visited.add(url)
#             continue
#         visited.add(url)
#         found = extract_article_links_from_tag_page(html, url)
#         article_urls.update(found)
#         next_page = find_pagination_next(html, url)
#         if next_page and next_page not in visited:
#             to_visit.append(next_page)
#         time.sleep(REQUEST_DELAY)
#     return sorted(article_urls)

# def load_articles_to_docs(urls, use_unstructured=True):
#     docs = []
#     failed = []
#     if use_unstructured:
#         try:
#             loader = UnstructuredURLLoader(urls=urls)
#             docs = loader.load()
#             return docs, failed
#         except Exception as e:
#             print(f"[info] Unstructured loader failed, falling back to WebBaseLoader: {e}")
#     for url in tqdm(urls, desc="Loading URLs"):
#         try:
#             loader = WebBaseLoader(url)
#             d = loader.load()
#             docs.extend(d)
#         except Exception as e:
#             failed.append((url, str(e)))
#         time.sleep(REQUEST_DELAY)
#     return docs, failed

# def build_faiss_vectorstore_for_blog(docs, persist_directory=FAISS_PERSIST_DIR,
#                                      chunk_size=800, chunk_overlap=100):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     split_docs = []
#     for d in docs:
#         text = getattr(d, "page_content", None) or getattr(d, "content", "")
#         metadata = getattr(d, "metadata", {}) or {}
#         chunks = splitter.create_documents([text], metadatas=[metadata])
#         split_docs.extend(chunks)
#     embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL)
#     vectordb = FAISS.from_documents(documents=split_docs, embedding=embeddings)
#     os.makedirs(persist_directory, exist_ok=True)
#     vectordb.save_local(persist_directory)
#     return vectordb

# def build_blog_index_node(state: WorkflowState) -> WorkflowState:
#     state.status = "indexing_blog"
#     try:
#         urls = crawl_tag(START_URL, max_pages=MAX_PAGES)
#         docs, failed = load_articles_to_docs(urls, use_unstructured=True)
#         if not docs:
#             raise RuntimeError("No blog docs loaded for indexing.")
#         global blog_faiss_store
#         blog_faiss_store = build_faiss_vectorstore_for_blog(docs, persist_directory=BLOG_FAISS_DIR)
#         state.status = "indexed_blog"
#     except Exception as e:
#         state.status = "error"
#         state.error = f"build_blog_index_node failed: {repr(e)}\n{traceback.format_exc()}"
#     return state

# def load_faiss_local(persist_dir: str, model_name: str = EMBEDDING_MODEL_NAME) -> FAISS:
#     embeddings = HuggingFaceEmbeddings(model_name=model_name)
#     if not os.path.isdir(persist_dir):
#         raise FileNotFoundError(f"FAISS persist dir not found: {persist_dir}")
#     # allow dangerous deserialization to support some older stores
#     vectordb = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
#     return vectordb



# def make_faiss_retriever_tool_from_store(faiss_store_local: FAISS, k: int = 10,
#                                         max_unique: int = 10, fetch_k: int = 50,
#                                         use_mmr: bool = True) -> Callable[[str, WorkflowState], WorkflowState]:
#     """
#     Build a retriever_node(query, state) that:
#       - calls the FAISS retriever (robust to method signatures)
#       - deduplicates results by link/title keeping the best chunk per source
#       - populates state.retrieved_docs (title, link, optional summary) up to max_unique
#       - deliberately DOES NOT populate state.retrieved_text (so agent won't receive snippets)
#     """
#     try:
#         if use_mmr:
#             retriever = faiss_store_local.as_retriever(
#                 search_type="mmr",
#                 search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.7},
#             )
#         else:
#             retriever = faiss_store_local.as_retriever(search_kwargs={"k": k})
#     except Exception:
#         retriever = faiss_store_local.as_retriever(search_kwargs={"k": k})

#     def call_get_relevant_documents(q: str):
#         last_exc = None
#         # 1) common name
#         try:
#             func = getattr(retriever, "get_relevant_documents", None)
#             if func is not None:
#                 if inspect.iscoroutinefunction(func):
#                     return asyncio.run(func(q))
#                 return func(q)
#         except Exception as e:
#             last_exc = e

#         # 2) try with run_manager kw
#         try:
#             func = getattr(retriever, "get_relevant_documents", None)
#             if func is not None:
#                 if inspect.iscoroutinefunction(func):
#                     return asyncio.run(func(q, run_manager=None))
#                 return func(q, run_manager=None)
#         except Exception as e:
#             last_exc = e

#         # 3) try get_relevant_texts -> wrap as docs
#         try:
#             func2 = getattr(retriever, "get_relevant_texts", None)
#             if func2 is not None:
#                 if inspect.iscoroutinefunction(func2):
#                     texts = asyncio.run(func2(q))
#                 else:
#                     texts = func2(q)
#                 docs = []
#                 for t in texts:
#                     docs.append(type("D", (), {"page_content": t, "metadata": {}})())
#                 return docs
#         except Exception as e:
#             last_exc = e

#         # 4) private fallback
#         try:
#             private = getattr(retriever, "_get_relevant_documents", None)
#             if private is not None:
#                 if inspect.iscoroutinefunction(private):
#                     return asyncio.run(private(q, run_manager=None))
#                 try:
#                     return private(q, run_manager=None)
#                 except TypeError:
#                     return private(q)
#         except Exception as e:
#             last_exc = e

#         raise last_exc or RuntimeError("Unable to call retriever methods with any known signature.")

#     def retriever_node(query: str, state: WorkflowState) -> WorkflowState:
#         state.status = f"retrieving_{getattr(state, 'selected_tool', 'rss')}"
#         try:
#             docs = call_get_relevant_documents(query)

#             per_key_best = {}
#             for idx, d in enumerate(docs or []):
#                 meta = getattr(d, "metadata", {}) or {}
#                 link = (meta.get("link") or meta.get("source") or meta.get("url") or meta.get("source_url") or "").strip()
#                 title = (meta.get("title") or "").strip()
#                 # Keep a short snippet only for internal ranking (not forwarded to agent)
#                 snippet = (meta.get("summary") or meta.get("snippet") or "") or (getattr(d, "page_content", "") or "")[:400]

#                 key = link or (title[:200] if title else f"chunk-{idx}")

#                 # Score extraction
#                 score = None
#                 for score_key in ("score", "similarity", "dist", "distance"):
#                     if score_key in meta:
#                         try:
#                             score = float(meta[score_key])
#                         except Exception:
#                             score = None
#                 if score is None:
#                     score = 1.0 / (1 + idx)

#                 existing = per_key_best.get(key)
#                 if existing is None or (score is not None and score > existing["score"]):
#                     per_key_best[key] = {"score": score, "title": title or snippet[:120], "snippet": snippet, "link": link}

#             sorted_items = sorted(per_key_best.values(), key=lambda x: (x["score"] if x["score"] is not None else 0), reverse=True)

#             state.retrieved_docs = []
#             seen = set()
#             for it in sorted_items:
#                 if len(state.retrieved_docs) >= max_unique:
#                     break
#                 dedupe_key = (it["link"] or it["title"]).strip()
#                 if not dedupe_key or dedupe_key in seen:
#                     continue
#                 seen.add(dedupe_key)
#                 # we intentionally keep only title + link + (short)summary meta for UI; agent will not get snippets
#                 state.retrieved_docs.append({
#                     "title": it["title"],
#                     "link": it["link"],
#                     # keep summary for your UI if you want, but we will not include it in LLM prompt
#                     "summary": (it["snippet"][:300] if it.get("snippet") else "")
#                 })

#             # Important: do NOT expose retrieved snippets to LLM — leave retrieved_text empty
#             state.retrieved_text = ""

#             state.status = f"retrieved_{getattr(state, 'selected_tool', 'rss')}"
#         except Exception as e:
#             state.status = "error"
#             state.error = f"{getattr(state, 'selected_tool', 'retriever')} retriever error: {repr(e)}\n{traceback.format_exc()}"
#         return state

#     return retriever_node

# def make_react_agent_node_with_tool(
#     retriever_call_fn: Callable[[str, WorkflowState], WorkflowState],
#     llm_model: str = "llama-3.1-70b-versatile"
# ):
#     """
#     FINAL AGENT
#     Output per article:
#     title
#     link
#     key_points
#     """

#     import json
#     import re
#     from langchain_groq import ChatGroq

#     def node(state: WorkflowState) -> WorkflowState:
#         state.status = "generating"

#         try:
#             sources = state.retrieved_docs or []

#             if not sources:
#                 state.generated_post = []
#                 state.status = "done"
#                 return state
#             source_payload = [
#                 {
#                     "title": s.get("title", ""),
#                     "link": s.get("link", ""),
#                     "summary": s.get("summary", "")
#                 }
#                 for s in sources
# ]

            

#             prompt = f"""
# You are an expert tech news and blog summarizer.

# TASK:
# For EACH source below:
# - Understand the topic
# - Generate key bullet points covering the main aspects

# STRICT RULES:
# - Output must be a SINGLE valid JSON ARRAY
# - ONE object per source
# - Each object MUST contain:
#   - title
#   - link
#   - key_points
# - key_points must contain 3–5 concise bullets
# - Do NOT merge sources
# - Do NOT add extra text

# JSON FORMAT:
# [
#   {{
#     "title": "<title>",
#     "link": "<url>",
#     "key_points": ["point 1", "point 2", "point 3"]
#   }}
# ]

# SOURCES:
# {json.dumps(source_payload, indent=2)}
# """

#             llm = ChatGroq(
#                 api_key=os.getenv("GROQ_API_KEY"),
#                 model=llm_model,
#                 temperature=0.2
#             )

#             response = llm.invoke(prompt)
#             raw_text = response.content if hasattr(response, "content") else str(response)

#             try:
#                 data = json.loads(raw_text)
#             except Exception:
#                 match = re.search(r"\[.*\]", raw_text, re.DOTALL)
#                 if not match:
#                     raise ValueError("Invalid JSON from LLM")
#                 data = json.loads(match.group(0))

#             state.generated_post = [
#                 {
#                     "title": item.get("title", ""),
#                     "link": item.get("link", ""),
#                     "key_points": [k for k in item.get("key_points", []) if k]
#                 }
#                 for item in data
#             ]

#             state.status = "done"
#             return state

#         except Exception as e:
#             state.status = "error"
#             state.error = f"agent error: {repr(e)}\n{traceback.format_exc()}"
#             return state

#     return node


# # def make_react_agent_node_with_tool(
# #     retriever_call_fn: Callable[[str, WorkflowState], WorkflowState],
# #     llm_model: str = "llama-3.1-70b-versatile"
# # ):
# #     """
# #     FINAL AGENT
# #     Output per article:
# #     title
# #     link
# #     key_points
# #     """

# #     import json
# #     import re
# #     from langchain_groq import ChatGroq

# #     def node(state: WorkflowState) -> WorkflowState:
# #         state.status = "generating"

# #         try:
# #             sources = state.retrieved_docs or []

# #             if not sources:
# #                 state.generated_post = []
# #                 state.status = "done"
# #                 return state

# #             source_payload = [
# #                 {
# #                     "title": s.get("title", ""),
# #                     "link": s.get("link", "")
# #                 }
# #                 for s in sources
# #             ]

# #             prompt = f"""
# # You are an expert tech news and blog summarizer.

# # TASK:
# # For EACH source below:
# # - Understand the topic
# # - Generate key bullet points covering the main aspects

# # STRICT RULES (MANDATORY):
# # - Output must be a SINGLE valid JSON ARRAY
# # - ONE object per source
# # - Each object MUST contain:
# #   - title
# #   - link
# #   - key_points
# # - key_points must contain 3–5 concise bullets
# # - Bullets must explain:
# #   - What happened
# #   - Why it matters
# #   - Impact / implications
# # - Do NOT merge sources
# # - Do NOT add extra text
# # - Use ONLY the provided sources

# # JSON FORMAT (EXACT):
# # [
# #   {{
# #     "title": "<title>",
# #     "link": "<url>",
# #     "key_points": [
# #       "<point 1>",
# #       "<point 2>",
# #       "<point 3>"
# #     ]
# #   }}
# # ]

# # SOURCES:
# # {json.dumps(source_payload, indent=2)}
# # """

# #             # llm = ChatGroq(
# #             #     model=llm_model,
# #             #     temperature=0.2
# #             # )
# #             llm = ChatGroq(
# #             api_key=os.getenv("GROQ_API_KEY"),
# #             model=llm_model,
# #             temperature=0.2
# # )


# #             response = llm.invoke(prompt)
# #             raw_text = response.content if hasattr(response, "content") else str(response)

# #             try:
# #                 data = json.loads(raw_text)
# #             except Exception:
# #                 match = re.search(r"\[.*\]", raw_text, re.DOTALL)
# #                 if not match:
# #                     raise ValueError("LLM output is not valid JSON")
# #                 data = json.loads(match.group(0))

# #             final_output = []
# #             for item in data:
# #                 final_output.append({
# #                     "title": str(item.get("title", "")).strip(),
# #                     "link": str(item.get("link", "")).strip(),
# #                     "key_points": [
# #                         str(k).strip()
# #                         for k in item.get("key_points", [])
# #                         if str(k).strip()
# #                     ]
# #                 })

# #             state.generated_post = final_output
# #             state.status = "done"
# #             return state

# #         except Exception as e:
# #             state.status = "error"
# #             state.error = f"agent error: {repr(e)}\n{traceback.format_exc()}"
# #             return state

# #     return node






# def make_combined_retriever_node(rss_retriever_fn: Callable[[str, WorkflowState], WorkflowState],
#                                  blog_retriever_fn: Callable[[str, WorkflowState], WorkflowState]) -> Callable[[str, WorkflowState], WorkflowState]:
#     def node(query: str, state: WorkflowState) -> WorkflowState:
#         # choose tool
#         tool = (state.selected_tool or "rss").lower()
#         if tool not in ("rss", "blog"):
#             tool = "rss"
#         state.status = f"retrieving_using_{tool}"
#         try:
#             if tool == "rss":
#                 return rss_retriever_fn(query, state)
#             else:
#                 return blog_retriever_fn(query, state)
#         except Exception as e:
#             state.status = "error"
#             state.error = f"combined retriever error: {repr(e)}\n{traceback.format_exc()}"
#             return state
#     return node

# class NodeSpec(BaseModel):
#     id: str
#     func: Callable[..., WorkflowState]
#     run_with_query: bool = False

# class StateGraph:
#     def __init__(self, nodes: List[NodeSpec], edges: List[tuple], start_node: str):
#         self.nodes_map = {n.id: n for n in nodes}
#         self.edges = edges
#         self.start_node = start_node

#     def run(self, state: WorkflowState):
#         current = self.start_node
#         visited = set()
#         while current:
#             if current in visited:
#                 state.status = "error"
#                 state.error = f"cycle detected at node {current}"
#                 return state
#             visited.add(current)
#             node = self.nodes_map.get(current)
#             if node is None:
#                 state.status = "error"
#                 state.error = f"node {current} not found"
#                 return state
#             try:
#                 if node.run_with_query:
#                     state = node.func(state.query, state)
#                 else:
#                     state = node.func(state)
#             except Exception as e:
#                 state.status = "error"
#                 state.error = f"node execution error: {repr(e)}\n{traceback.format_exc()}"
#                 return state

#             outgoing = [t for (f, t) in self.edges if f == current]
#             if not outgoing:
#                 break
#             current = outgoing[0]
#         return state


# def run_full_workflow_example(query: str = "latest generative ai release",
#                               selected_tool: str = "rss",
#                               build_if_missing: bool = False):
#     """
#     Run workflow for a single query.

#     - If FAISS indexes exist on disk (or are already loaded to rss_faiss_store / blog_faiss_store),
#       this function will skip rebuilding and run only the retriever -> agent path (fast).
#     - If indexes are missing and build_if_missing=True, it will run the build nodes (slow).
#     - If indexes are missing and build_if_missing=False, returns a state with an error message.
#     """
#     global rss_faiss_store, blog_faiss_store

#     # 1) Ensure RSS CSV exists (but don't necessarily build index)
#     if not os.path.exists(CSV_PATH):
#         try:
#             print("[main] RSS CSV not found -> collecting RSS feeds...")
#             articles = collect_rss_and_write_csv(output_csv=CSV_PATH)
#             CSV_META_MAP.clear()
#             for a in articles:
#                 CSV_META_MAP[a["link"]] = {"title": a["title"], "summary": a["summary"]}
#             print(f"[main] Collected {len(articles)} RSS items.")
#         except Exception as e:
#             print("[main] RSS collection failed (non-fatal):", e)

#     # 2) Try to load persisted FAISS indexes (do not build unless requested)
#     if rss_faiss_store is None:
#         try:
#             rss_faiss_store = load_faiss_local(RSS_FAISS_DIR, EMBEDDING_MODEL_NAME)
#             print("[main] Loaded RSS FAISS from disk:", RSS_FAISS_DIR)
#         except Exception as e:
#             print("[main] RSS FAISS load failed or not present:", e)
#             rss_faiss_store = None

#     if blog_faiss_store is None:
#         try:
#             blog_faiss_store = load_faiss_local(BLOG_FAISS_DIR, EMBEDDING_MODEL_NAME)
#             print("[main] Loaded BLOG FAISS from disk:", BLOG_FAISS_DIR)
#         except Exception as e:
#             print("[main] BLOG FAISS load failed or not present:", e)
#             blog_faiss_store = None

#     # 3) If any index missing and build_if_missing True, run the build nodes first.
#     if (rss_faiss_store is None or blog_faiss_store is None) and build_if_missing:
#         # Create a small graph that runs only missing builds in sequence, then continues
#         build_nodes = []
#         edges = [("start", "maybe_rss_collect")]
#         # maybe_rss_collect is a no-op placeholder in your code that ensures CSV exists
#         if rss_faiss_store is None:
#             build_nodes.append(("build_rss_index", build_rss_index_node))
#             edges.append(("maybe_rss_collect", "build_rss_index"))
#             edges.append(("build_rss_index", "build_blog_index"))  # ensure ordering
#         if blog_faiss_store is None:
#             build_nodes.append(("build_blog_index", build_blog_index_node))
#             if ("maybe_rss_collect", "build_rss_index") not in edges:
#                 edges.append(("maybe_rss_collect", "build_blog_index"))

#         # After builds, continue to retriever -> agent -> end
#         edges.append((edges[-1][1] if edges else "maybe_rss_collect", "retriever"))
#         edges.append(("retriever", "agent"))
#         edges.append(("agent", "end"))

#         # Construct NodeSpec list dynamically
#         nodes = [
#             NodeSpec(id="start", func=lambda s: s),
#             NodeSpec(id="maybe_rss_collect", func=lambda st: st if os.path.exists(CSV_PATH) else (collect_rss_and_write_csv(CSV_PATH) or st)),
#         ]
#         for nid, fn in build_nodes:
#             nodes.append(NodeSpec(id=nid, func=fn, run_with_query=False))

#         # Placeholder retriever that uses combined retriever
#         nodes.append(NodeSpec(id="retriever", func=lambda q, s: make_combined_retriever_node(
#             make_faiss_retriever_tool_from_store(rss_faiss_store, k=10, max_unique=10)if rss_faiss_store else (lambda q, s: s),
#             make_faiss_retriever_tool_from_store(blog_faiss_store, k=10, max_unique=10) if blog_faiss_store else (lambda q, s: s)
#         )(q, s), run_with_query=True))
#         agent_node_func = make_react_agent_node_with_tool(lambda q, s: make_combined_retriever_node(
#             make_faiss_retriever_tool_from_store(rss_faiss_store, k=10, max_unique=10) if rss_faiss_store else (lambda q, s: s),
#             make_faiss_retriever_tool_from_store(blog_faiss_store, k=10, max_unique=10) if blog_faiss_store else (lambda q, s: s)
#         )(q, s))
#         nodes.append(NodeSpec(id="agent", func=agent_node_func, run_with_query=False))
#         nodes.append(NodeSpec(id="end", func=lambda s: s, run_with_query=False))

#         graph = StateGraph(nodes=[n for n in nodes], edges=edges, start_node="start")

#         init_state = WorkflowState(request_id="build-run-001", query="")
#         res = graph.run(init_state)

#         # If builds succeeded, try to reload stores into memory for subsequent quick runs
#         if rss_faiss_store is None:
#             try:
#                 rss_faiss_store = load_faiss_local(RSS_FAISS_DIR, EMBEDDING_MODEL_NAME)
#             except Exception:
#                 pass
#         if blog_faiss_store is None:
#             try:
#                 blog_faiss_store = load_faiss_local(BLOG_FAISS_DIR, EMBEDDING_MODEL_NAME)
#             except Exception:
#                 pass

#         # if build failed, return that state early
#         if res.status == "error":
#             return res

#     # 4) At this point prefer to run only retriever -> agent (fast path).
#     # Build retriever functions from loaded stores (must exist)
#     if rss_faiss_store is None and blog_faiss_store is None:
#         # nothing to query — return an informative state
#         st = WorkflowState(request_id="req-none", query=query)
#         st.status = "error"
#         st.error = "No FAISS indexes loaded for RSS or BLOG. Set build_if_missing=True to build indexes, or ensure persisted indexes exist on disk."
#         return st

#     rss_retriever_node = make_faiss_retriever_tool_from_store(rss_faiss_store, k=6) if rss_faiss_store is not None else (lambda q, s: s)
#     blog_retriever_node = make_faiss_retriever_tool_from_store(blog_faiss_store, k=6) if blog_faiss_store is not None else (lambda q, s: s)
#     combined_retriever = make_combined_retriever_node(rss_retriever_node, blog_retriever_node)

#     # create the agent node once using the combined retriever
#     agent_node_func = make_react_agent_node_with_tool(lambda q, s: combined_retriever(q, s), llm_model="openai/gpt-oss-120b")

#     # minimal fast graph: start -> retriever -> agent -> end
#     nodes = [
#         NodeSpec(id="start", func=lambda state: state, run_with_query=False),
#         NodeSpec(id="retriever", func=combined_retriever, run_with_query=True),
#         NodeSpec(id="agent", func=agent_node_func, run_with_query=False),
#         NodeSpec(id="end", func=lambda state: state, run_with_query=False),
#     ]
#     edges = [
#         ("start", "retriever"),
#         ("retriever", "agent"),
#         ("agent", "end"),
#     ]

#     graph = StateGraph(nodes=nodes, edges=edges, start_node="start")
#     init_state = WorkflowState(request_id="req-001", query=query or "", selected_tool=(selected_tool or "rss"))
#     result_state = graph.run(init_state)
#     return result_state

# # ---------------------------------------------------------
# # If run as script: demo
# # ---------------------------------------------------------
# if __name__ == "__main__":
#     print("Running multi-tool Langraph pipeline (RSS + Blog).")
#     try:
#         res = run_full_workflow_example(query="What's new in RAG and LLMs?", selected_tool="rss")
#         print("Status:", res.status)
#         if res.error:
#             print("Error:", res.error)
#         else:
#             print("\nGenerated Post Preview:\n")
#             # print((res.generated_post or "")[:2000])
#             from pprint import pformat

#             if isinstance(res.generated_post, dict):
#                 print(pformat(res.generated_post)[:2000])
#             else:
#                 print(str(res.generated_post or "")[:2000])

#             print("\nTop retrieved docs:")
#             for i, s in enumerate(res.retrieved_docs[:6], start=1):
#                 print(f"[{i}] {s.get('title')}")
#                 print(s.get('summary')[:400])
#                 print("Link:", s.get('link'))
#     except Exception as e:
#         print("Fatal error:", e)
#         traceback.print_exc()
