

import os
import streamlit as st
from typing import List, Dict, Any
import datetime
import traceback
import textwrap
from final_langraph import run_full_workflow_example
from research_paper import get_research_papers, DEFAULT_TOPICS
from hackathons import get_events as get_hackathons
from conferences import get_events as get_conferences
from workshops import get_events as get_workshops
from typing import List

DEFAULT_BLOG = "KDnuggets"
BLOG_FAISS_DIR = "./kdnuggets_faiss"


BLOG_SOURCES = {
    "KDnuggets": {
        "faiss_dir": "./kdnuggets_faiss",
    },
    "Analytics Vidhya": {
        "faiss_dir": "./faiss_blogs/analyticsvidhya",
    },

    "Machine Learning Mastery": {
        "faiss_dir": "./faiss_blogs/ml_mastery",
    }

    }



st.set_page_config(page_title="Multi-tool Chatbot + Research + Workshops", layout="wide")
if "all_chat_sessions" not in st.session_state:
    st.session_state["all_chat_sessions"] = {}
if "current_chat_session_id" not in st.session_state:
    st.session_state["current_chat_session_id"] = "default_session"
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "selected_tool_temp" not in st.session_state:
    st.session_state["selected_tool_temp"] = "rss"
if "last_tool" not in st.session_state:
    st.session_state["last_tool"] = "rss"
if "research_choices" not in st.session_state:
    st.session_state["research_choices"] = DEFAULT_TOPICS if DEFAULT_TOPICS else []
if "research_selected" not in st.session_state:
    st.session_state["research_selected"] = st.session_state["research_choices"][0] if st.session_state["research_choices"] else ""
if "hackathon_results" not in st.session_state:
    st.session_state["hackathon_results"] = []
if "conference_results" not in st.session_state:
    st.session_state["conference_results"] = []
if "workshop_results" not in st.session_state:
    st.session_state["workshop_results"] = []
if "show_tool_selector" not in st.session_state:
    st.session_state["show_tool_selector"] = False

# Event filters
if "hackathon_location_filter" not in st.session_state:
    st.session_state["hackathon_location_filter"] = "All"
if "hackathon_type_filter" not in st.session_state:
    st.session_state["hackathon_type_filter"] = "All"
if "conference_location_filter" not in st.session_state:
    st.session_state["conference_location_filter"] = "All"
if "conference_type_filter" not in st.session_state:
    st.session_state["conference_type_filter"] = "All"
if "workshop_location_filter" not in st.session_state:
    st.session_state["workshop_location_filter"] = "All"
if "workshop_type_filter" not in st.session_state:
    st.session_state["workshop_type_filter"] = "All"
if "clear_next" not in st.session_state:
    st.session_state["clear_next"] = False

if "chat_user_input" not in st.session_state:
    st.session_state["chat_user_input"] = ""
if "last_assistant_render" not in st.session_state:
    st.session_state["last_assistant_render"] = ""
if "last_state" not in st.session_state:
    st.session_state["last_state"] = None

if "chat_input_tab" not in st.session_state:
    st.session_state["chat_input_tab"] = st.session_state.get("chat_user_input", "")
st.title("⭐ Internal AI Knowledge Portal")

tab_chat, tab_research, tab_hackathons, tab_conferences, tab_workshops = st.tabs(["Chatbot", "Research Papers", "Hackathons", "Conferences", "Workshops"])

def render_structured_post(gen: Dict[str, Any]) -> str:
    lines = []
    hook = gen.get("hook", "") if isinstance(gen, dict) else ""
    paragraphs = gen.get("paragraphs", []) if isinstance(gen, dict) else []
    bullets = gen.get("bullets", []) if isinstance(gen, dict) else []
    sources = gen.get("sources", []) if isinstance(gen, dict) else []

    if hook:
        lines.append(f"**Hook:**  \n{hook}\n")

    if paragraphs:
        lines.append("**Post:**")
        for p in paragraphs:
            lines.append(p + "\n")

    if bullets:
        lines.append("**Takeaways:**")
        for b in bullets:
            lines.append(f"- {b}")

    if sources:
        lines.append("\n**Sources:**")
        for s in sources:
            t = s.get("title", "") if isinstance(s, dict) else ""
            l = s.get("link", "") if isinstance(s, dict) else ""
            if t and l:
                lines.append(f"- {t}: {l}")
            elif l:
                lines.append(f"- {l}")
            elif t:
                lines.append(f"- {t}")

    return "\n\n".join(lines) if lines else ""

def apply_filters(events, location_filter, type_filter):
    """Apply location and type filters to events list."""
    if not events:
        return []

    filtered = events.copy()

    # Apply location filter
    if location_filter != "All":
        if location_filter == "Online":
            filtered = [e for e in filtered if e.get('location', '').lower() == 'online']
        else:
            filtered = [e for e in filtered if location_filter.lower() in e.get('location', '').lower()]

    # Apply type filter
    if type_filter != "All":
        filtered = [e for e in filtered if e.get('type', '') == type_filter]

    return filtered


def format_events_as_text(events, event_type):
    """Format events list as text for download."""
    if not events:
        return f"No {event_type} found."

    lines = [f"AI/ML {event_type.title()} - {len(events)} events\n"]
    lines.append("=" * 50)
    lines.append("")

    for i, event in enumerate(events, 1):
        lines.append(f"{i}. {event.get('title', 'No title')}")
        lines.append(f"   Date: {event.get('date', 'TBD')}")
        lines.append(f"   Location: {event.get('location', 'TBD')}")
        lines.append(f"   Type: {event.get('type', 'TBD')}")
        lines.append(f"   URL: {event.get('url', 'N/A')}")
        lines.append(f"   Source: {event.get('source', 'N/A')}")
        lines.append("")

    lines.append("=" * 50)
    lines.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return "\n".join(lines)


def format_papers_as_text(papers, topic):
    """Format research papers list as text for download."""
    if not papers:
        return f"No papers found for topic: {topic}"

    lines = [f"Research Papers - Topic: {topic} - {len(papers)} papers\n"]
    lines.append("=" * 50)
    lines.append("")

    for i, paper in enumerate(papers, 1):
        lines.append(f"{i}. {paper.get('title', 'No title')}")
        authors = paper.get("authors") or []
        if authors:
            lines.append(f"   Authors: {', '.join(authors)}")
        else:
            lines.append("   Authors: N/A")

        if paper.get("date"):
            try:
                date_str = paper['date'].date().isoformat() if hasattr(paper['date'], 'date') else str(paper['date'])
                lines.append(f"   Date: {date_str}")
            except Exception:
                lines.append(f"   Date: {paper['date']}")
        else:
            lines.append("   Date: N/A")

        if paper.get("url"):
            lines.append(f"   Abstract: {paper.get('url')}")
        if paper.get("pdf"):
            lines.append(f"   PDF: {paper.get('pdf')}")

        snippet = paper.get("snippet") or paper.get("raw_summary") or ""
        if snippet:
            lines.append(f"   Summary: {snippet[:500]}{'...' if len(snippet) > 500 else ''}")

        lines.append("")

    lines.append("=" * 50)
    lines.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return "\n".join(lines)


def get_chat_download_content(session_id: str) -> str:
    """Format a single chat session for download."""
    session_history = st.session_state["all_chat_sessions"].get(session_id, [])
    if not session_history:
        return f"No chat history found for session: {session_id}"

    lines = [f"Chat Session: {session_id}\n"]
    lines.append("=" * 50)
    lines.append("")

    for message in session_history:
        role = message.get("role", "Unknown").title()
        text = message.get("text", "")
        ts = message.get("ts", "N/A")
        lines.append(f"[{ts}] {role}: {text}")
    lines.append("")
    lines.append("=" * 50)
    lines.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return "\n".join(lines)





def extract_short_summary(item, max_chars=200) -> str:
    """
    Try several metadata fields for a short summary:
      - item['summary']
      - item['snippet']
      - item.get('page_content') first 300 chars
      - item.get('description')
    Return a single-line (or two-line) trimmed summary.
    """
    candidate = ""
    if isinstance(item, dict):
        candidate = (item.get("summary") or item.get("snippet") or item.get("description") or "").strip()
        if not candidate:
            candidate = (item.get("page_content") or item.get("content") or "").strip()
    else:
        candidate = str(item).strip()

    if not candidate:
        return ""

    s = " ".join(candidate.split())
    if len(s) <= max_chars:
        return s
    truncated = s[:max_chars].rsplit(" ", 1)[0]
    return truncated + "…"

def format_top_results_for_ui_with_summary_plain_title(state, top_n: int = 5) -> str:
    """
    Build a markdown block listing top_n results with:
      - clickable title (embedded link)
      - NO raw URL shown
      - 1–2 line summary
    """
    docs = getattr(state, "retrieved_docs", None)
    if not docs:
        docs = getattr(state, "sources", None) or []

    normalized = []
    for item in docs:
        if not item:
            continue

        title = (item.get("title") or "").strip()
        link = (item.get("link") or "").strip()
        summary = extract_short_summary(item, max_chars=200)

        if not title:
            continue

        normalized.append({
            "title": title,
            "link": link,
            "summary": summary
        })

    top = normalized[:top_n]

    if not top:
        return "No retriever results available."

    md_lines = [f"**Top {len(top)} Results:**\n"]

    for e in top:
        title = e["title"]
        link = e["link"]
        summary = e["summary"]

        # ✅ Title is clickable, link NOT shown separately
        # if link:
            # md_lines.append(f"- **[{title}]({link})**")
        if link:
            md_lines.append(f"- **[{title}]({link})**")
        else:
            md_lines.append(f"- **{title}**")

        if summary:
            md_lines.append(f"  _{summary}_")

        md_lines.append("")

    return "\n".join(md_lines)



def render_bulleted_articles(state) -> str:
    """
    Render final LLM output:
    title
    link
    bullet points
    """
    posts = getattr(state, "generated_post", None)

    if not posts or not isinstance(posts, list):
        return "No summarized results available."

    lines = []
    for i, item in enumerate(posts, 1):
        title = item.get("title", "").strip()
        link = item.get("link", "").strip()
        bullets = item.get("key_points", [])

        if not title and not bullets:
            continue

        if link:
            lines.append(f"### {i}. [{title}]({link})")
        else:
            lines.append(f"### {i}. {title}")


        # lines.append(f"### {i}. {title}")
        # if link:
        #     lines.append(link)

        if bullets:
            for b in bullets:
                lines.append(f"- {b}")
        else:
            lines.append("_No key points extracted._")

        lines.append("")  # spacing

    return "\n".join(lines)



with tab_chat:
    st.markdown("### 💬 Chat")

    # ---------------------------
    # TOOL + BLOG SELECTOR
    # ---------------------------
    BLOG_OPTIONS = list(BLOG_SOURCES.keys())

    TOOL_LABEL_MAP = {
        "AI News": "rss",
        "Blogs": "blog"
    }

    tool_display = st.selectbox(
        " ",
        list(TOOL_LABEL_MAP.keys()),
        index=1,
        key="chat_tools_selectbox",
        label_visibility="collapsed"
    )

    tool_choice_local = TOOL_LABEL_MAP[tool_display]

    if tool_choice_local == "blog":
        default_blog = st.session_state.get("selected_blog", BLOG_OPTIONS[0])
        selected_blog = st.selectbox(
            "Select Blog Source",
            BLOG_OPTIONS,
            index=BLOG_OPTIONS.index(default_blog),
            key="selected_blog_ui"
        )
        st.session_state["selected_blog"] = selected_blog
    else:
        st.session_state["selected_blog"] = None

    # ---------------------------
    # USER INPUT
    # ---------------------------
    user_query_input = st.text_input(
        "Ask a question",
        key="chat_input_tab",
        placeholder="Ask about AI news or blog insights...",
        label_visibility="collapsed"
    )

    send_pressed = st.button("Send", key="chat_send_btn")

    # ---------------------------
    # SEND HANDLER
    # ---------------------------
    if send_pressed and user_query_input:
        tool_choice = tool_choice_local
        session_id = st.session_state["current_chat_session_id"]

        # Store USER message
        st.session_state["all_chat_sessions"].setdefault(session_id, []).append({
            "role": "user",
            "text": user_query_input,
            "tool": tool_choice,
            "ts": datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
        })

        auto_build = False
        if tool_choice == "blog":
            blog_dir = BLOG_SOURCES[st.session_state["selected_blog"]]["faiss_dir"]
            if not os.path.exists(os.path.join(blog_dir, "index.faiss")):
                auto_build = True
                st.info(f"🔧 Building {st.session_state['selected_blog']} index (first time only)")

        with st.spinner("Retrieving results..."):
            try:
                state = run_full_workflow_example(
                    query=user_query_input,
                    selected_tool=tool_choice,
                    selected_blog=st.session_state.get("selected_blog"),
                    build_if_missing=auto_build
                )

                output_md = render_bulleted_articles(state)
                if not output_md or "No summarized" in output_md:
                    output_md = format_top_results_for_ui_with_summary_plain_title(state)

            except Exception as e:
                output_md = f"❌ Error: {e}\n\n{traceback.format_exc()}"

        # Store ASSISTANT message
        st.session_state["all_chat_sessions"][session_id].append({
            "role": "assistant",
            "text": output_md,
            "tool": tool_choice,
            "ts": datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
        })

        st.session_state["chat_user_input"] = ""
        st.rerun()

    # ---------------------------
    # CHAT HISTORY RENDER
    # ---------------------------
    st.markdown("### Conversation")

    history = st.session_state["all_chat_sessions"].get(
        st.session_state["current_chat_session_id"], []
    )

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

    # =========================================================
    # 🔽 SESSION + DOWNLOAD (BOTTOM — AS REQUESTED)
    # =========================================================
    st.markdown("---")
    st.subheader("Session Controls")

    col1, col2 = st.columns([1, 1])

    # ➕ New Chat
    with col1:
        if st.button("➕ Start New Chat", key="chat_new_session_btn"):
            new_session_id = f"session_{len(st.session_state['all_chat_sessions']) + 1}"
            st.session_state["all_chat_sessions"][new_session_id] = []
            st.session_state["current_chat_session_id"] = new_session_id
            st.session_state["chat_user_input"] = ""
            st.rerun()

    # 📥 Download Chat
    with col2:
        if history:
            download_content = get_chat_download_content(
                st.session_state["current_chat_session_id"]
            )
            st.download_button(
                label="📥 Download Chat (.txt)",
                data=download_content,
                file_name=f"chat_{st.session_state['current_chat_session_id']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_chat_btn"
            )

    # 🔽 Session Selector (BOTTOM)
    session_ids = list(st.session_state["all_chat_sessions"].keys())
    if session_ids:
        selected_session = st.selectbox(
            "Select Chat Session",
            session_ids,
            index=session_ids.index(st.session_state["current_chat_session_id"]),
            key="chat_session_selector_bottom"
        )
        st.session_state["current_chat_session_id"] = selected_session


DEFAULT_LOAD_MAX_DOCS = 8
DEFAULT_PER_TOPIC = 5
DEFAULT_TOP_K = 10


with tab_research:
    st.header("📄 Research Papers — arXiv")

    # ---------------- Layout: 2 x 2 ----------------
    col1, col2 = st.columns(2)
    with col1:
        topic = st.selectbox(
            "📌 Select Topic",
            options=st.session_state["research_choices"],
            index=(
                st.session_state["research_choices"].index(
                    st.session_state["research_selected"]
                )
                if st.session_state["research_selected"] in st.session_state["research_choices"]
                else 0
            ),
            key="research_topic_select"
        )

    with col2:
        keyword_input = st.text_input(
            "🔍 Search by Keyword",
            placeholder="e.g. RAG, transformers, diffusion",
            key="research_keyword_input"
        )

    col3, col4 = st.columns(2)
    with col3:
        author_input = st.text_input(
            "✍️ Search by Author",
            placeholder="e.g. Yann LeCun, Andrew Ng",
            key="research_author_input"
        )

    with col4:
        fetch_papers = st.button("▶️ Fetch Papers", key="research_fetch_btn")

    papers = []  # ✅ ensure scope exists

    # ---------------- Query Resolution ----------------
    if fetch_papers:
        author = author_input.strip()
        keyword = keyword_input.strip()

        # 🔹 Decide mode & query (PRIORITY ORDER)
        if author:
            mode = "author"
            query = author
            st.caption(f"🔎 Author search: **{author}**")

        elif keyword:
            mode = "keywords"
            query = keyword
            st.caption(f"🔎 Keyword search: **{keyword}**")

        else:
            mode = "topic"
            query = topic
            st.caption(f"🔎 Topic search: **{topic}**")

        with st.spinner("Fetching research papers from arXiv..."):
            try:
                papers = get_research_papers(query=query, mode=mode)
            except Exception as e:
                st.error(f"Failed to fetch papers: {e}")
                papers = []

    # ---------------- Results ----------------
    if fetch_papers:
        if not papers:
            st.info("No papers found for the given query.")
        else:
            st.success(f"Found {len(papers)} papers")

            # 📥 DOWNLOAD BUTTON (ADDED)
            text_content = format_papers_as_text(papers, query)
            st.download_button(
                label="📥 Download Research Papers (.txt)",
                data=text_content,
                file_name=f"research_{query}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_research"
            )

            for i, p in enumerate(papers, start=1):
                st.markdown(f"### {i}. {p.get('title', 'No title')}")

                authors = p.get("authors") or []
                if authors:
                    st.markdown(f"*Authors:* {', '.join(authors)}")

                if p.get("date"):
                    try:
                        st.markdown(f"*Date:* {p['date'].date().isoformat()}")
                    except Exception:
                        st.markdown(f"*Date:* {p['date']}")

                links = []
                if p.get("url"):
                    links.append(f"[Abstract]({p['url']})")
                if p.get("pdf"):
                    links.append(f"[PDF]({p['pdf']})")
                if links:
                    st.markdown(" | ".join(links))

                if p.get("llm_overview"):
                    st.markdown("**🧠 Overview**")
                    st.write(p["llm_overview"])

                if p.get("llm_key_points"):
                    st.markdown("**🔹 Key Contributions**")
                    for kp in p["llm_key_points"]:
                        st.markdown(f"- {kp}")

                st.markdown("---")


# with tab_research:
#     st.header("📄 Research Papers — arXiv")

#     # ---------------- Layout: 2 x 2 ----------------
#     col1, col2 = st.columns(2)
#     with col1:
#         topic = st.selectbox(
#             "📌 Select Topic",
#             options=st.session_state["research_choices"],
#             index=(
#                 st.session_state["research_choices"].index(
#                     st.session_state["research_selected"]
#                 )
#                 if st.session_state["research_selected"] in st.session_state["research_choices"]
#                 else 0
#             ),
#             key="research_topic_select"
#         )

#     with col2:
#         keyword_input = st.text_input(
#             "🔍 Search by Keyword",
#             placeholder="e.g. RAG, transformers, diffusion",
#             key="research_keyword_input"
#         )

#     col3, col4 = st.columns(2)
#     with col3:
#         author_input = st.text_input(
#             "✍️ Search by Author",
#             placeholder="e.g. Yann LeCun, Andrew Ng",
#             key="research_author_input"
#         )

#     with col4:
#         fetch_papers = st.button("▶️ Fetch Papers", key="research_fetch_btn")

#     # ---------------- Query Resolution ----------------

#     if fetch_papers:
#         author = author_input.strip()
#         keyword = keyword_input.strip()

#     # 🔹 Decide mode & query (PRIORITY ORDER)
#         if author:
#             mode = "author"
#             query = author
#             st.caption(f"🔎 Author search: **{author}**")

#         elif keyword:
#             mode = "keywords"
#             query = keyword
#             st.caption(f"🔎 Keyword search: **{keyword}**")

#         else:
#             mode = "topic"
#             query = topic
#             st.caption(f"🔎 Topic search: **{topic}**")

#         with st.spinner("Fetching research papers from arXiv..."):
#             try:
#                 papers = get_research_papers(
#                 query=query,
#                 mode=mode
#             )
#             except Exception as e:
#                 st.error(f"Failed to fetch papers: {e}")
#                 papers = []

#     # 🔹 Results
#         if not papers:
#             st.info("No papers found for the given query.")
#         else:
#             st.success(f"Found {len(papers)} papers")

#             for i, p in enumerate(papers, start=1):
#                 st.markdown(f"### {i}. {p.get('title', 'No title')}")

#                 authors = p.get("authors") or []
#                 if authors:
#                     st.markdown(f"*Authors:* {', '.join(authors)}")

#                 if p.get("date"):
#                     try:
#                         st.markdown(f"*Date:* {p['date'].date().isoformat()}")
#                     except Exception:
#                         st.markdown(f"*Date:* {p['date']}")

#                 links = []
#                 if p.get("url"):
#                     links.append(f"[Abstract]({p['url']})")
#                 if p.get("pdf"):
#                     links.append(f"[PDF]({p['pdf']})")
#                 if links:
#                     st.markdown(" | ".join(links))

#                 if p.get("llm_overview"):
#                     st.markdown("**🧠 Overview**")
#                     st.write(p["llm_overview"])

#                 if p.get("llm_key_points"):
#                     st.markdown("**🔹 Key Contributions**")
#                     for kp in p["llm_key_points"]:
#                         st.markdown(f"- {kp}")

#                 st.markdown("---")





import os
import json
HACKATHON_CACHE_FILE = "events_cache.json"
with tab_hackathons:
    st.header("AI/ML Hackathons")

    # ----------------------------
    # Filters
    # ----------------------------
    col1, col2 = st.columns(2)
    with col1:
        location_options = ["All", "Online", "Chennai", "Bangalore", "Hyderabad", "Pune", "Noida", "Delhi", "Mumbai", "Kolkata"]
        location_filter = st.selectbox(
            "Location",
            location_options,
            index=location_options.index(st.session_state.get("hackathon_location_filter", "All")),
            key="hackathon_location_select"
        )
        st.session_state["hackathon_location_filter"] = location_filter

    with col2:
        type_options = ["All", "Online", "Offline"]
        type_filter = st.selectbox(
            "Type",
            type_options,
            index=type_options.index(st.session_state.get("hackathon_type_filter", "All")),
            key="hackathon_type_select"
        )
        st.session_state["hackathon_type_filter"] = type_filter

    fetch_hackathons = st.button("Load hackathons", key="fetch_hackathons_top")

    # ----------------------------
    # Load + filter data
    # ----------------------------
    if fetch_hackathons:
        with st.spinner("Loading hackathons from cache..."):
            try:
                # 1️⃣ LOAD cached data
                all_results = get_hackathons()

                # 2️⃣ APPLY filters (POSITIONAL ARGS)
                results = apply_filters(
                    all_results,
                    location_filter,
                    type_filter
                )

            except Exception as e:
                st.error(f"Error loading hackathons: {e}")
                all_results = []
                results = []


        # ----------------------------
        # Display results
        # ----------------------------
        if not results:
            if not all_results:
                st.info("No upcoming hackathons found in cache. Run event_collector.py to update.")
            else:
                st.info("No hackathons match the selected filters.")
        else:
            st.markdown(f"**Showing {len(results)} hackathon(s)**")

            # Download button
            text_content = format_events_as_text(results, "hackathons")
            st.download_button(
                label="📥 Download Hackathons (.txt)",
                data=text_content,
                file_name=f"hackathons_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_hackathons"
            )

            for i, r in enumerate(results, start=1):
                st.markdown(f"**{i}. {r.get('title','No title')}**")
                st.markdown(f"📅 Date: {r.get('date', 'TBD')}")
                st.markdown(f"📍 Location: {r.get('location', 'TBD')}")
                st.markdown(f"🧭 Type: {r.get('type', 'TBD')}")
                st.markdown(f"🔗 [Link]({r.get('url')})")
                st.markdown(f"*Source: {r.get('source', 'N/A')}*")
                st.markdown("---")



with tab_conferences:
    st.header("AI/ML Conferences")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        location_options = ["All", "Online", "Chennai", "Bangalore", "Hyderabad", "Pune", "Noida", "Delhi", "Mumbai", "Kolkata"]
        st.session_state["conference_location_filter"] = st.selectbox(
            "Location",
            location_options,
            index=location_options.index(st.session_state.get("conference_location_filter", "All")),
            key="conference_location_select"
        )

    with col2:
        type_options = ["All", "Online", "Offline"]
        st.session_state["conference_type_filter"] = st.selectbox(
            "Type",
            type_options,
            index=type_options.index(st.session_state.get("conference_type_filter", "All")),
            key="conference_type_select"
        )

    fetch_conferences = st.button("Load conferences", key="fetch_conferences_top")

    if fetch_conferences:
        with st.spinner("Loading conferences from cache..."):
            try:
                all_results = get_conferences()
                results = apply_filters(
                    all_results,
                    st.session_state["conference_location_filter"],
                    st.session_state["conference_type_filter"]
                )
            except Exception as e:
                st.error(f"Error loading conferences: {e}")
                results = []

            if not results:
                if not all_results:
                    st.info("No upcoming conferences found in cache. Run event_collector.py to update.")
                else:
                    st.info("No conferences match the selected filters.")
            else:
                st.markdown(f"**Showing {len(results)} conference(s)**")

                # Download button
                text_content = format_events_as_text(results, "conferences")
                st.download_button(
                    label="📥 Download Conferences (.txt)",
                    data=text_content,
                    file_name=f"conferences_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_conferences"
                )

                for i, r in enumerate(results, start=1):
                    st.markdown(f"**{i}. {r.get('title','No title')}**")
                    st.markdown(f"Date: {r.get('date')}")
                    st.markdown(f"Location: {r.get('location', 'TBD')}")
                    st.markdown(f"Type: {r.get('type', 'TBD')}")
                    st.markdown(f"[Link]({r.get('url')})")
                    st.markdown(f"*Source: {r.get('source')}*")
                    st.markdown("---")


with tab_workshops:
    st.header("AI/ML Workshops")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        location_options = ["All", "Online", "Chennai", "Bangalore", "Hyderabad", "Pune", "Noida", "Delhi", "Mumbai", "Kolkata"]
        st.session_state["workshop_location_filter"] = st.selectbox(
            "Location",
            location_options,
            index=location_options.index(st.session_state.get("workshop_location_filter", "All")),
            key="workshop_location_select"
        )

    with col2:
        type_options = ["All", "Online", "Offline"]
        st.session_state["workshop_type_filter"] = st.selectbox(
            "Type",
            type_options,
            index=type_options.index(st.session_state.get("workshop_type_filter", "All")),
            key="workshop_type_select"
        )

    fetch_workshops = st.button("Load workshops", key="fetch_workshops_top")

    if fetch_workshops:
        with st.spinner("Loading workshops from cache..."):
            try:
                all_results = get_workshops()
                results = apply_filters(
                    all_results,
                    st.session_state["workshop_location_filter"],
                    st.session_state["workshop_type_filter"]
                )
            except Exception as e:
                st.error(f"Error loading workshops: {e}")
                results = []

            if not results:
                if not all_results:
                    st.info("No upcoming workshops found in cache. Run event_collector.py to update.")
                else:
                    st.info("No workshops match the selected filters.")
            else:
                st.markdown(f"**Showing {len(results)} workshop(s)**")

                # Download button
                text_content = format_events_as_text(results, "workshops")
                st.download_button(
                    label="📥 Download Workshops (.txt)",
                    data=text_content,
                    file_name=f"workshops_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_workshops"
                )

                for i, r in enumerate(results, start=1):
                    st.markdown(f"**{i}. {r.get('title','No title')}**")
                    st.markdown(f"Date: {r.get('date')}")
                    st.markdown(f"Location: {r.get('location', 'TBD')}")
                    st.markdown(f"Type: {r.get('type', 'TBD')}")
                    st.markdown(f"[Link]({r.get('url')})")
                    st.markdown(f"*Source: {r.get('source')}*")
                    st.markdown("---")
col_left, col_right = st.columns([3, 1])
with col_left:
    st.write("Session info:")
    st.write(f"Selected tool (last message): **{st.session_state['last_tool']}**")
    st.write(f"Chat sessions: {len(st.session_state['all_chat_sessions'])}")

with col_right:
    if st.button("Rebuild indexes (slow)", key="footer_rebuild_btn"):
        with st.spinner("Rebuilding indexes... this runs scraping + indexing and may take many minutes"):
            try:
                res = run_full_workflow_example(query="", selected_tool="rss", build_if_missing=True)
            except Exception as e:
                st.error(f"Rebuild failed with exception: {e}")
            else:
                if getattr(res, "status", "") == "error" or getattr(res, "error", None):
                    st.error(f"Rebuild failed: {getattr(res, 'error', 'unknown error')}")
                else:
                    st.success("Rebuild finished. Indexes persisted and ready for queries.")

    if st.button("Clear chat", key="footer_clear_btn"):
        # Clear current chat session
        if st.session_state["current_chat_session_id"] in st.session_state["all_chat_sessions"]:
            st.session_state["all_chat_sessions"][st.session_state["current_chat_session_id"]] = []
            st.session_state["chat_history"] = []
        st.session_state["last_assistant_render"] = ""
        st.session_state["last_state"] = None
        st.session_state["chat_user_input"] = ""
        if hasattr(st, "experimental_rerun"):
            st.rerun()
