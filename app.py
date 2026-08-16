"""Streamlit Web Dashboard for Second Brain — Nordic Aurora Glass Theme.

Design System: Data-Dense Dashboard (dark tech palette)
Typography: Fira Sans (UI) + Fira Code (data/metrics)
"""

import json
import math
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

import config
from utils import list_raw_captures, list_wiki_notes, read_frontmatter
from capture import capture, DuplicateError, clean_extracted_pdf_text
from classify import classify_all_pending
from link import link_all_notes
from build_graph import export_graph, build_graph
from ask import ask
from manage_notes import delete_note, update_note, get_backlinks, search_notes_semantic
from export_import import ingest_uploaded_file, generate_zip_backup
from analytics import get_analytics_data

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Second Brain — Knowledge Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System CSS ────────────────────────────────────────
style_css_path = config.STATIC_DIR / "style.css"

if style_css_path.exists():
    with open(style_css_path, "r", encoding="utf-8") as f:
        css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


# ── SVG Icon Helpers (Lucide-style) ──────────────────────────
def _svg(icon: str, size: int = 18, color: str = "currentColor") -> str:
    """Return inline SVG from a Lucide path + stroke attributes."""
    paths = {
        "brain":       '<path d="M12 2a7 7 0 0 0-7 7c0 2.4 1.2 4.5 3 5.7V17a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-2.3c1.8-1.2 3-3.3 3-5.7a7 7 0 0 0-7-7z"/>',
        "database":    '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5"/><path d="M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/>',
        "link":        '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
        "folder":      '<path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2z"/>',
        "send":        '<line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>',
        "file":        '<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/>',
        "trash":       '<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
        "search":      '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
        "refresh":     '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>',
        "download":    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
        "upload":      '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
        "git-branch":  '<line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/>',
        "bar-chart":   '<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>',
        "globe":       '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>',
        "message":     '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
        "grid":        '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>',
        "pie-chart":   '<path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/>',
        "edit":        '<path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>',
    }
    path_data = paths.get(icon, paths["file"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{path_data}</svg>'
    )


def _bento_card(label: str, value: str, icon: str = "", hint: str = "", accent: str = "#38bdf8") -> str:
    """Render a bento metric card as HTML."""
    icon_html = f'<span style="color:{accent}">{_svg(icon, 16, accent)}</span>' if icon else ""
    hint_html = f'<div class="bento-hint">{hint}</div>' if hint else ""
    return (
        f'<div class="bento-card">'
        f'  <div class="bento-label">{icon_html} {label}</div>'
        f'  <div class="bento-value" style="color:{accent}">{value}</div>'
        f'  {hint_html}'
        f'</div>'
    )


# ── Data Layer ───────────────────────────────────────────────
def get_stats():
    raw_files = list_raw_captures()
    wiki_files = list_wiki_notes()
    cats = {"Projects": 0, "Areas": 0, "Resources": 0, "Archives": 0}
    total_links = 0
    for note_path in wiki_files:
        meta, _ = read_frontmatter(note_path)
        cat = meta.get("category", "Resources")
        if cat in cats:
            cats[cat] += 1
        total_links += len(meta.get("links", []))
    return {
        "raw_count": len(raw_files),
        "wiki_count": len(wiki_files),
        "links_count": total_links // 2,
        "categories": cats,
    }


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Brand ────────────────────────────────────────────────
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'>"
        f"<div style='width:40px;height:40px;background:#1C1917;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#FAF8F5;'>"
        f"{_svg('brain', 22, '#FAF8F5')}"
        f"</div>"
        f"<div><h2 style='margin:0;font-size:1.35rem;font-family:var(--font-editorial);font-style:italic;'>Second Brain</h2>"
        f"<div style='font-size:0.75rem;color:#78716C;'>Self-Organizing Knowledge OS</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Quick Capture ────────────────────────────────────────
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:4px'>"
        f"{_svg('send', 16, '#1D4ED8')}"
        f"<span style='font-size:0.78rem;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.08em;color:#78716C'>Quick Capture</span></div>",
        unsafe_allow_html=True,
    )

    source_type = st.selectbox("Type", ["note", "link"], index=0, label_visibility="collapsed",
                                help="note = free-text note, link = web article URL")

    if "capture_counter" not in st.session_state:
        st.session_state["capture_counter"] = 0
    if "uploader_key_counter" not in st.session_state:
        st.session_state["uploader_key_counter"] = 0

    note_key = f"quick_note_{st.session_state['capture_counter']}"
    link_key = f"quick_link_{st.session_state['capture_counter']}"

    if source_type == "note":
        input_content = st.text_area(
            "Note Content",
            key=note_key,
            placeholder="Type your thoughts, ideas, or meeting notes here...",
            height=120,
        )
    else:
        input_content = st.text_input(
            "Web Article URL",
            key=link_key,
            placeholder="https://example.com/article...",
        )

    if st.button("Capture to Second Brain", use_container_width=True, type="primary"):
        if input_content and input_content.strip():
            try:
                cid = capture(input_content.strip(), source_type)
                with st.spinner("Auto-classifying & linking..."):
                    classify_all_pending()
                    link_all_notes()
                    export_graph()
                st.session_state["capture_counter"] += 1
                st.toast(f"Captured ID: {cid[:8]}... Brain updated!", icon="🧠")
                st.rerun()
            except DuplicateError:
                st.warning("Content already exists in Second Brain!")
            except Exception as e:
                st.error(f"Capture error: {e}")
        else:
            st.warning("Content cannot be empty.")

    # ── Document Upload ──────────────────────────────────────
    st.divider()
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:4px'>"
        f"{_svg('upload', 16, '#7C3AED')}"
        f"<span style='font-size:0.78rem;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.08em;color:#78716C'>Document Upload</span></div>",
        unsafe_allow_html=True,
    )

    uploader_key = f"sidebar_file_uploader_{st.session_state['uploader_key_counter']}"
    up_file = st.file_uploader(
        "Choose document or media file",
        type=["txt", "md", "py", "js", "json", "pdf", "png", "jpg", "jpeg", "mp4", "mov", "webm"],
        key=uploader_key,
        help="PDF documents, code files, images, or video clips",
    )
    if st.button("Upload & Process File", use_container_width=True):
        if up_file:
            with st.spinner("Extracting & processing..."):
                try:
                    cid = ingest_uploaded_file(up_file)
                    st.session_state["uploader_key_counter"] += 1
                    st.toast(f"Ingested ID: {cid[:8]}", icon="📁")
                    st.rerun()
                except DuplicateError:
                    st.warning("File already captured!")
                except Exception as e:
                    st.error(f"Ingestion error: {e}")
        else:
            st.warning("Please select a file first.")

    # ── Backup & System ──────────────────────────────────────
    st.divider()
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:4px'>"
        f"{_svg('database', 16, '#C2410C')}"
        f"<span style='font-size:0.78rem;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.08em;color:#78716C'>Backup & System</span></div>",
        unsafe_allow_html=True,
    )

    st.download_button(
        label="Download Wiki Backup (ZIP)",
        data=generate_zip_backup(),
        file_name="second_brain_wiki_backup.zip",
        mime="application/zip",
        use_container_width=True,
    )

    if st.button("Re-run Pipeline (Classify + Link)", use_container_width=True):
        with st.spinner("Processing..."):
            classify_all_pending()
            link_all_notes()
            export_graph()
        st.toast("Pipeline refreshed!", icon="🔄")
        st.rerun()


# ═══════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════
stats = get_stats()

# ── Header ───────────────────────────────────────────────────
st.markdown(
    f"""
    <div class='cyber-header-box'>
        <h1 class='gradient-title'>Personal Knowledge Hub</h1>
        <div class='sub-banner'>
            <span>Self-Organizing PARA Framework</span> • 
            <span>384-Dim Vector Embeddings</span> • 
            <span>RAG Query Engine</span>
        </div>
        <div style='margin-top: 10px;'>
            <div class='sys-chip'><span class='dot'></span>System Operational · {stats['wiki_count']} notes indexed · {stats['links_count']} semantic links</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Bento Metrics Row ────────────────────────────────────────
bento_row = "".join([
    _bento_card("Raw Captures",    str(stats["raw_count"]),                                  icon="database",   accent="#7C3AED"),
    _bento_card("Wiki Notes",      str(stats["wiki_count"]),                                 icon="grid",       accent="#1D4ED8"),
    _bento_card("Knowledge Links", str(stats["links_count"]),                                icon="link",       accent="#4D7C0F"),
    _bento_card("Projects / Areas",f"{stats['categories']['Projects']} / {stats['categories']['Areas']}", icon="git-branch", accent="#C2410C"),
    _bento_card("Resources",       str(stats["categories"]["Resources"]),                    icon="folder",     accent="#1D4ED8"),
])
st.markdown(
    f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:1.4rem">{bento_row}</div>',
    unsafe_allow_html=True,
)

# ── Workspace Tabs ───────────────────────────────────────────
tab_graph, tab_ask, tab_library, tab_analytics = st.tabs([
    "🕸️ Interactive Graph",
    "💬 Ask Second Brain",
    "📚 Knowledge Library",
    "📊 Analytics",
])


# ══════════════════════════════════════════════════════════════
#  TAB 1: GRAPH
# ══════════════════════════════════════════════════════════════
with tab_graph:
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.subheader("Knowledge Network Map")
        st.caption("Drag, zoom, search and click nodes to open notes.")
    with col_hdr2:
        if st.button("🔄 Refresh Graph", use_container_width=True, key="refresh_graph"):
            export_graph()
            st.rerun()

    export_graph()
    graph_html_path = config.STATIC_DIR / "graph.html"
    if graph_html_path.exists():
        with open(graph_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        graph_data = build_graph()
        json_str = json.dumps(graph_data, ensure_ascii=False)
        html_content = html_content.replace(
            '<script src="graph_data.js"></script>',
            f'<script src="graph_data.js"></script>\n  <script>window.GRAPH_DATA={json_str};</script>',
        )
        components.html(html_content, height=680, scrolling=False)
    else:
        st.error("graph.html not found in static/")


# ══════════════════════════════════════════════════════════════
#  TAB 2: RAG CHAT
# ══════════════════════════════════════════════════════════════
with tab_ask:
    st.subheader("Ask Your Second Brain")
    st.caption("Synthesize answers from your personal notes with multi-turn memory.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    col_chat_hdr1, col_chat_hdr2 = st.columns([3, 1])
    with col_chat_hdr2:
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()

    # Quick prompt chips
    st.markdown("##### 💡 Try these")
    chip_cols = st.columns(3)
    prompt_to_send = None
    if chip_cols[0].button("🐍 Python notes I have", use_container_width=True, key="chip1"):
        prompt_to_send = "What Python notes do I have in my brain?"
    if chip_cols[1].button("📋 PM mock interview", use_container_width=True, key="chip2"):
        prompt_to_send = "Summarize my PM mock interview note"
    if chip_cols[2].button("🔗 Virtual environments", use_container_width=True, key="chip3"):
        prompt_to_send = "What is a Python virtual environment and why use it?"

    st.divider()

    # Render conversation
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📌 Cited Sources"):
                    for src in msg["sources"]:
                        cat_class = f"badge-{src['category'].lower()}"
                        st.markdown(
                            f"<span class='{cat_class}'>{src['category']}</span> "
                            f"**{src['title']}** — {int(src['similarity'] * 100)}% match",
                            unsafe_allow_html=True,
                        )
                        st.caption(src["content"][:220] + ("..." if len(src["content"]) > 220 else ""))

    user_input = st.chat_input("Ask your Second Brain anything...")
    if user_input:
        prompt_to_send = user_input

    if prompt_to_send:
        st.session_state.chat_messages.append({"role": "user", "content": prompt_to_send})
        with st.spinner("Searching knowledge base & synthesizing..."):
            history = st.session_state.chat_messages[:-1]
            res = ask(prompt_to_send, conversation_history=history)
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": res["answer"],
            "sources": res.get("sources", []),
        })
        st.rerun()


# ══════════════════════════════════════════════════════════════
#  TAB 3: KNOWLEDGE LIBRARY
# ══════════════════════════════════════════════════════════════
with tab_library:
    st.subheader("Knowledge Library Explorer")

    wiki_notes = list_wiki_notes()
    if not wiki_notes:
        st.info("No notes found. Capture some notes to get started!")
    else:
        # Build ID lookup + collect tags
        id_to_meta = {}
        all_tags = set()
        all_note_tuples = []

        for note_path in wiki_notes:
            meta, body = read_frontmatter(note_path)
            nid = meta.get("id", note_path.stem)
            id_to_meta[nid] = meta
            all_note_tuples.append((note_path, meta, body))
            for t in meta.get("tags", []):
                if t:
                    all_tags.add(t.strip())
        sorted_all_tags = sorted(all_tags)

        # ── Filters ──────────────────────────────────────────
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            filter_cat = st.radio("Category", ["All"] + config.PARA_CATEGORIES, horizontal=True)
        with col_f2:
            selected_tags = st.multiselect("Tags", sorted_all_tags, placeholder="Filter by tags...")

        col_s1, col_s2 = st.columns([3, 1])
        with col_s1:
            search_kw = st.text_input("Search Title, Tags, or Content", placeholder="e.g. concurrency, architecture...")
        with col_s2:
            search_mode = st.selectbox("Mode", ["🤖 AI Search", "🔤 Keyword"], index=0)

        # ── Filter Logic ─────────────────────────────────────
        displayed_notes = []

        if search_kw.strip() and search_mode == "🤖 AI Search":
            with st.spinner("Searching..."):
                semantic_results = search_notes_semantic(search_kw.strip(), limit=50)
                for item in semantic_results:
                    cat = item["meta"].get("category", "Resources")
                    tags = item["meta"].get("tags", [])
                    if filter_cat != "All" and cat != filter_cat:
                        continue
                    if selected_tags and not any(t in tags for t in selected_tags):
                        continue
                    displayed_notes.append((item["path"], item["meta"], item["body"], item["similarity"]))
        else:
            for note_path, meta, body in all_note_tuples:
                cat = meta.get("category", "Resources")
                title = meta.get("title", note_path.stem)
                tags = meta.get("tags", [])
                if filter_cat != "All" and cat != filter_cat:
                    continue
                if selected_tags and not any(t in tags for t in selected_tags):
                    continue
                if search_kw.strip():
                    kw = search_kw.lower().strip()
                    if not (kw in title.lower() or kw in body.lower() or any(kw in t.lower() for t in tags)):
                        continue
                displayed_notes.append((note_path, meta, body, None))

        total_notes = len(displayed_notes)
        if total_notes == 0:
            if search_kw.strip():
                st.info(f"No notes match '{search_kw}'. Try clearing filters or changing search mode.")
            else:
                st.info("No notes match the selected filters.")
        else:
            # ── Pagination ───────────────────────────────────
            col_p1, col_p2 = st.columns([3, 1])
            with col_p2:
                page_size = st.selectbox("Per page", [5, 10, 20], index=1)
            total_pages = max(1, math.ceil(total_notes / page_size))

            if "library_page" not in st.session_state:
                st.session_state.library_page = 1
            st.session_state.library_page = min(st.session_state.library_page, total_pages)

            with col_p1:
                st.caption(f"**{total_notes}** notes · Page **{st.session_state.library_page}** of **{total_pages}**")

            start_idx = (st.session_state.library_page - 1) * page_size
            page_notes = displayed_notes[start_idx : start_idx + page_size]

            for note_tuple in page_notes:
                note_path, meta, body = note_tuple[0], note_tuple[1], note_tuple[2]
                sim = note_tuple[3] if len(note_tuple) > 3 else None

                note_id = meta.get("id", note_path.stem)
                cat_class = f"badge-{meta.get('category', 'resources').lower()}"
                match_badge = f" · 🎯 {int(sim * 100)}% Match" if sim is not None else ""
                cat_label = meta.get("category", "Resources").upper()
                expander_title = f"[{cat_label}]  {meta.get('title', note_path.stem)}{match_badge}"

                with st.expander(expander_title):
                    col_n1, col_n2 = st.columns([3, 1])
                    with col_n1:
                        st.markdown(f"**Category:** <span class='{cat_class}'>{meta.get('category')}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Created:** `{meta.get('created', 'N/A')}`")
                        tags_html = " ".join([f"<span class='tag-chip'>#{t}</span>" for t in meta.get("tags", [])])
                        st.markdown(f"**Tags:** {tags_html}", unsafe_allow_html=True)
                    with col_n2:
                        st.markdown(f"**ID:** `{note_id[:8]}`")
                        st.markdown(f"**Links:** `{len(meta.get('links', []))}`")

                    st.markdown("---")
                    st.markdown(f'<div class="note-preview-box">{clean_extracted_pdf_text(body)}</div>', unsafe_allow_html=True)

                    # ── Connected Notes ───────────────────────
                    st.markdown("---")
                    st.markdown("##### 🔗 Connected Knowledge")
                    col_link_out, col_link_in = st.columns(2)

                    with col_link_out:
                        st.markdown("**Outgoing →**")
                        for l in meta.get("links", []):
                            target_id = l.get("id")
                            target_info = id_to_meta.get(target_id, {})
                            t_cat_class = f"badge-{target_info.get('category', 'resources').lower()}"
                            st.markdown(
                                f"<span class='{t_cat_class}'>{target_info.get('category', '?')}</span> "
                                f"**{target_info.get('title', target_id[:8])}** "
                                f"`{int(l.get('similarity', 0) * 100)}%`",
                                unsafe_allow_html=True,
                            )
                        if not meta.get("links"):
                            st.caption("No outgoing links")

                    with col_link_in:
                        st.markdown("**← Backlinks**")
                        backlinks = get_backlinks(note_id)
                        for bl in backlinks:
                            bl_cat_class = f"badge-{bl.get('category', 'resources').lower()}"
                            st.markdown(
                                f"<span class='{bl_cat_class}'>{bl.get('category', '?')}</span> "
                                f"**{bl.get('title')}** "
                                f"`{int(bl.get('similarity', 0) * 100)}%`",
                                unsafe_allow_html=True,
                            )
                        if not backlinks:
                            st.caption("No incoming backlinks")

                    # ── Edit / Delete ─────────────────────────
                    st.divider()
                    col_btn1, col_btn2 = st.columns([1, 1])

                    with col_btn1:
                        with st.expander("✏️ Edit Note"):
                            with st.form(f"edit_form_{note_id}"):
                                edit_title = st.text_input("Title", value=meta.get("title", ""))
                                edit_cat = st.selectbox(
                                    "Category",
                                    config.PARA_CATEGORIES,
                                    index=config.PARA_CATEGORIES.index(meta.get("category", "Resources"))
                                    if meta.get("category") in config.PARA_CATEGORIES else 2,
                                )
                                edit_tags_str = st.text_input("Tags (comma separated)", value=", ".join(meta.get("tags", [])))
                                edit_body = st.text_area("Body Content", value=body, height=160)

                                if st.form_submit_button("Save Changes", use_container_width=True):
                                    parsed_tags = [t.strip() for t in edit_tags_str.split(",") if t.strip()]
                                    update_note(
                                        note_id=note_id,
                                        new_title=edit_title,
                                        new_category=edit_cat,
                                        new_tags=parsed_tags,
                                        new_body=edit_body,
                                    )
                                    st.toast("Note updated!", icon="✅")
                                    st.rerun()

                    with col_btn2:
                        with st.popover("🗑️ Delete Note", use_container_width=True):
                            st.warning("⚠️ Delete this note permanently?")
                            if st.button("Yes, Delete Note", key=f"del_btn_{note_id}", type="primary", use_container_width=True):
                                delete_note(note_id)
                                st.toast("Note deleted!", icon="🗑️")
                                st.rerun()

            # Pagination controls
            if total_pages > 1:
                st.write("")
                col_pg1, _, col_pg3 = st.columns([1, 2, 1])
                with col_pg1:
                    if st.button("◀ Previous", disabled=st.session_state.library_page <= 1, key="pg_prev"):
                        st.session_state.library_page -= 1
                        st.rerun()
                with col_pg3:
                    if st.button("Next ▶", disabled=st.session_state.library_page >= total_pages, key="pg_next"):
                        st.session_state.library_page += 1
                        st.rerun()


# ══════════════════════════════════════════════════════════════
#  TAB 4: ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab_analytics:
    st.subheader("Knowledge Base Analytics")
    st.caption("Deep metrics into knowledge accumulation, graph connectivity, and category balance.")

    analytics = get_analytics_data()

    # Scorecards
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    col_a1.metric("Total Captures", analytics["raw_count"])
    col_a2.metric("Knowledge Notes", analytics["wiki_count"])
    col_a3.metric("Graph Connections", analytics["total_links"])
    col_a4.metric("Avg Links / Note", analytics["avg_links_per_note"])

    st.divider()

    # Row 1: Growth + Category
    col_chart1, col_chart2 = st.columns([3, 2])

    with col_chart1:
        st.markdown("##### Knowledge Growth Timeline")
        timeline = analytics["growth_timeline"]
        if timeline:
            cumulative_data = {t["date"]: t["cumulative"] for t in timeline}
            st.line_chart(cumulative_data, height=260)
        else:
            st.info("No timeline data yet.")

    with col_chart2:
        st.markdown("##### PARA Distribution")
        cats = analytics["categories"]
        st.bar_chart(cats, height=260)

    st.divider()

    # Row 2: Top Notes + Tags
    col_lead1, col_lead2 = st.columns(2)

    with col_lead1:
        st.markdown("##### 🏆 Most Connected Notes")
        for rank, note in enumerate(analytics.get("top_connected", []), 1):
            cat_class = f"badge-{note['category'].lower()}"
            st.markdown(
                f"**#{rank}** <span class='{cat_class}'>{note['category']}</span> "
                f"**{note['title']}** — {note['link_count']} connections",
                unsafe_allow_html=True,
            )
        if not analytics.get("top_connected"):
            st.info("No connected notes yet.")

    with col_lead2:
        st.markdown("##### 🏷️ Popular Tags")
        top_tags = analytics.get("top_tags", [])
        if top_tags:
            tag_html = " ".join([
                f"<span class='tag-chip'>#{tag} ({count})</span>"
                for tag, count in top_tags
            ])
            st.markdown(tag_html, unsafe_allow_html=True)
        else:
            st.info("No tags found across notes.")
