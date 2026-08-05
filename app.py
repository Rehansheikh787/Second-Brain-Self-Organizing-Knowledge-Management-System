"""Streamlit Web Dashboard for Second Brain — Graph Visualization, RAG Search, and Knowledge Explorer."""

import json
import math
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

import config
from utils import list_raw_captures, list_wiki_notes, read_frontmatter
from capture import capture, DuplicateError
from classify import classify_all_pending
from link import link_all_notes
from build_graph import export_graph, build_graph
from ask import ask
from manage_notes import delete_note, update_note, get_backlinks
from export_import import ingest_uploaded_file, generate_zip_backup

# Streamlit Page Config
st.set_page_config(
    page_title="Second Brain - Knowledge Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Dark Glassmorphism UI
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 14px 18px;
        backdrop-filter: blur(10px);
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(30, 41, 59, 0.5);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        color: #94a3b8;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        font-weight: 600;
    }

    /* Source Citation Cards */
    .citation-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    
    .badge-projects { background-color: #f59e0b; color: #0f172a; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
    .badge-areas { background-color: #3b82f6; color: #ffffff; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
    .badge-resources { background-color: #10b981; color: #0f172a; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
    .badge-archives { background-color: #64748b; color: #ffffff; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }

    /* RAG Answer Container */
    .answer-box {
        background: rgba(30, 41, 59, 0.8);
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 18px;
        margin-top: 15px;
        margin-bottom: 20px;
        font-size: 1.05rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


def get_stats():
    """Calculate dashboard statistics."""
    raw_files = list_raw_captures()
    wiki_files = list_wiki_notes()
    
    cats = {"Projects": 0, "Areas": 0, "Resources": 0, "Archives": 0}
    total_links = 0
    
    for note_path in wiki_files:
        meta, _ = read_frontmatter(note_path)
        cat = meta.get("category", "Resources")
        if cat in cats:
            cats[cat] += 1
        links = meta.get("links", [])
        total_links += len(links)
        
    return {
        "raw_count": len(raw_files),
        "wiki_count": len(wiki_files),
        "links_count": total_links // 2,
        "categories": cats
    }


# === SIDEBAR: QUICK CAPTURE & PIPELINE ===
with st.sidebar:
    st.image("https://img.icons8.com/isometric-line/100/38bdf8/brain.png", width=64)
    st.title("Second Brain")
    st.caption("Self-Organizing Knowledge Management")
    st.divider()

    st.subheader("📥 Quick Capture")
    source_type = st.selectbox("Type", ["note", "link", "file"], index=0)
    input_content = st.text_area("Content or URL / File Path", placeholder="Enter your note text or URL...", height=120)

    if st.button("Capture Note", use_container_width=True, type="primary"):
        if input_content.strip():
            try:
                cid = capture(input_content.strip(), source_type)
                st.success(f"Captured ID: {cid[:8]}...")
                
                with st.spinner("Auto-classifying & linking..."):
                    classify_all_pending()
                    link_all_notes()
                    export_graph()
                st.toast("Brain updated successfully!", icon="🧠")
                st.rerun()
            except DuplicateError as e:
                st.warning("Note already exists in Second Brain!")
            except Exception as e:
                st.error(f"Capture error: {e}")
        else:
            st.warning("Content cannot be empty.")

    st.divider()
    st.subheader("📁 Upload File or Media")
    up_file = st.file_uploader(
        "Choose document or media file",
        type=["txt", "md", "py", "js", "json", "pdf", "png", "jpg", "jpeg", "mp4", "mov", "webm"],
        key="sidebar_file_uploader"
    )
    if st.button("Ingest Uploaded File", use_container_width=True):
        if up_file:
            with st.spinner("Extracting & processing pipeline..."):
                try:
                    cid = ingest_uploaded_file(up_file)
                    st.success(f"Ingested ID: {cid[:8]}")
                    st.toast("File captured & classified!", icon="📁")
                    st.rerun()
                except DuplicateError:
                    st.warning("File already captured in Second Brain!")
                except Exception as e:
                    st.error(f"Ingestion error: {e}")
        else:
            st.warning("Please select a file first.")

    st.divider()
    st.caption("📦 Backup & System")
    st.download_button(
        label="Download Wiki Backup (ZIP)",
        data=generate_zip_backup(),
        file_name="second_brain_wiki_backup.zip",
        mime="application/zip",
        use_container_width=True
    )
    
    if st.button("Re-run Classify & Auto-Link", use_container_width=True):
        with st.spinner("Processing pipeline..."):
            classify_all_pending()
            link_all_notes()
            export_graph()
        st.success("Pipeline refreshed!")
        st.rerun()


# === MAIN CONTENT ===
stats = get_stats()

# Header Banner
st.title("🧠 Personal Knowledge Hub")
st.markdown("Automated PARA Organization • Semantic Embedding Graph • Natural Language RAG")

# Top KPI Metric Cards
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Raw Captures", stats["raw_count"])
col2.metric("Wiki Notes", stats["wiki_count"])
col3.metric("Knowledge Links", stats["links_count"])
col4.metric("Projects / Areas", f"{stats['categories']['Projects']} / {stats['categories']['Areas']}")
col5.metric("Resources", stats['categories']['Resources'])

st.write("")

# Main Workspace Tabs
tab_graph, tab_ask, tab_library = st.tabs([
    "🕸️ Interactive Graph",
    "🤖 Ask Second Brain (RAG)",
    "📚 Knowledge Library"
])


# === TAB 1: GRAPH VISUALIZATION ===
with tab_graph:
    st.subheader("Knowledge Network Map")
    st.caption("Drag, zoom, search, and click nodes to open detailed note content.")
    
    # Ensure fresh graph.json export
    graph_file = config.GRAPH_JSON
    if not graph_file.exists():
        export_graph()
        
    graph_html_path = config.STATIC_DIR / "graph.html"
    if graph_html_path.exists():
        with open(graph_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        # Inject current graph data directly into HTML for Streamlit iframe compatibility
        graph_data = build_graph()
        json_str = json.dumps(graph_data, ensure_ascii=False)
        injected_script = f"<script>window.GRAPH_DATA = {json_str};</script>"
        html_content = html_content.replace("<head>", f"<head>\n  {injected_script}")
            
        components.html(html_content, height=680, scrolling=False)
    else:
        st.error("Graph UI HTML template not found in static/graph.html")


# === TAB 2: RAG Q&A (THE ORACLE) ===
with tab_ask:
    st.subheader("Ask Your Second Brain")
    st.caption("Synthesize answers grounded strictly in your personal notes.")
    
    question = st.text_input(
        "Ask a question",
        placeholder="e.g. What is Python virtual environment and why do we use it?",
        key="rag_query_input"
    )
    
    col_ask1, col_ask2 = st.columns([1, 4])
    top_k = col_ask1.slider("Context Notes (Top K)", min_value=1, max_value=8, value=4)
    
    if question.strip():
        with st.spinner("Synthesizing answer from your notes..."):
            res = ask(question.strip(), top_k=top_k)
            
        st.markdown(f'<div class="answer-box">{res["answer"]}</div>', unsafe_allow_html=True)
        st.caption(f"🎯 Context Match Confidence: **{int(res['confidence'] * 100)}%**")
        
        st.divider()
        st.markdown("##### 📌 Cited Source Notes")
        
        if res["sources"]:
            for src in res["sources"]:
                cat_class = f"badge-{src['category'].lower()}"
                with st.expander(f"[{src['category']}] {src['title']} (Similarity: {int(src['similarity'] * 100)}%)"):
                    st.markdown(f"**Category:** <span class='{cat_class}'>{src['category']}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Tags:** {', '.join(['#' + t for t in src.get('tags', [])])}")
                    st.markdown("**Content:**")
                    st.info(src["content"])
        else:
            st.info("No matching source notes were cited.")


# === TAB 3: KNOWLEDGE LIBRARY ===
with tab_library:
    st.subheader("Knowledge Library Explorer")
    
    wiki_notes = list_wiki_notes()
    if not wiki_notes:
        st.info("No notes found in wiki/. Capture some notes to get started!")
    else:
        # Build quick ID lookup map for titles
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
        
        # Category and Tag filters
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            filter_cat = st.radio(
                "Filter Category",
                ["All"] + config.PARA_CATEGORIES,
                horizontal=True
            )
        with col_f2:
            selected_tags = st.multiselect("Filter Tags", sorted_all_tags, placeholder="Select tags...")
            
        search_kw = st.text_input("Search Title, Tags, or Content", placeholder="Type keywords...")
        
        # Filter notes logic
        displayed_notes = []
        for note_path, meta, body in all_note_tuples:
            cat = meta.get("category", "Resources")
            title = meta.get("title", note_path.stem)
            tags = meta.get("tags", [])
            
            if filter_cat != "All" and cat != filter_cat:
                continue
                
            if selected_tags:
                if not any(st_tag in tags for st_tag in selected_tags):
                    continue
                    
            if search_kw.strip():
                kw = search_kw.lower().strip()
                match_title = kw in title.lower()
                match_body = kw in body.lower()
                match_tag = any(kw in t.lower() for t in tags)
                if not (match_title or match_body or match_tag):
                    continue
                    
            displayed_notes.append((note_path, meta, body))
            
        # Pagination setup
        col_p1, col_p2 = st.columns([3, 1])
        with col_p2:
            page_size = st.selectbox("Notes per page", [5, 10, 20], index=1)
            
        total_notes = len(displayed_notes)
        total_pages = max(1, math.ceil(total_notes / page_size))
        
        if "library_page" not in st.session_state:
            st.session_state.library_page = 1
        st.session_state.library_page = min(st.session_state.library_page, total_pages)
        
        with col_p1:
            st.caption(f"Showing {total_notes} notes • Page {st.session_state.library_page} of {total_pages}")
            
        # Slice page items
        start_idx = (st.session_state.library_page - 1) * page_size
        end_idx = start_idx + page_size
        page_notes = displayed_notes[start_idx:end_idx]
        
        # Render notes
        for note_path, meta, body in page_notes:
            note_id = meta.get("id", note_path.stem)
            cat_class = f"badge-{meta.get('category', 'resources').lower()}"
            
            with st.expander(f"[{meta.get('category', 'Resources')}] {meta.get('title', note_path.stem)}"):
                col_n1, col_n2 = st.columns([3, 1])
                with col_n1:
                    st.markdown(f"**Category:** <span class='{cat_class}'>{meta.get('category')}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Created:** `{meta.get('created', 'N/A')}`")
                    st.markdown(f"**Tags:** {', '.join(['#' + t for t in meta.get('tags', [])])}")
                with col_n2:
                    st.markdown(f"**ID:** `{note_id[:8]}`")
                    st.markdown(f"**Links Count:** `{len(meta.get('links', []))}`")
                    
                st.markdown("---")
                st.markdown(body, unsafe_allow_html=True)
                
                # --- CONNECTED NOTES & BACKLINKS SECTION ---
                st.markdown("---")
                st.markdown("##### 🔗 Connected Knowledge Network")
                col_link_out, col_link_in = st.columns(2)
                
                # Outgoing Links
                with col_link_out:
                    st.markdown("**Outgoing Links →**")
                    out_links = meta.get("links", [])
                    if not out_links:
                        st.caption("No outgoing links")
                    else:
                        for l in out_links:
                            target_id = l.get("id")
                            sim_pct = int(l.get("similarity", 0) * 100)
                            target_info = id_to_meta.get(target_id, {})
                            target_title = target_info.get("title", target_id[:8])
                            target_cat = target_info.get("category", "Resources")
                            t_cat_class = f"badge-{target_cat.lower()}"
                            st.markdown(
                                f"• <span class='{t_cat_class}'>{target_cat}</span> **{target_title}** `{sim_pct}% match`",
                                unsafe_allow_html=True
                            )

                # Incoming Backlinks
                with col_link_in:
                    st.markdown("**← Incoming Backlinks**")
                    backlinks = get_backlinks(note_id)
                    if not backlinks:
                        st.caption("No incoming backlinks")
                    else:
                        for bl in backlinks:
                            sim_pct = int(bl.get("similarity", 0) * 100)
                            bl_cat = bl.get("category", "Resources")
                            bl_cat_class = f"badge-{bl_cat.lower()}"
                            st.markdown(
                                f"• <span class='{bl_cat_class}'>{bl_cat}</span> **{bl.get('title')}** `{sim_pct}% match`",
                                unsafe_allow_html=True
                            )

                st.divider()
                col_btn1, col_btn2 = st.columns([1, 1])
                
                # --- EDIT FORM ---
                with col_btn1:
                    with st.expander("✏️ Edit Note"):
                        with st.form(f"edit_form_{note_id}"):
                            edit_title = st.text_input("Title", value=meta.get("title", ""))
                            edit_cat = st.selectbox(
                                "Category",
                                config.PARA_CATEGORIES,
                                index=config.PARA_CATEGORIES.index(meta.get("category", "Resources")) if meta.get("category") in config.PARA_CATEGORIES else 2
                            )
                            edit_tags_str = st.text_input("Tags (comma separated)", value=", ".join(meta.get("tags", [])))
                            edit_body = st.text_area("Body Content", value=body, height=160)
                            
                            submit_edit = st.form_submit_button("Save Changes", use_container_width=True)
                            if submit_edit:
                                parsed_tags = [t.strip() for t in edit_tags_str.split(",") if t.strip()]
                                update_note(
                                    note_id=note_id,
                                    new_title=edit_title,
                                    new_category=edit_cat,
                                    new_tags=parsed_tags,
                                    new_body=edit_body
                                )
                                st.toast("Note updated successfully!", icon="✅")
                                st.rerun()
                                
                # --- DELETE BUTTON WITH CONFIRMATION ---
                with col_btn2:
                    with st.expander("🗑️ Delete Note"):
                        confirm_del = st.checkbox("Confirm deletion", key=f"del_chk_{note_id}")
                        if st.button("Delete Permanently", key=f"del_btn_{note_id}", type="primary", disabled=not confirm_del):
                            delete_note(note_id)
                            st.toast("Note deleted everywhere!", icon="🗑️")
                            st.rerun()
                            
        # Pagination Controls Footer
        if total_pages > 1:
            st.write("")
            col_pg1, col_pg2, col_pg3 = st.columns([1, 2, 1])
            with col_pg1:
                if st.button("◀ Previous Page", disabled=st.session_state.library_page <= 1):
                    st.session_state.library_page -= 1
                    st.rerun()
            with col_pg3:
                if st.button("Next Page ▶", disabled=st.session_state.library_page >= total_pages):
                    st.session_state.library_page += 1
                    st.rerun()
