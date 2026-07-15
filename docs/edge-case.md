# Second Brain — Edge Cases & Corner Scenarios

This document outlines potential edge cases, failure modes, and corner scenarios for the Second Brain system based on the `Architecture.md` and `Implementation-plan.md` specifications, along with how the system mitigates them.

---

## 1. Capture Layer (`capture.py`)

| Scenario | System Behavior / Mitigation |
| :--- | :--- |
| **Empty or Whitespace Input** | Raises `ValueError`. Validated before any hashing or ID generation. |
| **Duplicate Content** | Computes SHA-256 hash of content. Scans `raw/` directory. If collision found, raises `DuplicateError` and rejects capture. |
| **Invalid Source Type** | CLI arguments restricted to `["note", "link", "file"]`. Raises `ValueError` if bypassed programmatically. |
| **File Not Found** | If `source_type == "file"` but path is invalid, raises `FileNotFoundError`. |
| **File Encoding Errors** | Tries to read file as `utf-8`. If `UnicodeDecodeError` occurs, falls back to `latin-1` to salvage the text. |
| **Binary/Huge Files** | *Unmitigated Edge Case:* The script reads the entire file into memory. A massive or pure binary file (like a large PDF or video) might cause a memory crash or pollute the LLM later. Should be limited manually by the user. |

---

## 2. Intelligence Layer (`classify.py` / Groq LLM)

| Scenario | System Behavior / Mitigation |
| :--- | :--- |
| **LLM Returns Invalid JSON** | `llm_client.py` catches `JSONDecodeError` and retries up to 2 times. If it still fails, raises `ValueError` and skips the note for that run. |
| **LLM Wraps JSON in Markdown** | (e.g., \`\`\`json ... \`\`\`) Stripped manually via string manipulation before passing to `json.loads()`. |
| **Invalid Category Returned** | If the LLM returns a category outside `["Projects", "Areas", "Resources", "Archives"]`, the system logs a warning and defaults to `"Resources"`. |
| **Missing Fields in JSON** | `dict.get()` with fallbacks is used (e.g., defaults to `"Untitled"` for summary, `[]` for tags). |
| **Groq API Rate Limits (HTTP 429)** | Exponential backoff implemented (1s → 2s → 4s). If exhausted, skips note. Batch classification pauses 2 seconds between requests proactively. |
| **Network Timeout / 500 Error** | Caught by generic Exception handler, logs error, skips note, and continues batch processing. |

---

## 3. Linking Layer (`link.py` / sentence-transformers)

| Scenario | System Behavior / Mitigation |
| :--- | :--- |
| **Empty Knowledge Base** | Returns `0` links gracefully. Loading handles missing `embeddings.npz` by returning empty arrays. |
| **Self-Linking** | `exclude_id` parameter in `find_similar()` explicitly prevents a note from linking to itself. |
| **Duplicate Links** | Checked via `existing_links` set. Bidirectional links are safely merged without duplicating array entries. |
| **Manual Edits to Wiki Content** | *Edge Case:* The current logic checks `if note_id in existing_set` to reuse embeddings. If a user manually edits a note's text in `wiki/` but keeps the same ID, the embedding will **not** automatically recompute unless the `id` is purged from `embeddings.npz`. |
| **First Run (No Internet)** | `SentenceTransformer` requires internet to download the ~80MB model on the very first run. Will throw an error if offline. Once downloaded, works 100% locally. |

---

## 4. Graph Visualization (`build_graph.py` & Cytoscape.js)

| Scenario | System Behavior / Mitigation |
| :--- | :--- |
| **Orphan Nodes** | Notes with 0 links are still added to the `nodes` array in `graph.json` and will render as disconnected nodes in the graph. |
| **Cyclic Links** | Handled natively by Cytoscape.js. Duplicate identical edges are stripped by the `seen_edges` set in `build_graph.py`. |
| **Massive Node Count** | Cytoscape.js can comfortably handle up to ~2000 nodes. Performance will degrade beyond this. The `cose` layout algorithm might take several seconds to stabilize for very large graphs. |
| **Missing Frontmatter Fields** | Gracefully defaults to `"Untitled"`, `"Resources"`, etc., during `graph.json` generation so the UI doesn't crash. |

---

## 5. Query & UI Layer (`ask.py` & Streamlit)

| Scenario | System Behavior / Mitigation |
| :--- | :--- |
| **Query Against Empty Brain** | If no embeddings exist, returns fallback: *"I don't have any notes yet. Capture some knowledge first!"* |
| **Unrelated Query** | If highest cosine similarity is `< 0.40`, RAG is aborted. Returns *"I don't have notes closely related..."* and suggests the 3 closest topics instead of hallucinating. |
| **Context Window Overflow** | If the top-K notes are extremely long, Groq's token limit might be exceeded. Mitigated by truncating each note's context contribution to `body[:500]` characters. |
| **LLM Hallucination** | System prompt explicitly enforces: *"Answer the user's question using ONLY the provided notes."* |
| **Missing Graph Files at Startup** | Streamlit UI checks if `graph.html` and `graph.json` exist. If missing, shows a polite `st.warning` instead of throwing a stack trace. |
| **Deployment Storage** | **Critical:** On Streamlit Cloud, the file system is ephemeral. Any notes captured or classified via the deployed app will be **deleted** when the container sleeps/reboots. The system is designed as a deployed *read-only* viewer, with captures happening locally. |
