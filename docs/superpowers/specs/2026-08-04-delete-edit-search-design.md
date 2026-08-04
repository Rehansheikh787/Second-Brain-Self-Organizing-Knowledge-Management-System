# Delete/Edit Notes & Advanced Search — Design Spec

**Date:** 2026-08-04  
**Status:** Approved  
**Scope:** Full note lifecycle management (Delete, Edit with auto-relocate) + Tag chips, Full-Text search, and Pagination in Streamlit Knowledge Library.

## Summary

Introduce complete CRUD capabilities and enhanced navigation for Second Brain notes:
1. Note Deletion with cascading cleanup (`raw/`, `wiki/`, `embeddings.npz`, cross-note links, and `graph.json`).
2. Note Editing (Title, Category, Tags, Body) with automatic file relocation when Category changes.
3. Interactive Tag Filter Chips, Full-Text Search across titles + tags + bodies, and Page-based Pagination (5/10/20 notes per page) in the Streamlit Knowledge Library tab.

## 1. Notes Management Module (`manage_notes.py`)

A new module to encapsulate deletion and edit workflows safely outside the Streamlit UI layer.

### `delete_note(note_id: str) -> bool`
- **Find and remove wiki note**: Search `wiki/**/*.md` for matching `id` in frontmatter (or filename `id.md`). Unlink/delete file.
- **Remove raw capture**: Delete `raw/<note_id>.json` if it exists.
- **Clean stored embeddings**: Load `embeddings.npz`, filter out `note_id` from `ids` and `vectors`, save updated `.npz`.
- **Clean bidirectional links**: Scan all remaining wiki notes. If any note's `links` metadata references `note_id`, remove that entry and rewrite frontmatter.
- **Refresh Knowledge Graph**: Call `export_graph()` to update `graph.json` and `static/graph_data.js`.
- **Return**: `True` if successfully deleted, `False` if note not found.

### `update_note(note_id: str, new_title: str, new_category: str, new_tags: list[str], new_body: str) -> Path`
- **Locate existing wiki note**: Read frontmatter and current file path.
- **Update metadata & body**: Update `title`, `category`, `tags`, preserving `id`, `created`, `source_type`, `links`, `embedding_version`.
- **Category relocation**:
  - If `new_category` != `old_category`, ensure `wiki/<new_category>/` exists.
  - Move file from `wiki/<old_category>/<note_id>.md` to `wiki/<new_category>/<note_id>.md`.
  - Delete old empty category folder if empty.
- **Re-link & update graph**: Call `link_all_notes()` to update embeddings/links and `export_graph()`.
- **Return**: New `Path` of the updated wiki note.

## 2. Knowledge Library UI Enhancements (`app.py`)

### Tag Chips & Multi-Tag Filtering
- Extract all unique tags across all wiki notes.
- Display a multi-select or tag button container in `tab_library`.
- Filter notes where selected tags subset matches note tags.

### Enhanced Full-Text Search
- Filter query matches against: `title` (case-insensitive), `tags` list, and `body` markdown content.

### Pagination
- Add page size selector (`st.selectbox("Notes per page", [5, 10, 20], index=1)`).
- Add page switcher (`< Previous` / `Next >` buttons with current page state stored in `st.session_state`).

### In-App Edit & Delete Controls
- Inside each note expander card:
  - **Edit Form**: Expander/Pop-up form allowing modifications to Title, Category radio (`Projects`, `Areas`, `Resources`, `Archives`), Tags input (comma-separated), and Body text area. Clicking "Save Changes" invokes `update_note(...)` and triggers `st.rerun()`.
  - **Delete Button**: "🗑️ Delete Note" button with a confirmation checkbox ("Confirm deletion"). Clicking triggers `delete_note(note_id)` and `st.rerun()`.

## 3. Unit Tests (`tests/test_manage_notes.py`)

- `test_delete_note_removes_files_and_links()`: Creates 2 linked notes, deletes 1, verifies wiki note, raw capture, embedding ID, cross-link in remaining note, and graph.json edge are removed.
- `test_update_note_relocates_category()`: Creates note in `Resources`, updates category to `Projects`, verifies file moves to `wiki/Projects/<id>.md` and metadata is updated.

## Out of Scope
- Trash/Restore bin (deletions are immediate).
- User authentication / multi-user scoping.
