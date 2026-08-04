# Delete/Edit Notes & Advanced Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement full CRUD lifecycle (Delete with cascade cleanup, Edit with category relocation) and advanced library navigation (Tag chips, full-text search, pagination) in Streamlit.

**Architecture:** Create `manage_notes.py` helper module with clean file/embedding/link/graph cascade operations; build unit tests in `tests/test_manage_notes.py`; update `app.py` Knowledge Library tab with tag chips, pagination, edit forms, and delete confirmation controls.

**Tech Stack:** Python 3.10+, Streamlit, NumPy, PyYAML, pytest.

## Global Constraints

- **Cascade integrity:** Deleting a note must clean raw JSON, wiki Markdown, embeddings array, graph edges, and cross-references in remaining notes.
- **Relocation safety:** Editing a category must move the file on disk (`wiki/<old>/` -> `wiki/<new>/`) and preserve UUID filename and all other frontmatter fields.
- **UI stability:** All updates/deletions must call `export_graph()` and trigger `st.rerun()` so UI state remains synchronized.

---

### Task 1: Notes Management Module (`manage_notes.py`) & Unit Tests

**Files:**
- Create: `manage_notes.py`
- Create: `tests/test_manage_notes.py`

**Interfaces:**
- Consumes: `config`, `utils`, `link`, `build_graph`
- Produces:
  - `delete_note(note_id: str) -> bool`
  - `update_note(note_id: str, new_title: str, new_category: str, new_tags: list[str], new_body: str) -> Path`

- [ ] **Step 1: Write failing unit tests in `tests/test_manage_notes.py`**

```python
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np

def test_delete_note_cascades_cleanup():
    from manage_notes import delete_note
    from utils import write_frontmatter, read_frontmatter

    with tempfile.TemporaryDirectory() as tmpdir:
        wiki_dir = Path(tmpdir) / "wiki"
        raw_dir = Path(tmpdir) / "raw"
        wiki_dir.mkdir()
        raw_dir.mkdir()
        res_dir = wiki_dir / "Resources"
        res_dir.mkdir()
        embeddings_file = Path(tmpdir) / "embeddings.npz"
        graph_json = Path(tmpdir) / "graph.json"

        # Create raw capture and 2 wiki notes linked to each other
        raw_path = raw_dir / "note1.json"
        raw_path.write_text(json.dumps({"id": "note1", "content": "test"}))

        note1_path = res_dir / "note1.md"
        write_frontmatter(note1_path, {
            "id": "note1", "title": "Note 1", "category": "Resources",
            "tags": ["test"], "links": [{"id": "note2", "similarity": 0.8}]
        }, "Body 1")

        note2_path = res_dir / "note2.md"
        write_frontmatter(note2_path, {
            "id": "note2", "title": "Note 2", "category": "Resources",
            "tags": ["test"], "links": [{"id": "note1", "similarity": 0.8}]
        }, "Body 2")

        np.savez(embeddings_file, ids=np.array(["note1", "note2"]), vectors=np.zeros((2, 384)))

        with patch("manage_notes.WIKI_DIR", wiki_dir), \
             patch("utils.WIKI_DIR", wiki_dir), \
             patch("manage_notes.RAW_DIR", raw_dir), \
             patch("utils.RAW_DIR", raw_dir), \
             patch("manage_notes.EMBEDDINGS_FILE", embeddings_file), \
             patch("manage_notes.GRAPH_JSON", graph_json), \
             patch("build_graph.WIKI_DIR", wiki_dir), \
             patch("build_graph.GRAPH_JSON", graph_json):

            success = delete_note("note1")
            assert success is True
            assert not note1_path.exists()
            assert not raw_path.exists()

            # Verify note2 links no longer reference note1
            meta2, _ = read_frontmatter(note2_path)
            linked_ids = [l["id"] for l in meta2.get("links", [])]
            assert "note1" not in linked_ids

            # Verify embeddings array updated
            stored_data = np.load(embeddings_file)
            assert "note1" not in stored_data["ids"]

def test_update_note_relocates_category():
    from manage_notes import update_note
    from utils import write_frontmatter, read_frontmatter

    with tempfile.TemporaryDirectory() as tmpdir:
        wiki_dir = Path(tmpdir) / "wiki"
        wiki_dir.mkdir()
        res_dir = wiki_dir / "Resources"
        res_dir.mkdir()
        embeddings_file = Path(tmpdir) / "embeddings.npz"
        graph_json = Path(tmpdir) / "graph.json"

        note1_path = res_dir / "note1.md"
        write_frontmatter(note1_path, {
            "id": "note1", "title": "Old Title", "category": "Resources",
            "tags": ["old"]
        }, "Old Body")

        np.savez(embeddings_file, ids=np.array(["note1"]), vectors=np.zeros((1, 384)))

        with patch("manage_notes.WIKI_DIR", wiki_dir), \
             patch("utils.WIKI_DIR", wiki_dir), \
             patch("manage_notes.EMBEDDINGS_FILE", embeddings_file), \
             patch("manage_notes.GRAPH_JSON", graph_json), \
             patch("link.WIKI_DIR", wiki_dir), \
             patch("link.EMBEDDINGS_FILE", embeddings_file), \
             patch("build_graph.WIKI_DIR", wiki_dir), \
             patch("build_graph.GRAPH_JSON", graph_json):

            new_path = update_note("note1", "New Title", "Projects", ["new"], "New Body")
            assert new_path.exists()
            assert "Projects" in str(new_path)
            assert not note1_path.exists()

            meta, body = read_frontmatter(new_path)
            assert meta["title"] == "New Title"
            assert meta["category"] == "Projects"
            assert meta["tags"] == ["new"]
            assert body == "New Body"
```

- [ ] **Step 2: Run pytest to verify failure**

Run: `uv run pytest tests/test_manage_notes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'manage_notes'`

- [ ] **Step 3: Implement `manage_notes.py`**

```python
"""Note lifecycle operations — delete note with cascading cleanup, update note with category relocation."""

import logging
from pathlib import Path
import numpy as np

from config import WIKI_DIR, RAW_DIR, EMBEDDINGS_FILE, GRAPH_JSON, PARA_CATEGORIES
from utils import list_wiki_notes, read_frontmatter, write_frontmatter, load_json, save_json
from link import load_embeddings, save_embeddings, link_all_notes
from build_graph import export_graph

logger = logging.getLogger(__name__)


def delete_note(note_id: str) -> bool:
    """
    Delete note by ID across wiki/, raw/, embeddings.npz, cross-links, and graph.json.
    Returns True if deleted, False if note not found.
    """
    wiki_notes = list_wiki_notes()
    target_path = None
    
    for note_path in wiki_notes:
        meta, _ = read_frontmatter(note_path)
        if meta.get("id") == note_id or note_path.stem == note_id:
            target_path = note_path
            break
            
    if not target_path or not target_path.exists():
        logger.warning(f"Note {note_id} not found for deletion.")
        return False
        
    # 1. Delete wiki markdown file
    target_path.unlink()
    
    # 2. Delete raw capture JSON if exists
    raw_path = RAW_DIR / f"{note_id}.json"
    if raw_path.exists():
        raw_path.unlink()
        
    # 3. Clean embeddings array
    stored_ids, stored_vectors = load_embeddings()
    if note_id in stored_ids:
        idx = stored_ids.index(note_id)
        new_ids = [nid for i, nid in enumerate(stored_ids) if i != idx]
        new_vectors = np.delete(stored_vectors, idx, axis=0) if stored_vectors.shape[0] > 0 else stored_vectors
        save_embeddings(new_ids, new_vectors)
        
    # 4. Clean cross-references in remaining wiki notes
    remaining_notes = list_wiki_notes()
    for note_path in remaining_notes:
        meta, body = read_frontmatter(note_path)
        links = meta.get("links", [])
        filtered_links = [l for l in links if l.get("id") != note_id]
        if len(filtered_links) != len(links):
            meta["links"] = filtered_links
            write_frontmatter(note_path, meta, body)
            
    # 5. Export fresh graph.json
    export_graph(GRAPH_JSON)
    logger.info(f"Successfully deleted note {note_id} everywhere.")
    return True


def update_note(
    note_id: str,
    new_title: str,
    new_category: str,
    new_tags: list[str],
    new_body: str
) -> Path:
    """
    Update note title, category, tags, and body.
    Relocates file to wiki/<new_category>/<note_id>.md if category changes.
    Returns Path to updated note.
    """
    if new_category not in PARA_CATEGORIES:
        new_category = "Resources"
        
    wiki_notes = list_wiki_notes()
    target_path = None
    target_meta = {}
    
    for note_path in wiki_notes:
        meta, body = read_frontmatter(note_path)
        if meta.get("id") == note_id or note_path.stem == note_id:
            target_path = note_path
            target_meta = meta
            break
            
    if not target_path or not target_path.exists():
        raise FileNotFoundError(f"Note {note_id} not found for editing.")
        
    old_category = target_meta.get("category", "Resources")
    
    # Update metadata
    target_meta["title"] = str(new_title).strip()[:120]
    target_meta["category"] = new_category
    target_meta["tags"] = [str(t).strip() for t in new_tags if str(t).strip()]
    
    dest_dir = WIKI_DIR / new_category
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{note_id}.md"
    
    # If category changed and path is different, remove old file
    if target_path.resolve() != dest_path.resolve():
        if target_path.exists():
            target_path.unlink()
            
    write_frontmatter(dest_path, target_meta, new_body.strip())
    
    # Refresh embeddings, linking & graph
    link_all_notes()
    export_graph(GRAPH_JSON)
    
    logger.info(f"Successfully updated note {note_id} at {dest_path}")
    return dest_path
```

- [ ] **Step 4: Run pytest to verify all manage_notes tests pass**

Run: `uv run pytest tests/test_manage_notes.py -v`
Expected: PASS

- [ ] **Step 5: Commit Task 1**

```bash
git add manage_notes.py tests/test_manage_notes.py
git commit -m "feat: add manage_notes module for delete with cascade cleanup and note editing with category relocation"
```

---

### Task 2: Streamlit UI Integration (`app.py`) — Delete, Edit, Tag Chips, Search, & Pagination

**Files:**
- Modify: `app.py:252-301`

**Interfaces:**
- Consumes: `delete_note`, `update_note` from `manage_notes`
- Produces: Enhanced Knowledge Library tab with Tag filtering, Full-Text search, Page Pagination, Edit forms, and Delete buttons.

- [ ] **Step 1: Add Tag Chips, Search, Pagination, Edit & Delete forms to `tab_library` in `app.py`**

In `app.py`, update `tab_library` section:
1. Extract unique tags across all notes and display tag selection chips (`st.multiselect("Filter Tags", all_tags)`).
2. Filter notes by category, selected tags, and full-text keyword matching (title + tags + body).
3. Implement page pagination controls (e.g. 5/10/20 per page).
4. For each displayed note expander:
   - Render "✏️ Edit" expander containing an inline form (Title input, Category selectbox, Tags text input, Body text area, "Save Changes" button -> calls `update_note` & `st.rerun()`).
   - Render "🗑️ Delete" popover/checkbox confirmation ("Confirm deletion", "Delete Permanently" button -> calls `delete_note` & `st.rerun()`).

- [ ] **Step 2: Run pytest to ensure no syntax/import errors**

Run: `uv run pytest -v`
Expected: PASS

- [ ] **Step 3: Commit Task 2**

```bash
git add app.py
git commit -m "feat: add tag chips, pagination, edit forms, and delete confirmation to Streamlit Knowledge Library"
```

---

### Task 3: Full Integration Verification Across Test Suite

- [ ] **Step 1: Execute full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 2: Verify git status clean**

Run: `git status`
Confirm clean working tree.
