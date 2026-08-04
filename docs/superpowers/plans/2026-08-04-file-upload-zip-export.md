# File & Media Upload & ZIP Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement multi-format file and media upload ingestion (Text, Markdown, Code, PDF, Images, Videos) and downloadable ZIP backup generation for Second Brain.

**Architecture:** Install `pypdf` dependency; create `export_import.py` helper module for document extraction, media storage in `static/uploads/`, and in-memory ZIP streaming; build unit tests in `tests/test_export_import.py`; update `app.py` sidebar with file uploader and backup download button.

**Tech Stack:** Python 3.10+, pypdf, Streamlit, PyYAML, pytest.

## Global Constraints

- **Storage Location:** Media asset uploads (`.png`, `.jpg`, `.jpeg`, `.mp4`, `.mov`, `.webm`) must save under `config.STATIC_DIR / "uploads"` and be referenced via relative URL `/static/uploads/<filename>`.
- **ZIP Preservation:** Generated ZIP backups must preserve the `wiki/` directory hierarchy (`Projects/`, `Areas/`, `Resources/`, `Archives/`) and include all uploaded media assets.
- **Pipeline Integration:** Ingesting an uploaded file must trigger the standard 3-phase pipeline (`classify_all_pending()`, `link_all_notes()`, `export_graph()`).

---

### Task 1: Install `pypdf` Dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add `pypdf` to `requirements.txt`**

Add `pypdf>=3.0.0` to `requirements.txt`.

- [ ] **Step 2: Install `pypdf` into virtual environment**

Run: `uv pip install pypdf`
Expected: Successfully installed pypdf.

- [ ] **Step 3: Commit Task 1**

```bash
git add requirements.txt
git commit -m "build: add pypdf dependency for PDF document text extraction"
```

---

### Task 2: Export & Import Module (`export_import.py`) & Unit Tests

**Files:**
- Create: `export_import.py`
- Create: `tests/test_export_import.py`

**Interfaces:**
- Consumes: `config`, `capture`, `classify`, `link`, `build_graph`, `pypdf`
- Produces:
  - `extract_file_content(uploaded_file) -> tuple[str, str]`
  - `ingest_uploaded_file(uploaded_file) -> str`
  - `generate_zip_backup() -> bytes`

- [ ] **Step 1: Write failing unit tests in `tests/test_export_import.py`**

```python
import io
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

def test_extract_file_content_text():
    from export_import import extract_file_content

    mock_file = MagicMock()
    mock_file.name = "sample.txt"
    mock_file.read.return_value = b"Hello world text"

    content, file_type = extract_file_content(mock_file)
    assert content == "Hello world text"
    assert file_type == "text"

def test_extract_file_content_media_saves_file():
    from export_import import extract_file_content

    with tempfile.TemporaryDirectory() as tmpdir:
        static_dir = Path(tmpdir) / "static"
        static_dir.mkdir()

        mock_file = MagicMock()
        mock_file.name = "photo.png"
        mock_file.read.return_value = b"\x89PNG fake data"

        with patch("export_import.STATIC_DIR", static_dir):
            content, file_type = extract_file_content(mock_file)

            saved_path = static_dir / "uploads" / "photo.png"
            assert saved_path.exists()
            assert file_type == "media"
            assert "photo.png" in content
            assert "![" in content

def test_generate_zip_backup_creates_valid_zip():
    from export_import import generate_zip_backup

    with tempfile.TemporaryDirectory() as tmpdir:
        wiki_dir = Path(tmpdir) / "wiki"
        res_dir = wiki_dir / "Resources"
        res_dir.mkdir(parents=True)
        (res_dir / "note.md").write_text("Note content")

        static_dir = Path(tmpdir) / "static"
        uploads_dir = static_dir / "uploads"
        uploads_dir.mkdir(parents=True)
        (uploads_dir / "photo.png").write_text("fake image")

        with patch("export_import.WIKI_DIR", wiki_dir), \
             patch("export_import.STATIC_DIR", static_dir):

            zip_bytes = generate_zip_backup()
            assert isinstance(zip_bytes, bytes)
            assert len(zip_bytes) > 0

            # Verify zip archive contents
            with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
                names = zf.namelist()
                assert any("note.md" in n for n in names)
                assert any("photo.png" in n for n in names)
```

- [ ] **Step 2: Run pytest to verify failure**

Run: `uv run pytest tests/test_export_import.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'export_import'`

- [ ] **Step 3: Implement `export_import.py`**

```python
"""Multi-format file extraction, media asset management, and ZIP backup generator."""

import io
import logging
import zipfile
from pathlib import Path

from config import WIKI_DIR, STATIC_DIR
from capture import capture
from classify import classify_all_pending
from link import link_all_notes
from build_graph import export_graph

logger = logging.getLogger(__name__)


def extract_file_content(uploaded_file) -> tuple[str, str]:
    """
    Extract text or save media asset from an uploaded file object.
    Returns tuple: (content_string, format_type)
    """
    filename = Path(uploaded_file.name).name
    ext = Path(filename).suffix.lower()
    
    media_exts = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".mp4", ".mov", ".webm"}
    
    if ext in media_exts:
        uploads_dir = STATIC_DIR / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        file_path = uploads_dir / filename
        
        file_bytes = uploaded_file.read()
        file_path.write_bytes(file_bytes)
        
        rel_path = f"static/uploads/{filename}"
        if ext in {".mp4", ".mov", ".webm"}:
            markdown_content = f"### 🎬 Captured Video: {filename}\n\n<video src=\"/{rel_path}\" controls width=\"100%\"></video>"
        else:
            markdown_content = f"### 🖼️ Captured Image: {filename}\n\n![{filename}](/{rel_path})"
            
        return markdown_content, "media"
        
    elif ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(uploaded_file)
            text_parts = []
            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    text_parts.append(txt)
            content = "\n\n".join(text_parts).strip()
            if not content:
                content = f"PDF File captured: {filename} (no extractable text)"
            return f"# PDF Document: {filename}\n\n{content}", "pdf"
        except Exception as e:
            logger.warning(f"Failed PDF extraction via pypdf: {e}")
            return f"# PDF Document: {filename}\n\n(Text extraction unavailable)", "pdf"
            
    else:
        # Plain text, Markdown, or Code file
        try:
            file_bytes = uploaded_file.read()
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1")
        return text.strip(), "text"


def ingest_uploaded_file(uploaded_file) -> str:
    """
    Extract uploaded file content, run capture, auto-classify, link, and export graph.
    Returns capture ID.
    """
    content, ftype = extract_file_content(uploaded_file)
    capture_id = capture(content, source_type="file")
    
    classify_all_pending()
    link_all_notes()
    export_graph()
    
    logger.info(f"Successfully ingested uploaded file {uploaded_file.name} (ID: {capture_id})")
    return capture_id


def generate_zip_backup() -> bytes:
    """
    Generate an in-memory ZIP archive of wiki/ directory and static/uploads/.
    Returns zip bytes.
    """
    buffer = io.BytesIO()
    
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add wiki Markdown files
        if WIKI_DIR.exists():
            for file_path in WIKI_DIR.glob("**/*"):
                if file_path.is_file():
                    arcname = Path("wiki") / file_path.relative_to(WIKI_DIR)
                    zf.write(file_path, arcname=arcname)
                    
        # Add static/uploads media files
        uploads_dir = STATIC_DIR / "uploads"
        if uploads_dir.exists():
            for file_path in uploads_dir.glob("**/*"):
                if file_path.is_file():
                    arcname = Path("uploads") / file_path.relative_to(uploads_dir)
                    zf.write(file_path, arcname=arcname)
                    
    return buffer.getvalue()
```

- [ ] **Step 4: Run pytest to verify all test_export_import tests pass**

Run: `uv run pytest tests/test_export_import.py -v`
Expected: PASS

- [ ] **Step 5: Commit Task 2**

```bash
git add export_import.py tests/test_export_import.py
git commit -m "feat: add export_import module for document text extraction, media uploads, and ZIP backup generation"
```

---

### Task 3: Streamlit UI Integration (`app.py`) — File Uploader & Download Backup Button

**Files:**
- Modify: `app.py:125-165`

**Interfaces:**
- Consumes: `ingest_uploaded_file`, `generate_zip_backup` from `export_import`
- Produces: Sidebar file uploader and downloadable ZIP backup button.

- [ ] **Step 1: Add File Uploader & Download Backup button to sidebar in `app.py`**

1. Import `ingest_uploaded_file` and `generate_zip_backup` from `export_import`.
2. In sidebar, add file uploader widget:
   ```python
   st.subheader("📁 Upload File / Media")
   up_file = st.file_uploader(
       "Choose file",
       type=["txt", "md", "py", "js", "json", "pdf", "png", "jpg", "jpeg", "mp4", "mov", "webm"],
       key="sidebar_file_uploader"
   )
   if st.button("Ingest Uploaded File", use_container_width=True):
       if up_file:
           with st.spinner("Extracting & processing pipeline..."):
               cid = ingest_uploaded_file(up_file)
           st.success(f"Ingested: {cid[:8]}")
           st.rerun()
   ```
3. In sidebar bottom or header, add ZIP download button:
   ```python
   st.download_button(
       label="📦 Download Wiki Backup (ZIP)",
       data=generate_zip_backup(),
       file_name="second_brain_backup.zip",
       mime="application/zip",
       use_container_width=True
   )
   ```

- [ ] **Step 2: Run pytest to ensure all test suites pass**

Run: `uv run pytest -v`
Expected: PASS

- [ ] **Step 3: Commit Task 3**

```bash
git add app.py
git commit -m "feat: integrate sidebar file uploader and ZIP backup download button into Streamlit app"
```

---

### Task 4: Full Integration Verification Across Test Suite

- [ ] **Step 1: Execute full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 2: Verify git status clean**

Run: `git status`
Confirm clean working tree.
