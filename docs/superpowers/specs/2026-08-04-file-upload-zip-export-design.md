# File & Media Upload & ZIP Export — Design Spec

**Date:** 2026-08-04  
**Status:** Approved  
**Scope:** Multi-format file ingestion (Text, Markdown, Code, PDF, Images, Videos) + Downloadable ZIP archive backup for Second Brain.

## Summary

Expand Second Brain ingestion and backup capabilities:
1. Support multi-format file uploads in sidebar (`.txt`, `.md`, `.py`, `.js`, `.json`, `.pdf`, `.png`, `.jpg`, `.jpeg`, `.mp4`, `.mov`, `.webm`).
2. Text extraction for documents (PDF via `pypdf`, plain text / code via UTF-8 decoder).
3. Media asset storage (`static/uploads/<filename>`) with embedded Markdown visual cards (`![Image](/static/uploads/file)`) auto-classified into PARA wiki notes.
4. One-click "📦 Download Wiki Backup" in Streamlit using in-memory ZIP streaming (`zipfile.ZipFile` -> `BytesIO`).

## 1. Export & Import Module (`export_import.py`)

### `extract_file_content(uploaded_file) -> tuple[str, str]`
- **Text & Code files** (`.txt`, `.md`, `.py`, `.js`, `.json`, `.yaml`, `.csv`, `.html`, `.css`): Read UTF-8 text string. Return `(content, "text")`.
- **PDF Documents** (`.pdf`): Use `pypdf.PdfReader` to iterate through pages and aggregate extracted text string. Return `(extracted_text, "pdf")`.
- **Images & Videos** (`.png`, `.jpg`, `.jpeg`, `.webp`, `.svg`, `.mp4`, `.mov`, `.webm`):
  - Save raw bytes to `config.STATIC_DIR / "uploads" / <filename>`.
  - Return Markdown snippet referencing saved media asset path, e.g.:
    `![Uploaded Image: filename](/static/uploads/filename)\n\nMedia file captured on UTC timestamp.`
  - Return `(media_markdown, "media")`.

### `ingest_uploaded_file(uploaded_file) -> str`
- Calls `extract_file_content(...)`.
- Passes extracted content string to `capture(content, source_type="file")`.
- Triggers pipeline: `classify_all_pending()`, `link_all_notes()`, `export_graph()`.
- Returns capture ID.

### `generate_zip_backup() -> bytes`
- Create an in-memory `BytesIO` buffer.
- Open `zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED)`.
- Walk `config.WIKI_DIR` recursively and add all `.md` files under `wiki/` prefix.
- Walk `config.STATIC_DIR / "uploads"` if it exists and add files under `uploads/` prefix.
- Close zip and return `buffer.getvalue()`.

## 2. Streamlit UI Updates (`app.py`)

### Sidebar File Uploader
- Below Quick Capture text area:
  ```python
  st.subheader("📁 Upload File or Media")
  uploaded_file = st.file_uploader(
      "Choose file",
      type=["txt", "md", "py", "js", "json", "pdf", "png", "jpg", "jpeg", "mp4", "mov", "webm"]
  )
  ```
- When user clicks "Upload & Process", call `ingest_uploaded_file(uploaded_file)`, show success toast, and `st.rerun()`.

### Download Backup Button
- In sidebar or Knowledge Library header:
  ```python
  st.download_button(
      label="📦 Download Wiki Backup (ZIP)",
      data=generate_zip_backup(),
      file_name="second_brain_wiki_backup.zip",
      mime="application/zip",
      use_container_width=True
  )
  ```

## 3. Unit Tests (`tests/test_export_import.py`)

- `test_extract_file_content_text()`: Verifies plain text read.
- `test_extract_file_content_media_saves_to_uploads()`: Mocks binary image, verifies file saved in `static/uploads/` and markdown image snippet returned.
- `test_generate_zip_backup_creates_valid_zip()`: Creates dummy wiki & uploads files, runs `generate_zip_backup()`, verifies returned bytes form a valid ZIP file containing expected files.

## Out of Scope
- OCR for images (text extraction from images).
