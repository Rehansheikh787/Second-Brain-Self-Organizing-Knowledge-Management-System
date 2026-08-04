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
    filename = Path(uploaded_file.name).name
    content, ftype = extract_file_content(uploaded_file)
    capture_id = capture(content, source_type="file", original_filename=filename)
    
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
